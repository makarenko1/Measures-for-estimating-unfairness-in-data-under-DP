import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class TupleContribution:
    """
    Implementation of the Tuple Contribution metric
    for the case where the protected set, response set, and admissible set
    each contain at most one attribute (a single column).

    Computes unsigned marginal differences (MD) for given fairness criteria,
    optionally adds Laplace noise for differential privacy, and returns
    the sum of top-k MD values across criteria.
    """

    def __init__(self, datapath=None, data=None):
        if (datapath is None and data is None) or (datapath is not None and data is not None):
            raise ValueError("Usage: pass exactly one of datapath or data")
        self.dataset = pd.read_csv(datapath) if datapath is not None else data.copy()

    def calculate(self, fairness_criteria, k=500, epsilon=None, encode_and_clean=False):
        """
        Compute the top-k unsigned marginal differences for each fairness criterion.

        Each criterion consists of two or three column names:
            (protected, response) or (protected, response, admissible).

        If epsilon is given, Laplace noise is added for privacy.
        Prints a short summary and returns the resulting contribution value.
        """
        start_time = time.time()

        contribution = 0.0
        min_a_count = None

        for criterion in fairness_criteria:
            if len(criterion) not in [2, 3]:
                raise ValueError("Invalid input")

            protected_col, response_col, admissible_col = (criterion[0], criterion[1],
                                                           None if len(criterion) == 2 else criterion[2])
            cols = [protected_col, response_col] + ([admissible_col] if admissible_col is not None else [])
            if encode_and_clean:
                df = self._encode_and_clean(self.dataset, cols)
            else:
                df = self.dataset

            if admissible_col is not None:
                temp = int(df[admissible_col].value_counts().min())
                min_a_count = min(min_a_count, temp) if min_a_count is not None else temp

            if k is None:
                k = len(df)

            if admissible_col is None:
                values = self._calculate_unconditional_helper(df[protected_col].to_numpy(), df[response_col].to_numpy())
            else:
                values = self._calculate_conditional_helper(df[protected_col].to_numpy(), df[response_col].to_numpy(),
                                                            df[admissible_col].to_numpy())

            top_k = sorted(values, reverse=True)[:min(k, len(values))]
            contribution += float(np.sum(top_k))

        if epsilon is not None:
            n = len(self.dataset)
            if min_a_count is not None:
                print(f"min_a_count: {min_a_count}")
                sensitivity = len(fairness_criteria) * ((9 * k / n) + 0.5)
            else:
                sensitivity = len(fairness_criteria) * ((3 * k / n) + 0.5)
            contribution = contribution + np.random.laplace(loc=0.0, scale=sensitivity / float(epsilon))

        elapsed_time = time.time() - start_time
        print(f"Tuple Contribution for fairness criteria {fairness_criteria}: {contribution:.4f} with data size: "
              f"{len(self.dataset)} and epsilon: {epsilon if epsilon is not None else 'infinity'}. Calculation took "
              f"{elapsed_time:.3f} seconds.")
        return contribution

    @staticmethod
    def _encode_and_clean(df, cols):
        """
        Cleans and label-encode selected columns.

        Drops rows with missing values and converts categorical entries
        into integer codes for the specified columns.

        Returns
        -------
        pandas.DataFrame
            A cleaned and encoded copy of the input data.
        """
        df = df.replace(["NA", "N/A", ""], pd.NA).dropna(subset=cols).copy()
        for c in cols:
            df[c] = LabelEncoder().fit_transform(df[c])
        return df

    @staticmethod
    def _calculate_unconditional_helper(protected_col_values, response_col_values):
        """
        Compute unsigned marginal differences for all observed (S,O) pairs:
            MD(s,o) = |P(S,O) − P(S)P(O)|
        Returns MD values for unique tuples.
        """
        protected_col_values = np.asarray(protected_col_values, dtype=np.int64)
        response_col_values = np.asarray(response_col_values, dtype=np.int64)
        n_protected, n_response = protected_col_values.max() + 1, response_col_values.max() + 1
        N = len(protected_col_values)

        flat = protected_col_values * n_response + response_col_values
        C = np.bincount(flat, minlength=n_protected * n_response).reshape(n_protected, n_response).astype(float)# counts
        p_protected_response = C / N
        p_protected = p_protected_response.sum(axis=1, keepdims=True)  # [n_protected,1]
        p_response = p_protected_response.sum(axis=0, keepdims=True)   # [1,n_response]
        MD = np.abs(p_protected_response - p_protected @ p_response)   # MD matrix

        # Extract only cells that appear in the data to define tuples + multiplicities
        protected_idx, response_idx = np.nonzero(C)  # multiplicity for each unique (s,o)
        return MD[protected_idx, response_idx]

    @staticmethod
    def _calculate_conditional_helper(protected_col_values, response_col_values, admissible_col_values):
        """
        Compute unsigned marginal differences for all observed (S,O,A) tuples:
            MD(s,o|a) = |P(S,O|A) − P(S|A)P(O|A)|
        Returns MD values for unique tuples.
        """
        protected_col_values = np.asarray(protected_col_values, dtype=np.int64)
        response_col_values = np.asarray(response_col_values, dtype=np.int64)
        admissible_col_values = np.asarray(admissible_col_values, dtype=np.int64)
        n_protected, n_response, n_admissible = (protected_col_values.max() + 1, response_col_values.max() + 1,
                                                 admissible_col_values.max() + 1)

        flat = admissible_col_values * (n_protected * n_response) + protected_col_values * n_response + response_col_values
        C = np.bincount(flat, minlength=n_admissible * n_protected * n_response).reshape(
            n_admissible, n_protected, n_response).astype(float)  # counts per (a,s,o)

        N_admissible = C.sum(axis=(1, 2))  # total per a
        mask_a = N_admissible > 0

        p_protected_response = np.zeros_like(C)
        p_protected_response[mask_a] = C[mask_a] / N_admissible[mask_a, None, None]  # P(S,O|a)
        p_protected = p_protected_response.sum(axis=2, keepdims=True)                # P(S|a)
        p_response = p_protected_response.sum(axis=1, keepdims=True)                 # P(O|a)
        p_admissible = np.zeros_like(N_admissible, dtype=float)
        p_admissible[mask_a] = N_admissible[mask_a] / float(admissible_col_values.shape[0])  # P(A)
        MD = np.abs(p_protected_response - p_protected * p_response)                 # residuals per (a,s,o)
        MD_weighted = MD * p_admissible[:, None, None]

        # Extract only observed cells to define tuples + multiplicities
        admissible_idx, protected_idx, response_idx = np.nonzero(C)  # multiplicity for each unique (s,o,a)
        return MD_weighted[admissible_idx, protected_idx, response_idx]
