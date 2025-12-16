import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ProxyMutualInformationLipschitz:
    """
    Computes a smoothed proxy for mutual or conditional mutual information using
    Lipschitz-smoothed joint distributions.
    For unconditional MI: I(S;O)
    For conditional MI: I(S;O|A)
    """

    def __init__(self, datapath=None, data=None):
        """
        Initializes the proxy estimator with a dataset path.

        Parameters:
        -----------
        datapath : str
            Path to the CSV dataset file.
        """
        if (datapath is None and data is None) or (datapath is not None and data is not None):
            raise Exception("Usage: Should pass either datapath or data itself")
        if datapath is not None:
            self.dataset = pd.read_csv(datapath)
        else:
            self.dataset = data

    def calculate(self, s_col, o_col, a_col=None, alpha=1.0):
        """
        Calculate Lipschitz-smoothed mutual information I(S;O) or conditional MI I(S;O|A).

        Parameters:
        -----------
        s_col : str
            Sensitive attribute (S).
        o_col : str
            Outcome attribute (O).
        a_col : str, optional
            Attribute to condition on (A).
        alpha : float
            Smoothing parameter α (e.g., 1, log(n), sqrt(n), n).

        Returns:
        --------
        float : Approximated (conditional) mutual information score.
        """
        start_time = time.time()
        self.dataset.replace(["NA", "N/A", ""], pd.NA, inplace=True)

        columns = [s_col, o_col] + ([a_col] if a_col else [])
        self.dataset.dropna(inplace=True, subset=columns)

        # Encode columns
        s = LabelEncoder().fit_transform(self.dataset[s_col])
        o = LabelEncoder().fit_transform(self.dataset[o_col])
        if a_col:
            a = LabelEncoder().fit_transform(self.dataset[a_col])
        n = len(s)

        if not a_col:
            # Unconditional case: I(S;O)
            num_s, num_o = np.max(s) + 1, np.max(o) + 1
            joint = np.zeros((num_s, num_o))
            for si, oi in zip(s, o):
                joint[si, oi] += 1

            smoothed_joint = (joint + alpha) / (n + alpha * num_s * num_o)
            p_s = smoothed_joint.sum(axis=1, keepdims=True)
            p_o = smoothed_joint.sum(axis=0, keepdims=True)
            ratio = np.divide(smoothed_joint, p_s @ p_o, out=np.ones_like(smoothed_joint), where=smoothed_joint > 0)
            mi = np.sum(smoothed_joint * np.log2(ratio))

        else:
            # Conditional case: I(S;O | A)
            num_s, num_o, num_a = np.max(s) + 1, np.max(o) + 1, np.max(a) + 1
            total_mi = 0.0

            for a_val in range(num_a):
                indices = a == a_val
                count = np.sum(indices)
                if count == 0:
                    continue

                sub_s = s[indices]
                sub_o = o[indices]
                joint = np.zeros((num_s, num_o))
                for si, oi in zip(sub_s, sub_o):
                    joint[si, oi] += 1

                smoothed_joint = (joint + alpha) / (count + alpha * num_s * num_o)
                p_s = smoothed_joint.sum(axis=1, keepdims=True)
                p_o = smoothed_joint.sum(axis=0, keepdims=True)
                ratio = np.divide(smoothed_joint, p_s @ p_o, out=np.ones_like(smoothed_joint), where=smoothed_joint > 0)
                total_mi += (count / n) * np.sum(smoothed_joint * np.log2(ratio))

            mi = total_mi
            scaled_mi = np.log1p(mi) / np.log1p(1) if mi > 0 else 0  # base scaling
            scaled_mi = scaled_mi * (1 + (1 / (1 + mi)))  # compress larger values
            mi = scaled_mi

        elapsed_time = time.time() - start_time
        print(f"Lipschitz MI (α={alpha}): Proxy Mutual Information between '{s_col}' and '{o_col}'"
              + (f" conditioned on '{a_col}'" if a_col else "")
              + f": {mi:.4f}. Calculation took {elapsed_time:.3f} seconds.")
        return round(mi, 4)
