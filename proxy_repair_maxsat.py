import time
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from z3 import Bool, Or, Not, Optimize, sat


class ProxyRepairMaxSat:
    """
    ProxyRepairMaxSat

    Implementation of the proxy repair metric based on a MaxSAT formulation.
    This version assumes that the protected, response, and admissible sets
    each contain at most one attribute (a single column).

    The method converts each fairness criterion into a set of logical
    clauses (soft and hard) representing multivalued dependency (MVD)
    constraints, then solves a weighted MaxSAT problem to estimate the
    minimal data repair needed for fairness. Laplace noise can be added
    for differential privacy.
    """

    def __init__(self, datapath=None, data=None):
        if (datapath is None and data is None) or (datapath is not None and data is not None):
            raise ValueError("Usage: Should pass either datapath or data itself")
        if datapath is not None:
            self.dataset = pd.read_csv(datapath)
        else:
            self.dataset = data

    def calculate(
        self,
        fairness_criteria,
        epsilon=None,
        encode_and_clean=False,
        chunk_size: int | None = 100,
        soft_clauses_percentage: float = 1.0,
    ):
        """
        Computes the proxy repair measure using a MaxSAT solver.

        Each criterion is defined by two or three column names:
          (protected, response) or (protected, response, admissible).

        The method encodes tuples as Boolean variables, builds soft and hard
        3-CNF clauses for each criterion, and solves the combined MaxSAT
        optimization.

        Parameters
        ----------
        fairness_criteria : list[list[str]]
        epsilon : float or None
        encode_and_clean : bool
        chunk_size : int or None, default 100
            If not None, data is processed in chunks of this size (per criterion)
            and the repair is summed across chunks (approximate).
        soft_clauses_percentage : float, default 1.0
            Fraction of soft clauses to add to the solver.
            1.0 = use all soft clauses (original behavior),
            0.5 = use ~50% of soft clauses (randomly sampled).

        Returns
        -------
        float
            The estimated repair value (possibly noised).
        """
        # Clamp percentage to [0, 1]
        soft_clauses_percentage = max(0.0, min(1.0, soft_clauses_percentage))

        start_time = time.time()
        total_repair = 0

        for criterion in fairness_criteria:
            if len(criterion) not in [2, 3]:
                raise ValueError("Invalid input: each criterion must have 2 or 3 columns")

            protected_col, response_col, admissible_col = (
                criterion[0],
                criterion[1],
                None if len(criterion) == 2 else criterion[2],
            )
            cols = [protected_col, response_col] + (
                [admissible_col] if admissible_col is not None else []
            )

            # Prepare dataframe for this criterion
            if encode_and_clean:
                df_base = self._encode_and_clean(self.dataset, cols)
            else:
                df_base = self.dataset[cols]

            n_rows = len(df_base)

            if chunk_size is None:
                # Single chunk: full data
                chunk_repair = self._repair_for_df(
                    df_base, cols, admissible_col, soft_clauses_percentage
                )
                total_repair += chunk_repair
            else:
                # Multiple chunks
                for start in range(0, n_rows, chunk_size):
                    end = start + chunk_size
                    df_chunk = df_base.iloc[start:end]
                    if df_chunk.empty:
                        continue
                    chunk_repair = self._repair_for_df(
                        df_chunk, cols, admissible_col, soft_clauses_percentage
                    )
                    total_repair += chunk_repair

        if epsilon is not None:
            sensitivity = 2 * len(fairness_criteria)
            total_repair = total_repair + np.random.laplace(
                loc=0, scale=sensitivity / epsilon
            )

        elapsed_time = time.time() - start_time
        print(
            f"Repair MaxSAT: Proxy Repair MaxSAT for fairness criteria {fairness_criteria}: "
            f"{total_repair:.4f} with data size: {len(self.dataset)} and epsilon: "
            f"{epsilon if epsilon is not None else 'infinity'}. "
            f"Calculation took {elapsed_time:.3f} seconds."
        )
        return total_repair

    def _repair_for_df(
        self,
        df: pd.DataFrame,
        cols,
        admissible_col,
        soft_clauses_percentage: float,
    ) -> int:
        """
        Solve the MaxSAT problem for a single dataframe (full data or a chunk)
        and return the repair size for that chunk.
        """
        D = self._add_id(df, cols)
        if not D:
            return 0

        opt = Optimize()
        D_star = self._conversion_to_solving_general_3cnf(
            D, admissible_col, opt, soft_clauses_percentage
        )

        if opt.check() != sat:
            print("No satisfying assignment found for a chunk/criterion.")
            return 0

        model = opt.model()
        D_set = set(D)
        DR = {t for t in D_star if model.evaluate(Bool(f"x_{t}"), model_completion=True)}
        return len(D_set.symmetric_difference(DR))

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
    def _add_id(df, cols):
        """
        Adds a local ID per distinct (S,O[,A]) and returns tuples for
        the MVD S, O, ID | A:

          - with admissible_col: (S, O, A, ID)
          - without admissible_col: (S, O, ID)
        """
        df_local = df[cols].copy()
        df_local["ID"] = df_local.groupby(cols).cumcount() + 1

        s_col, o_col = cols[0], cols[1]

        if len(cols) == 3:
            a_col = cols[2]
            # Use DataFrame -> tuples instead of explicit Python list comprehension join
            return list(
                df_local[[s_col, o_col, a_col, "ID"]].itertuples(index=False, name=None)
            )
        else:
            return list(
                df_local[[s_col, o_col, "ID"]].itertuples(index=False, name=None)
            )

    @staticmethod
    def _clause_key(t1, t2, t3):
        """
        Generates a unique key for a clause defined by three tuples.

        Used internally to avoid duplicate logical clauses.
        """
        return tuple(sorted([f"x_{t1}", f"x_{t2}", f"x_{t3}"]))

    @staticmethod
    def _conversion_to_solving_general_3cnf(D, admissible_col, opt, soft_clauses_percentage: float):
        """
        Converts a dataset into a 3-CNF MaxSAT formulation.

        Constructs:
          - D_star: the set of all tuple combinations satisfying MVD structure
                    (via pandas joins rather than pure Python list comprehensions)
          - Soft clauses: one per tuple, encouraging consistency with the data
          - Hard clauses: 3-CNF constraints enforcing fairness dependencies

        Only a fraction `soft_clauses_percentage` of soft clauses are added.

        Returns
        -------
            D_star (set of tuples)
        """
        if not D:
            return set()

        var_cache = {}

        def v(t):
            if t not in var_cache:
                var_cache[t] = Bool(f"x_{t}")
            return var_cache[t]

        D_set = set(D)
        arr = np.array(D, dtype=object)

        if admissible_col is not None:
            # D elements: (s, o, a, id)
            dfD = pd.DataFrame(arr, columns=["s", "o", "a", "id"])

            # ---- jdb construction (your D_star_df) ----
            dstar_frames = []
            for a_val, group in dfD.groupby("a"):
                S_df = group[["s"]].drop_duplicates()
                O_df = group[["o", "id"]].drop_duplicates()
                S_df["key"] = 1
                O_df["key"] = 1
                cart = S_df.merge(O_df, on="key").drop("key", axis=1)
                cart["a"] = a_val
                dstar_frames.append(cart)

            D_star_df = (
                pd.concat(dstar_frames, ignore_index=True)
                if dstar_frames
                else pd.DataFrame(columns=["s", "o", "id", "a"])
            )

            # D_star = set of tuples (this is the variable universe)
            D_star = set(D_star_df[["s", "o", "a", "id"]].itertuples(index=False, name=None))

            # ---- soft clauses over jdb ----
            if soft_clauses_percentage > 0.0:
                for t in D_star:
                    if soft_clauses_percentage < 1.0 and np.random.random() >= soft_clauses_percentage:
                        continue
                    x_t = v(t)
                    opt.add_soft(x_t if t in D_set else Not(x_t), weight=1)

            # ---- hard clauses over jdb x jdb ----
            # Iterate per a over jdb
            for a_val, group in D_star_df.groupby("a"):
                S_vals = group["s"].unique()
                O_pairs = group[["o", "id"]].drop_duplicates().to_records(index=False)
                O_pairs = list(O_pairs)

                for (o1, i1), (o2, i2) in combinations(O_pairs, 2):
                    for s1, s2 in combinations(S_vals, 2):
                        t1 = (s1, o1, a_val, i1)
                        t2 = (s2, o2, a_val, i2)
                        t3 = (s1, o2, a_val, i2)
                        opt.add(Or(Not(v(t1)), Not(v(t2)), v(t3)))

            return D_star

        else:
            # D elements: (s, o, id)
            dfD = pd.DataFrame(arr, columns=["s", "o", "id"])

            # ---- jdb construction: S × (o,id) ----
            S_df = dfD[["s"]].drop_duplicates()
            O_df = dfD[["o", "id"]].drop_duplicates()
            S_df["key"] = 1
            O_df["key"] = 1
            D_star_df = S_df.merge(O_df, on="key").drop("key", axis=1)

            D_star = set(D_star_df[["s", "o", "id"]].itertuples(index=False, name=None))

            # ---- soft clauses ----
            if soft_clauses_percentage > 0.0:
                for t in D_star:
                    if soft_clauses_percentage < 1.0 and np.random.random() >= soft_clauses_percentage:
                        continue
                    x_t = v(t)
                    opt.add_soft(x_t if t in D_set else Not(x_t), weight=1)

            # ---- hard clauses over jdb x jdb ----
            S_vals = D_star_df["s"].unique()
            O_pairs = D_star_df[["o", "id"]].drop_duplicates().to_records(index=False)
            O_pairs = list(O_pairs)

            for (o1, i1), (o2, i2) in combinations(O_pairs, 2):
                for s1, s2 in combinations(S_vals, 2):
                    t1 = (s1, o1, i1)
                    t2 = (s2, o2, i2)
                    t3 = (s1, o2, i2)
                    opt.add(Or(Not(v(t1)), Not(v(t2)), v(t3)))

            return D_star
