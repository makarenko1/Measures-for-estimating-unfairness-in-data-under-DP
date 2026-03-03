import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Literal

import numpy as np
import pandas as pd
import torch
from diffprivlib.models import RandomForestClassifier
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D as MplLine2D
from matplotlib.ticker import LogLocator, FuncFormatter
from opacus import PrivacyEngine
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mutual_information import MutualInformation
from proxy_mutual_information_tvd import ProxyMutualInformationTVD
from proxy_repair_maxsat import ProxyRepairMaxSat
from tuple_contribution import TupleContribution
from unused_measures.proxy_mutual_information_privbayes import ProxyMutualInformationPrivbayes

adult_criteria = [["education-num", "income>50K"], ["sex", "income>50K"], ["race", "income>50K"],
                  ["sex", "income>50K", "hours-per-week"]]

census_criteria = [["HEALTH", "INCTOT", "EDUC"], ["HEALTH", "OCC", "EDUC"], ["HEALTH", "MARST", "AGE"],
                   ["HEALTH", "INCTOT", "AGE"]]

stackoverflow_criteria = [["Country", "RemoteWork", "Employment"], ["Age", "PurchaseInfluence", "OrgSize"],
                          ["Country", "MainBranch", "YearsCodePro"], ["Age", "MainBranch", "EdLevel"]]

compas_criteria = [["race", "is_recid", "age_cat"], ["sex", "is_recid", "priors_count"],
                   ["race", "decile_score", "c_charge_degree"], ["sex", "v_decile_score", "age_cat"]]

healthcare_criteria = [["race", "complications", "age_group"], ["smoker", "complications", "age_group"],
                       ["race", "income", "county"], ["smoker", "income", "num_children"]]

datasets = {
    "Adult": {
        "path": "data/adult.csv",
        "criteria": adult_criteria,
    },
    "IPUMS-CPS": {
        "path": "data/census.csv",
        "criteria": census_criteria,
    },
    "Stackoverflow": {
        "path": "data/stackoverflow.csv",
        "criteria": stackoverflow_criteria,
    },
    "Compas": {
        "path": "data/compas.csv",
        "criteria": compas_criteria,
    },
    "Healthcare": {
        "path": "data/healthcare.csv",
        "criteria": healthcare_criteria,
    },
}

datasets_shortened = {
    "Adult": {
        "path": "data/adult.csv",
        "criteria": adult_criteria,
    },
    "Stackoverflow": {
        "path": "data/stackoverflow.csv",
        "criteria": stackoverflow_criteria,
    },
    "Compas": {
        "path": "data/compas.csv",
        "criteria": compas_criteria,
    }
}

# Formatter: round to 3 decimals, then strip trailing zeros and dot
def _yfmt(y, pos):
    s = f"{y:.3f}"
    s = s.rstrip('0').rstrip('.')
    return s
y_formatter = FuncFormatter(_yfmt)


def get_sample(
    df: pd.DataFrame,
    n: int,
    fairness_criteria: List[List[str]],
    p_start: float = 0.5,
    p_step: float = 0.1,
    p_cap: float = 1.0,
    mode: Literal["raise", "cap", "fallback"] = "cap",
) -> pd.DataFrame:
    """
    Sampling with min-count>=2 constraint.

    Primary rule (strong): build a *combined key* from all admissible columns and ensure
    no combined-key bucket appears with size 1 in the sample (activate buckets via pairs).

    If infeasible:
      - mode="raise": raise ValueError
      - mode="cap": set n = eligible_total (max feasible) and proceed
      - mode="fallback": fall back to per-column constraint (each admissible column value
        appearing in the sample must appear >=2), which can allow larger n.

    Notes:
      - With combined-key fairness, max feasible is:
            eligible_total = sum(bucket_size for buckets with size>=2)
      - Singleton combined-key buckets are never selectable under the strong rule.
    """
    if n < 0:
        raise ValueError("Sampling error: n must be nonnegative")
    if n == 0:
        return df.iloc[0:0].copy()

    rng = np.random.default_rng()

    # Extract admissible columns
    admissible_cols = []
    for crit in fairness_criteria:
        if len(crit) not in (2, 3):
            raise ValueError("Sampling error: each criterion must have length 2 or 3")
        if len(crit) == 3 and crit[2] is not None:
            admissible_cols.append(crit[2])

    admissible_cols = sorted(set(admissible_cols))

    # No admissible columns => plain sample
    if not admissible_cols:
        if n > len(df):
            raise ValueError("Sampling error: n larger than df (no replacement)")
        return df.sample(n=n, replace=False).reset_index(drop=True)

    for c in admissible_cols:
        if c not in df.columns:
            raise ValueError(f"Sampling error: Admissible column '{c}' not in df.columns")

    if n == 1:
        raise ValueError("Sampling error: Impossible to sample n=1 while requiring min group size >= 2")

    # ----- Strong mode: combined key -----
    def _combined_key_series(frame: pd.DataFrame) -> pd.Series:
        if len(admissible_cols) == 1:
            return frame[admissible_cols[0]]
        return frame[admissible_cols].astype(object).apply(lambda r: tuple(r.values.tolist()), axis=1)

    key = _combined_key_series(df)
    groups = df.groupby(key, dropna=False, sort=False).groups
    eligible_total = sum(len(idxs) for idxs in groups.values() if len(idxs) >= 2)

    if n > eligible_total:
        if mode == "cap":
            n = eligible_total
        elif mode == "fallback":
            return _get_sample_per_column_min(df, n, admissible_cols, rng)
        else:
            raise ValueError(
                f"Sampling error: can sample at most {eligible_total} rows from buckets with size>=2, requested n={n}."
            )

    # Build disjoint pairs per combined-key bucket
    all_pairs = []
    for k_val, idxs in groups.items():
        idxs = np.array(list(idxs))
        if len(idxs) < 2:
            continue
        rng.shuffle(idxs)
        m = (len(idxs) // 2) * 2
        for t in range(0, m, 2):
            all_pairs.append((int(idxs[t]), int(idxs[t + 1]), k_val))

    # Pair-sampling loop
    selected = set()
    p = float(p_start)
    pair_pool = all_pairs.copy()

    while len(selected) + 2 <= n and pair_pool:
        rng.shuffle(pair_pool)
        before = len(selected)
        new_pool = []
        for i, j, k_val in pair_pool:
            if len(selected) + 2 > n:
                new_pool.append((i, j, k_val))
                continue
            if i in selected or j in selected:
                continue
            if rng.random() < p:
                selected.add(i)
                selected.add(j)
            else:
                new_pool.append((i, j, k_val))
        pair_pool = new_pool

        p = min(p + p_step, p_cap)

        if len(selected) == before and abs(p - p_cap) < 1e-12:
            for i, j, _k in pair_pool:
                if len(selected) + 2 > n:
                    break
                if i in selected or j in selected:
                    continue
                selected.add(i)
                selected.add(j)
            break

    # Top up from already-present combined-key buckets
    if len(selected) < n:
        present_keys = set(key.loc[list(selected)].tolist())

        candidates = []
        for k_val in present_keys:
            for idx in groups.get(k_val, []):
                idx = int(idx)
                if idx not in selected:
                    candidates.append(idx)

        rng.shuffle(candidates)
        need = n - len(selected)
        selected.update(candidates[:need])

        if len(selected) != n:
            # Still feasible overall, but we didn't activate enough buckets.
            # Deterministically activate more buckets to finish.
            # (Add one new pair from any unused bucket, then top up again.)
            unused_pairs = [(i, j, k) for (i, j, k) in all_pairs if (i not in selected and j not in selected)]
            rng.shuffle(unused_pairs)
            for i, j, k_val in unused_pairs:
                if len(selected) + 2 > n:
                    break
                selected.add(i)
                selected.add(j)

                # refresh present keys and candidates
                present_keys.add(k_val)
                for idx in groups.get(k_val, []):
                    idx = int(idx)
                    if idx not in selected:
                        candidates.append(idx)

                if len(selected) >= n:
                    break

            rng.shuffle(candidates)
            need = n - len(selected)
            selected.update(candidates[:need])

            if len(selected) != n:
                raise ValueError(
                    f"Sampling error: feasible overall (eligible_total={eligible_total}) but couldn't construct n={n} "
                    f"under combined-key rule (got {len(selected)})."
                )

    out = df.loc[list(selected)].copy()

    # Sanity check per admissible column: values that appear must appear >=2
    for a_col in admissible_cols:
        vc = out[a_col].value_counts(dropna=False)
        if len(vc) > 0 and int(vc.min()) < 2:
            raise RuntimeError(f"Sampling error: singleton group in '{a_col}'")

    if len(out) != n:
        raise RuntimeError(f"Sampling error: expected n={n}, got {len(out)}")

    return out.sample(frac=1).reset_index(drop=True)


def _get_sample_per_column_min(
    df: pd.DataFrame,
    n: int,
    admissible_cols: List[str],
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Fallback sampler: ensures for EACH admissible column independently,
    any value that appears in the sample appears at least twice.

    This is weaker than combined-key fairness, but often matches "fairness per attribute".
    Greedy constructive algorithm:
      - maintain a pool of eligible rows; when we pick a row, we also pick a "buddy" row
        that matches it on all admissible cols where possible.
      - if exact buddy on all cols not available, we enforce per-column min2 by tracking counts
        and preferentially picking rows that avoid creating singletons.
    """
    if n > len(df):
        raise ValueError("Sampling error: n larger than df (no replacement)")

    if n == 1:
        raise ValueError("Sampling error: Impossible to sample n=1 while requiring min group size >= 2")

    # Precompute value->indices per column
    col_to_val_idxs = {}
    for c in admissible_cols:
        m = {}
        for v, idxs in df.groupby(c, dropna=False, sort=False).groups.items():
            m[v] = np.array(list(idxs), dtype=int)
        col_to_val_idxs[c] = m

    # Values that are singletons in any column can never appear (would force count 1)
    forbidden = set()
    for c in admissible_cols:
        for v, idxs in col_to_val_idxs[c].items():
            if len(idxs) < 2:
                forbidden.update(int(i) for i in idxs)

    eligible_idxs = [i for i in range(len(df)) if i not in forbidden]
    if n > len(eligible_idxs):
        raise ValueError(
            f"Sampling error (fallback): can sample at most {len(eligible_idxs)} rows after removing per-column singletons, requested n={n}."
        )

    # Start by sampling pairs of the same combined vector of admissible values (best effort)
    key = df[admissible_cols].astype(object).apply(lambda r: tuple(r.values.tolist()), axis=1)
    groups = df.iloc[eligible_idxs].groupby(key.iloc[eligible_idxs], dropna=False, sort=False).groups

    selected = set()
    # activate by pairs from exact combined buckets first
    group_items = list(groups.items())
    rng.shuffle(group_items)
    for _k, idxs in group_items:
        idxs = list(idxs)
        if len(selected) + 2 > n:
            break
        if len(idxs) >= 2:
            rng.shuffle(idxs)
            a, b = int(idxs.pop()), int(idxs.pop())
            if a in selected or b in selected:
                continue
            selected.add(a); selected.add(b)

    # Top up one-by-one but avoid creating singletons per column
    counts = {c: {} for c in admissible_cols}
    def add_counts(i: int):
        for c in admissible_cols:
            v = df.at[i, c]
            counts[c][v] = counts[c].get(v, 0) + 1

    for i in selected:
        add_counts(i)

    remaining = [i for i in eligible_idxs if i not in selected]
    rng.shuffle(remaining)

    def would_create_singleton(i: int) -> bool:
        # If adding i would introduce a value with count 0 and we might never add a second,
        # we mitigate by requiring that value has at least one other remaining row.
        for c in admissible_cols:
            v = df.at[i, c]
            if counts[c].get(v, 0) == 0:
                # Need at least one other row left with same value
                idxs = col_to_val_idxs[c].get(v, np.array([], dtype=int))
                exists_other = any((int(j) not in selected and int(j) != i) for j in idxs)
                if not exists_other:
                    return True
        return False

    while len(selected) < n and remaining:
        i = int(remaining.pop())
        if i in selected:
            continue
        if would_create_singleton(i):
            continue
        selected.add(i)
        add_counts(i)

    if len(selected) != n:
        raise ValueError(
            f"Sampling error (fallback): couldn't reach n={n} while maintaining per-column min2 (got {len(selected)})."
        )

    out = df.loc[list(selected)].copy()

    # Sanity check per column min2
    for c in admissible_cols:
        vc = out[c].value_counts(dropna=False)
        if len(vc) > 0 and int(vc.min()) < 2:
            raise RuntimeError(f"Sampling error (fallback): singleton group in '{c}'")

    return out.sample(frac=1).reset_index(drop=True)


def create_plot_0(
        epsilon: Optional[float] = 10.0,
        num_tuples: int = 10000,
        repetitions: int = 10,
        outfile: str = "plots/plot0.png",
):
    """Values of measures for IPUMS-CPS (for criterions with more unfairness we expect higher values)."""

    def demographic_parity_with_rf(
            df: pd.DataFrame,
            protected_col: str,
            response_col: str,
            admissible_col: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Demographic parity (or conditional statistical parity for conditional criteria) with a RandomForest
        classifier. Returns (dp_gap, rf_loss), where rf_loss is 1 - accuracy on test.
        """
        if df.empty:
            return 0.0, 0.0
        if response_col not in df.columns or protected_col not in df.columns:
            return 0.0, 0.0
        if admissible_col is not None and admissible_col not in df.columns:
            admissible_col = None
        feature_cols = [c for c in df.columns if c != response_col]
        if not feature_cols:
            return 0.0, 0.0

        X = df[feature_cols].to_numpy(dtype=float)
        y = df[response_col].to_numpy()
        s = df[protected_col].to_numpy()
        if admissible_col is not None:
            a = df[admissible_col].to_numpy()
        else:
            a = None
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        unique_labels, counts = np.unique(y, return_counts=True)
        if len(unique_labels) <= 1 or counts.min() < 2:
            stratify_arg = None
        else:
            stratify_arg = y
        if a is None:
            X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                X_scaled, y, s,
                test_size=0.3,
                stratify=stratify_arg,
            )
            a_train, a_test = None, None
        else:
            X_train, X_test, y_train, y_test, s_train, s_test, a_train, a_test = train_test_split(
                X_scaled, y, s, a,
                test_size=0.3,
                stratify=stratify_arg,
            )

        clf = RandomForestClassifier(
            n_estimators=20,
            max_depth=15,
            shuffle=True,
            classes=[0, 1],
            epsilon=epsilon
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        positive_label = 1
        pos = (y_pred == positive_label).astype(float)
        rf_loss = 1.0 - accuracy_score(y_test, y_pred)

        if a is None:
            rates = pd.Series(pos).groupby(pd.Series(s_test)).mean()
            dp_gap = float(abs(rates.max() - rates.min())) if len(rates) > 0 else 0.0
        else:
            df_eval = pd.DataFrame({
                "pos": pos,
                "s": s_test,
                "a": a_test,
            })
            gaps = []
            weights = []
            for a_val, grp in df_eval.groupby("a"):
                rates = grp["pos"].groupby(grp["s"]).mean()
                dp_a = rates.max() - rates.min()
                gaps.append(dp_a)
                weights.append(len(grp))
            # Weighted average DP-gap:
            dp_gap = np.average(gaps, weights=weights) if gaps else 0.0

        return dp_gap, rf_loss

    def demographic_parity_with_nn(
            df: pd.DataFrame,
            protected_col: str,
            response_col: str,
            admissible_col: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Demographic parity (or conditional statistical parity for conditional criteria) with a small neural net trained
        with DP-SGD (via Opacus PrivacyEngine).

        Returns (dp_gap, nn_loss), where nn_loss is 1 - accuracy on test.
        """
        if df.empty:
            return 0.0, 0.0
        if response_col not in df.columns or protected_col not in df.columns:
            return 0.0, 0.0
        if admissible_col is not None and admissible_col not in df.columns:
            admissible_col = None
        feature_cols = [c for c in df.columns if c != response_col]
        if not feature_cols:
            return 0.0, 0.0

        X = df[feature_cols].to_numpy(dtype=float)
        y_raw = df[response_col].to_numpy()
        s = df[protected_col].to_numpy()
        a = df[admissible_col].to_numpy() if admissible_col is not None else None
        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw).astype(np.int64)
        num_classes = len(le.classes_)
        if num_classes < 2:
            return 0.0, 0.0

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        unique_labels, counts = np.unique(y_enc, return_counts=True)
        if len(unique_labels) <= 1 or counts.min() < 2:
            stratify_arg = None
        else:
            stratify_arg = y_enc

        if a is None:
            X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                X_scaled, y_enc, s,
                test_size=0.15,
                stratify=stratify_arg,
            )
            a_train = a_test = None
        else:
            X_train, X_test, y_train, y_test, s_train, s_test, a_train, a_test = train_test_split(
                X_scaled, y_enc, s, a,
                test_size=0.15,
                stratify=stratify_arg,
            )

        # ---- DP-SGD NN training ----
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        train_ds = TensorDataset(X_train_t, y_train_t)
        batch_size = min(100, len(train_ds))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        input_dim = X_train.shape[1]

        class SimpleNN(nn.Module):
            def __init__(self, d_in, d_hidden=32, num_classes=2):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_in, d_hidden),
                    nn.ReLU(),
                    nn.Linear(d_hidden, num_classes),
                )

            def forward(self, x):
                return self.net(x)

        model = SimpleNN(input_dim, d_hidden=32, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

        # Attach PrivacyEngine (DP-SGD)
        if epsilon is not None:
            max_grad_norm = 1.0
            target_epsilon = epsilon
            target_delta = 1e-5
            epochs = 150
            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                epochs=epochs,
                max_grad_norm=max_grad_norm,
            )
        else:
            epochs = 150
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

        model.eval()
        with torch.no_grad():
            logits_test = model(X_test_t.to(device))
            y_pred = torch.argmax(logits_test, dim=1).cpu().numpy()
        positive_label = 1
        pos = (y_pred == positive_label).astype(float)
        nn_loss = 1.0 - accuracy_score(y_test, y_pred)

        if a is None:
            rates = pd.Series(pos).groupby(pd.Series(s_test)).mean()
            dp_gap = float(abs(rates.max() - rates.min())) if len(rates) > 0 else 0.0
        else:
            df_eval = pd.DataFrame({
                "pos": pos,
                "s": s_test,
                "a": a_test,
            })
            gaps = []
            weights = []
            for a_val, grp in df_eval.groupby("a"):
                rates = grp["pos"].groupby(grp["s"]).mean()
                dp_a = rates.max() - rates.min()
                gaps.append(dp_a)
                weights.append(len(grp))
            # Weighted average DP-gap:
            dp_gap = np.average(gaps, weights=weights) if gaps else 0.0

        return dp_gap, nn_loss

    # ------------ main experiment ------------

    all_rows = []
    dp_values_rf = []
    dp_values_nn = []
    rf_losses_avg = []
    nn_losses_avg = []

    path = "data/adult.csv"
    criteria = adult_criteria  # assumed defined elsewhere

    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    for crit_idx, criterion in enumerate(criteria, start=1):
        df = pd.read_csv(path)
        data = _encode_and_clean(path, df.columns)
        n = min(num_tuples, len(data))

        sum_tvd = 0.0
        sum_repair = 0.0
        sum_tc = 0.0
        sum_dp_rf = 0.0
        sum_dp_nn = 0.0
        sum_loss_rf = 0.0
        sum_loss_nn = 0.0

        for rep in range(repetitions):
            sample = get_sample(df=data, n=n, fairness_criteria=[criterion])
            sample_measures = sample[criterion]

            tvd_proxy = ProxyMutualInformationTVD(data=sample_measures)
            sum_tvd += float(tvd_proxy.calculate([criterion], epsilon=epsilon))

            repair_proxy = ProxyRepairMaxSat(data=sample_measures)
            sum_repair += float(repair_proxy.calculate([criterion], epsilon=epsilon))

            tc_proxy = TupleContribution(data=sample_measures)
            sum_tc += float(tc_proxy.calculate([criterion], epsilon=epsilon))

            protected_col, response_col = criterion[0], criterion[1]
            admissible_col = criterion[2] if len(criterion) == 3 else None

            dp_rf, loss_rf = demographic_parity_with_rf(
                df=sample,
                protected_col=protected_col,
                response_col=response_col,
                admissible_col=admissible_col
            )
            dp_nn, loss_nn = demographic_parity_with_nn(
                df=sample,
                protected_col=protected_col,
                response_col=response_col,
                admissible_col=admissible_col,
            )

            sum_dp_rf += dp_rf
            sum_dp_nn += dp_nn
            sum_loss_rf += loss_rf
            sum_loss_nn += loss_nn

        tvd_avg = sum_tvd / repetitions
        repair_avg = sum_repair / repetitions
        tc_avg = sum_tc / repetitions
        dp_rf_avg = sum_dp_rf / repetitions
        dp_nn_avg = sum_dp_nn / repetitions
        rf_loss_avg = sum_loss_rf / repetitions
        nn_loss_avg = sum_loss_nn / repetitions

        all_rows.append([
            round(tvd_avg, 4),
            repair_avg,
            round(tc_avg, 4),
        ])
        dp_values_rf.append(dp_rf_avg)
        dp_values_nn.append(dp_nn_avg)
        rf_losses_avg.append(rf_loss_avg)
        nn_losses_avg.append(nn_loss_avg)

        print(
            f"Criterion {criterion}: "
            f"RF avg loss = {rf_loss_avg:.4f}, NN avg loss = {nn_loss_avg:.4f}"
        )

        print(
            f"Criterion {criterion}: "
            f"RF acc = {1 - rf_loss_avg:.4f}, NN acc = {1 - nn_loss_avg:.4f}"
        )

    num_criteria = len(criteria)
    x = np.arange(num_criteria)
    criterion_numbers = [str(i) for i in range(1, num_criteria + 1)]
    measure_labels = ["Proxy\nMutualInformationTVD", "Proxy\nRepairMaxSAT", "TupleContribution"]
    subplot_colors = ["tab:blue", "tab:orange", "tab:green"]

    # ---------- main three metrics ----------
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(max(8, num_criteria * 1.5), 3.5),
        sharex=True
    )
    all_rows_np = np.array(all_rows, dtype=float)

    for ax, mlabel, col_idx, color in zip(axes, measure_labels, [0, 1, 2], subplot_colors):
        vals = all_rows_np[:, col_idx]
        ax.bar(x, vals, color=color)
        ax.set_yscale('log')
        ax.set_title(mlabel)
        ax.set_xticks(x)
        ax.set_xticklabels(criterion_numbers)

    for ax in axes:
        ax.set_xlabel("criterion")

    plt.tight_layout(rect=[0, 0, 1, 0.9])

    dir_name = os.path.dirname(outfile)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.show()

    # ---------- demographic parity plots: RF and NN separately ----------
    plt.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.titlesize": 18,
    })

    # RF DP plot
    dp_outfile_rf = f"{outfile.split('.')[0]}_dp_rf.png"
    fig_rf, ax_rf = plt.subplots(figsize=(3.5, 3.5))
    dp_rf_np = np.array(dp_values_rf, dtype=float)
    ax_rf.bar(x, dp_rf_np, color="tab:purple")
    ax_rf.set_xticks(x)
    ax_rf.set_xticklabels(criterion_numbers)
    ax_rf.set_xlabel("criterion")
    fig_rf.suptitle("Demographic\nParity (RF)")

    plt.tight_layout()
    dp_dir = os.path.dirname(dp_outfile_rf)
    if dp_dir:
        os.makedirs(dp_dir, exist_ok=True)
    plt.savefig(dp_outfile_rf, dpi=600, bbox_inches="tight")
    plt.show()

    # NN DP plot
    dp_outfile_nn = f"{outfile.split('.')[0]}_dp_nn.png"
    fig_nn, ax_nn = plt.subplots(figsize=(3.5, 3.5))
    dp_nn_np = np.array(dp_values_nn, dtype=float)
    ax_nn.bar(x, dp_nn_np, color="tab:gray")
    ax_nn.set_xticks(x)
    ax_nn.set_xticklabels(criterion_numbers)
    ax_nn.set_xlabel("criterion")
    fig_nn.suptitle("Demographic\nParity (NN+DP)")

    plt.tight_layout()
    dp_dir_nn = os.path.dirname(dp_outfile_nn)
    if dp_dir_nn:
        os.makedirs(dp_dir_nn, exist_ok=True)
    plt.savefig(dp_outfile_nn, dpi=600, bbox_inches="tight")
    plt.show()


def create_plot_1(outfile: str="plots/plot1.png"):
    # --- config ---------------------------------------------------
    # Row order: MI, PrivBayes Original, PrivBayes with offset, TVD
    PROXIES = [
        ("Mutual\nInformation", "MI"),
        ("PrivBayes Proxy", "PRIV_ORIG"),
        ("PrivBayes Proxy\nwith offset", "PRIV_OFFSET"),
        ("TVD Proxy", "TVD"),
    ]

    PROXY_COLOR = {
        "MI": "#1f77b4",
        "PRIV_ORIG": "#ff7f0e",
        "PRIV_OFFSET": "#9467bd",
        "TVD": "#2ca02c",
    }

    # Bigger fonts everywhere
    TITLE_FS = 30  # column titles
    ROWLAB_FS = 28  # row (y-axis) labels
    TICK_FS = 28  # tick labels (bottom axis + y-ticks)
    ANNOT_FS = 22  # numbers above bars

    # bar layout
    BAR_SPACING = 0.55  # <--- tighter spacing (was effectively 0.65)
    BAR_WIDTH = 0.5

    # --- compute values
    vals = {k: {} for _, k in PROXIES}
    for ds_name, ds_config in datasets_shortened.items():
        path, criteria = ds_config["path"], ds_config["criteria"]
        mi_scores, priv_scores_orig, priv_scores_offset, tvd_scores = [], [], [], []
        for criterion in criteria:
            if len(criterion) == 3:
                s_col, o_col, a_col = criterion
            else:
                s_col, o_col = criterion
                a_col = None
            mi_scores.append(MutualInformation(datapath=path).calculate([criterion],
                                                                        encode_and_clean=True))
            priv_scores_orig.append(ProxyMutualInformationPrivbayes(datapath=path).calculate(
                s_col, o_col, a_col, add_offset=False
            ))
            priv_scores_offset.append(ProxyMutualInformationPrivbayes(datapath=path).calculate(
                s_col, o_col, a_col
            ))
            tvd_scores.append(ProxyMutualInformationTVD(datapath=path).calculate(
                [criterion], encode_and_clean=True
            ))
        vals["MI"][ds_name] = mi_scores
        vals["PRIV_ORIG"][ds_name] = priv_scores_orig
        vals["PRIV_OFFSET"][ds_name] = priv_scores_offset
        vals["TVD"][ds_name] = tvd_scores

    # labels per dataset
    ds_labels = {
        ds_name: [str(i) for i in range(1, len(ds_config["criteria"]) + 1)]
        for ds_name, ds_config in datasets_shortened.items()
    }

    # --- figure
    n_rows, n_cols = len(PROXIES), len(datasets_shortened)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 14), constrained_layout=True)

    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    def annotate_bar(ax, rect):
        h = rect.get_height()
        x = rect.get_x() + rect.get_width() / 2.0
        va = 'bottom' if h >= 0 else 'top'
        offset = 6 if h >= 0 else -8
        ax.annotate(
            f"{h:.3f}", xy=(x, h), xytext=(0, offset),
            textcoords="offset points", ha="center", va=va, fontsize=ANNOT_FS
        )

    # column titles
    for c, ds_name in enumerate(datasets_shortened):
        axes[0, c].set_title(ds_name, fontsize=TITLE_FS)

    # draw bars
    for r, (proxy_title, proxy_key) in enumerate(PROXIES):
        color = PROXY_COLOR[proxy_key]

        # common y-range per row
        row_vals = []
        for ds_name in datasets_shortened:
            row_vals.extend(vals[proxy_key][ds_name])
        row_vals = np.asarray(row_vals, dtype=float)

        for c, ds_name in enumerate(datasets_shortened):
            ax = axes[r, c]
            y = vals[proxy_key][ds_name]
            labels = ds_labels[ds_name]

            if row_vals.size == 0:
                row_min, row_max = -1.0, 1.0
            else:
                row_min, row_max = float(np.min(row_vals)), float(np.max(row_vals))

            pad = 0.05 * max(1.0, abs(row_max - row_min))  # 5% padding

            if proxy_key in ("MI", "PRIV_OFFSET", "TVD"):
                ymin, ymax = 0.0, row_max + pad
            else:
                ymin, ymax = row_min - pad, row_max + pad

            ax.set_ylim(ymin, ymax)

            # tighter spacing
            x = np.arange(len(y)) * BAR_SPACING
            bars = ax.bar(x, y, color=color, width=BAR_WIDTH)
            for rect in bars:
                annotate_bar(ax, rect)

            ax.set_xticks(x)
            if r == n_rows - 1:
                ax.set_xticklabels(labels, ha="right", fontsize=TICK_FS)
                # x label only for bottom row
                ax.set_xlabel("criterion", fontsize=ROWLAB_FS)
            else:
                ax.set_xticklabels([])

            if c == 0:
                ax.set_ylabel(proxy_title, fontsize=ROWLAB_FS, labelpad=20)

            ax.yaxis.grid(True, linestyle=":", linewidth=0.9, alpha=0.65)
            ax.tick_params(axis='y', labelsize=TICK_FS)
            # <<< format y tick labels here
            ax.yaxis.set_major_formatter(y_formatter)

    plt.savefig(outfile, dpi=600)
    plt.show()


def create_plot_2(outfile: str="plots/plot2.png"):
    # --- config ---------------------------------------------------
    MEASURES = [
        ("TVD Proxy", "TVD"),
        ("Tuple Contribution", "AUC"),
    ]

    MEASURE_COLOR = {
        "TVD": "#1f77b4",   # blue
        "AUC": "#ff7f0e",   # orange
    }

    TITLE_FS = 32  # column titles
    ROWLAB_FS = 32  # row (y-axis) labels
    TICK_FS = 28  # tick labels (bottom axis + y-ticks)
    ANNOT_FS = 22  # numbers above bars

    BAR_SPACING = 0.55
    BAR_WIDTH = 0.5

    # --- compute values ------------------------------------------
    vals = {k: {} for _, k in MEASURES}
    for ds_name, ds_config in datasets_shortened.items():
        path, criteria = ds_config["path"], ds_config["criteria"]
        tvd_scores, auc_scores = [], []
        for criterion in criteria:
            tvd_scores.append(ProxyMutualInformationTVD(datapath=path).calculate(
                [criterion], encode_and_clean=True
            ))
            auc_scores.append(TupleContribution(datapath=path).calculate(
                [criterion], encode_and_clean=True
            ))
        vals["TVD"][ds_name] = tvd_scores
        vals["AUC"][ds_name] = auc_scores

    ds_labels = {
        ds_name: [str(i) for i in range(1, len(ds_config["criteria"]) + 1)]
        for ds_name, ds_config in datasets_shortened.items()
    }

    # --- figure ---------------------------------------------------
    n_rows, n_cols = len(MEASURES), len(datasets_shortened)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), constrained_layout=True)

    def annotate_bar(ax, rect):
        h = rect.get_height()
        x = rect.get_x() + rect.get_width() / 2.0
        va = 'bottom' if h >= 0 else 'top'
        offset = 6 if h >= 0 else -8
        ax.annotate(
            f"{h:.3f}", xy=(x, h), xytext=(0, offset),
            textcoords="offset points", ha="center", va=va, fontsize=ANNOT_FS
        )

    # Column titles
    for c, ds_name in enumerate(datasets_shortened):
        axes[0, c].set_title(ds_name, fontsize=TITLE_FS)

    # Draw bars ----------------------------------------------------
    for r, (measure_title, measure_key) in enumerate(MEASURES):
        color = MEASURE_COLOR[measure_key]
        row_vals = []
        for ds_name in datasets_shortened:
            row_vals.extend(vals[measure_key][ds_name])
        row_vals = np.asarray(row_vals, dtype=float)
        row_min, row_max = float(np.min(row_vals)), float(np.max(row_vals))
        pad = 0.05 * max(1.0, abs(row_max - row_min))
        ymin, ymax = 0.0, row_max + pad

        for c, ds_name in enumerate(datasets_shortened):
            ax = axes[r, c]
            y = vals[measure_key][ds_name]
            labels = ds_labels[ds_name]

            ax.set_ylim(ymin, ymax)
            x = np.arange(len(y)) * BAR_SPACING
            bars = ax.bar(x, y, color=color, width=BAR_WIDTH)
            for rect in bars:
                annotate_bar(ax, rect)

            ax.set_xticks(x)
            if r == n_rows - 1:
                ax.set_xticklabels(labels, ha="right", fontsize=TICK_FS)
                # x label only for bottom row
                ax.set_xlabel("criterion", fontsize=ROWLAB_FS)
            else:
                ax.set_xticklabels([])

            if c == 0:
                ax.set_ylabel(measure_title, fontsize=ROWLAB_FS, labelpad=20)

            ax.yaxis.grid(True, linestyle=":", linewidth=0.9, alpha=0.65)
            ax.tick_params(axis='y', labelsize=TICK_FS)
            ax.yaxis.set_major_formatter(y_formatter)  # <<< apply formatter

    plt.savefig(outfile, dpi=600)
    plt.show()


def create_plot_3(
        epsilon: float = 1.0,
        num_tuples: int = 1000,
        repetitions: int = 10,
        outfile: str = "plots/plot3.png",
):
    """
    For each dataset: histogram with X = fairness criteria (indexed 1..k), Y = value.
    For each criterion, show two bars: RepairSat and its proxy
    Repairsat with chunks, averaged over `repetitions`.
    """

    plt.rcParams.update({
        "axes.titlesize": 34,
        "axes.labelsize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 34,
    })

    fig, axes = plt.subplots(1, 5, figsize=(30, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(data))
        repair_regular_sums = {}
        repair_with_chunks_sums = {}
        repair_regular_counts = {}
        repair_with_chunks_counts = {}

        for criterion in criteria:
            if len(criterion) == 3:
                protected, response, admissible = criterion
                crit_label = f"{protected} , {response} | {admissible}"
            else:
                protected, response = criterion[0], criterion[1]
                crit_label = f"{protected} , {response}"

            repair_regular_sums[crit_label] = 0.0
            repair_with_chunks_sums[crit_label] = 0.0
            repair_regular_counts[crit_label] = 0
            repair_with_chunks_counts[crit_label] = 0

        for _ in range(repetitions):
            sample = get_sample(df=data, n=n, fairness_criteria=criteria)
            repair_regular = ProxyRepairMaxSat(data=sample)
            repair_with_chunks = ProxyRepairMaxSat(data=sample)

            for criterion in criteria:
                if len(criterion) == 3:
                    protected, response, admissible = criterion
                    crit_label = f"{protected} , {response} | {admissible}"
                else:
                    protected, response = criterion[0], criterion[1]
                    crit_label = f"{protected} , {response}"
                with ThreadPoolExecutor() as executor:
                    try:
                        if criterion == ['Country', 'RemoteWork', 'Employment']:
                            raise TimeoutError
                        repair_regular_val = executor.submit(
                            repair_regular.calculate, fairness_criteria=[criterion], epsilon=epsilon, chunk_size=None
                        ).result(timeout=timeout_seconds)
                        repair_regular_sums[crit_label] += float(repair_regular_val)
                        repair_regular_counts[crit_label] += 1
                    except TimeoutError:
                        print("Skipping iteration due to timeout.")
                with ThreadPoolExecutor() as executor:
                    try:
                        repair_with_chunks_val = executor.submit(
                            repair_with_chunks.calculate, [criterion], epsilon=epsilon, chunk_size=100
                        ).result(timeout=timeout_seconds)
                        repair_with_chunks_sums[crit_label] += float(repair_with_chunks_val)
                        repair_with_chunks_counts[crit_label] += 1
                    except TimeoutError:
                        print("Skipping iteration due to timeout.")

        crit_labels = sorted(repair_regular_sums.keys())
        x = np.arange(len(crit_labels), dtype=float)
        width = 0.35
        repair_regular_vals = []
        repair_with_chunks_vals = []
        for cl in crit_labels:
            mi_mean = repair_regular_sums[cl] / repair_regular_counts[cl] if repair_regular_counts[cl] > 0 else np.nan
            tvd_mean = repair_with_chunks_sums[cl] / repair_with_chunks_counts[cl] if (
                    repair_with_chunks_counts[cl] > 0) else np.nan
            repair_regular_vals.append(mi_mean)
            repair_with_chunks_vals.append(tvd_mean)
        repair_regular_vals = np.array(repair_regular_vals, dtype=float)
        repair_with_chunks_vals = np.array(repair_with_chunks_vals, dtype=float)
        tau, _ = kendalltau(repair_regular_vals, repair_with_chunks_vals)
        print(f"Kendall tau for {ds_name}: {tau}")
        mi_bars = ax.bar(x - width / 2, repair_regular_vals, width, label="Regular RepairMaxSAT")
        ax.bar(x + width / 2, repair_with_chunks_vals, width, label="RepairMaxSAT with chunks")
        ax.set_xlabel("criterion")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(1, len(crit_labels) + 1)])

        for rect, val in zip(mi_bars, repair_regular_vals):
            if not np.isnan(val):
                height = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    height,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=18,
                )

        ax.set_title(ds_name)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(
        f"Comparison of regular RepairMaxSAT and RepairMaxSAT with chunks of size 100, "
        f"{round(num_tuples / 1000)}K tuples",
        y=1.03,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.show()


def create_plot_4(
        epsilon: float = 1.0,
        num_tuples: int = 1000,
        repetitions: int = 10,
        outfile: str = "plots/plot4.png",
):
    """
    For each dataset: line plot with X = fairness criteria (indexed 1..k), Y = runtime (seconds).
    Two lines per dataset:
      - Regular RepairMaxSAT (chunk_size=None)
      - RepairMaxSAT with chunks (chunk_size=100)
    Shadow bands show min..max runtime over `repetitions` for each criterion (like run_experiment_3).
    """

    plt.rcParams.update({
        "axes.titlesize": 34,
        "axes.labelsize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 34,
    })

    fig, axes = plt.subplots(1, 5, figsize=(30, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]

        cols_list = []
        for criterion in criteria:
            cols_list += criterion

        data = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(data))

        # per-criterion list of runtimes (over repetitions)
        # key = crit_label -> list[float]
        reg_times_rep = {}
        chk_times_rep = {}

        # Initialize labels
        crit_labels = []
        for criterion in criteria:
            if len(criterion) == 3:
                protected, response, admissible = criterion
                crit_label = f"{protected} , {response} | {admissible}"
            else:
                protected, response = criterion[0], criterion[1]
                crit_label = f"{protected} , {response}"

            crit_labels.append(crit_label)
            reg_times_rep[crit_label] = []
            chk_times_rep[crit_label] = []

        # Run repetitions
        for _ in range(repetitions):
            sample = get_sample(df=data, n=n, fairness_criteria=criteria)
            repair_regular = ProxyRepairMaxSat(data=sample)
            repair_with_chunks = ProxyRepairMaxSat(data=sample)

            for criterion in criteria:
                if len(criterion) == 3:
                    protected, response, admissible = criterion
                    crit_label = f"{protected} , {response} | {admissible}"
                else:
                    protected, response = criterion[0], criterion[1]
                    crit_label = f"{protected} , {response}"

                with ThreadPoolExecutor() as executor:
                    try:
                        if criterion == ["Country", "RemoteWork", "Employment"]:
                            raise TimeoutError
                        start = time.time()
                        _ = executor.submit(
                            repair_regular.calculate,
                            fairness_criteria=[criterion],
                            epsilon=epsilon,
                            chunk_size=None,
                        ).result(timeout=timeout_seconds)
                        reg_times_rep[crit_label].append(float(time.time() - start))
                    except TimeoutError:
                        print("Skipping regular RepairMaxSAT timing due to timeout.")
                        reg_times_rep[crit_label].append(np.nan)
                with ThreadPoolExecutor() as executor:
                    try:
                        start = time.time()
                        _ = executor.submit(
                            repair_with_chunks.calculate,
                            [criterion],
                            epsilon=epsilon,
                            chunk_size=100,
                        ).result(timeout=timeout_seconds)
                        chk_times_rep[crit_label].append(float(time.time() - start))
                    except TimeoutError:
                        print("Skipping chunked RepairMaxSAT timing due to timeout.")
                        chk_times_rep[crit_label].append(np.nan)

        # Sort criteria labels deterministically to match your other plots
        crit_labels_sorted = sorted(crit_labels)
        x = np.arange(1, len(crit_labels_sorted) + 1, dtype=float)

        # Aggregate mean/min/max per criterion (ignoring NaNs)
        reg_mean, reg_min, reg_max = [], [], []
        chk_mean, chk_min, chk_max = [], [], []

        for cl in crit_labels_sorted:
            r = np.asarray(reg_times_rep[cl], dtype=float)
            r = r[~np.isnan(r)]
            if r.size == 0:
                reg_mean.append(np.nan); reg_min.append(np.nan); reg_max.append(np.nan)
            else:
                reg_mean.append(float(r.mean())); reg_min.append(float(r.min())); reg_max.append(float(r.max()))

            c = np.asarray(chk_times_rep[cl], dtype=float)
            c = c[~np.isnan(c)]
            if c.size == 0:
                chk_mean.append(np.nan); chk_min.append(np.nan); chk_max.append(np.nan)
            else:
                chk_mean.append(float(c.mean())); chk_min.append(float(c.min())); chk_max.append(float(c.max()))

        reg_mean = np.asarray(reg_mean, dtype=float)
        reg_min  = np.asarray(reg_min,  dtype=float)
        reg_max  = np.asarray(reg_max,  dtype=float)

        chk_mean = np.asarray(chk_mean, dtype=float)
        chk_min  = np.asarray(chk_min,  dtype=float)
        chk_max  = np.asarray(chk_max,  dtype=float)

        # ---- plot lines + shadow bands ----
        line_reg, = ax.plot(x, reg_mean, marker="o", linewidth=2, label="Regular RepairMaxSAT")
        mask_reg = ~np.isnan(reg_mean) & ~np.isnan(reg_min) & ~np.isnan(reg_max)
        if mask_reg.any():
            ax.fill_between(
                x[mask_reg],
                reg_min[mask_reg],
                reg_max[mask_reg],
                alpha=0.2,
                color=line_reg.get_color(),
                linewidth=0,
            )

        line_chk, = ax.plot(x, chk_mean, marker="o", linewidth=2, label="RepairMaxSAT with chunks")
        mask_chk = ~np.isnan(chk_mean) & ~np.isnan(chk_min) & ~np.isnan(chk_max)
        if mask_chk.any():
            ax.fill_between(
                x[mask_chk],
                chk_min[mask_chk],
                chk_max[mask_chk],
                alpha=0.2,
                color=line_chk.get_color(),
                linewidth=0,
            )

        ax.set_xlabel("criterion")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(i)) for i in x])
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("runtime (s)")
    fig.suptitle(
        f"Runtime comparison of regular RepairMaxSAT and RepairMaxSAT with chunks of size 100, "
        f"{round(num_tuples / 1000)}K tuples",
        y=1.03,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.show()


######################################### Experiments ##########################################

measures = {
    "Proxy Mutual Information TVD": ProxyMutualInformationTVD,
    "Proxy RepairMaxSat": ProxyRepairMaxSat,
    "Tuple Contribution": TupleContribution,
}

timeout_seconds = 24 * 60 * 60

def _encode_and_clean(data_path, cols):
    """
    Read CSV, clean missing values, normalize numeric columns, and label-encode
    categorical columns in `cols`.

    - Replace ["NA", "N/A", ""] with NaN and drop rows with missing values in `cols`.
    - For numeric columns in `cols`, replace negative values with 0.
    - For data/census.csv:
        * Bin AGE into buckets of size 10 (e.g., 1–10 -> 10, 11–20 -> 20, ...).
        * Drop rows with INCTOT > 200000.
        * Discretize INCTOT into 10,000-wide buckets (0–9999 -> 0, 10000–19999 -> 10000, ...).
    - For categorical columns in `cols`, apply LabelEncoder.
    """
    df = pd.read_csv(data_path)
    df = df.replace(["NA", "N/A", ""], pd.NA).copy()

    # 1) Numeric: replace negative values with 0 (only in selected cols)
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            df.loc[df[c] < 0, c] = 0

    # 2) Special handling for IPUMS-CPS census data
    if data_path == "data/census.csv":
        # Bin AGE into buckets of 10: 1–10 -> 10, 11–20 -> 20, ...
        if "AGE" in df.columns:
            age = pd.to_numeric(df["AGE"], errors="coerce")
            # Clamp to at least 1 so that 0 or invalids go into the first bucket
            age = age.fillna(1)
            age = np.clip(age, 1, None)
            # (age-1)//10 gives 0 for 1–10, 1 for 11–20, etc.; then +1 and *10 -> 10, 20, ...
            df["AGE"] = (((age - 1) // 10) + 1) * 10

        # Remove INCTOT > 200000 and discretize into 10k buckets
        if "INCTOT" in df.columns:
            inctot = pd.to_numeric(df["INCTOT"], errors="coerce")
            df = df[inctot <= 200000].copy()
            inctot = pd.to_numeric(df["INCTOT"], errors="coerce").fillna(0)
            # Bucket size 10,000; adjust if you want different granularity
            df["INCTOT"] = (inctot // 10000) * 10000

    # 3) Categorical: label-encode only non-numeric columns in `cols`
    for c in cols:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("object")  # ensure object dtype
            df[c] = df[c].fillna("Missing")  # treat NaN as a category
            df[c] = LabelEncoder().fit_transform(df[c])

    return df


def plot_legend(outfile="plots/legend_proxies.png"):
    """Creating a standalone legend figure for Experiment 1 with the four measures arranged in a single horizontal row,
    and save it to `outfile`.
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(8, 1.6))
    handles = []
    for i, label in enumerate(measures.keys()):
        line = MplLine2D(
            [2, 3], [2, 2],
            color=colors[i % len(colors)],
            marker="o",
            linestyle="-",
            linewidth=2,
            label=label,
        )
        ax.add_line(line)
        handles.append(line)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        handles,
        measures.keys(),
        loc="center",
        ncol=len(measures.keys()),
        frameon=True,
        fontsize=10,
    )

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.show()


def run_experiment_1(
    epsilon=1.0,
    repetitions=10,
    outfile="plots/experiment1.png"
):
    """Plotting average runtimes over 'repetitions' repetitions per measure and dataset as function of
    the number of tuples."""

    num_tupless_per_dataset = {
        "Adult": [1000, 5000, 10000, 15000, 30000],
        "IPUMS-CPS": [5000, 10000, 50000, 100000, 300000, 600000, 1000000],
        "Stackoverflow": [5000, 10000, 20000, 40000, 60000],
        "Compas": [1000, 1500, 3000, 7000, 10000],
        "Healthcare": [100, 200, 400, 700, 1000],
    }

    plt.rcParams.update({
        "axes.titlesize": 34,
        "axes.labelsize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 34,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)
        num_tuples_this_dataset = num_tupless_per_dataset[ds_name]
        results = {
            measure_name: {"mean": [], "min": [], "max": []}
            for measure_name in measures.keys()
        }

        for measure_name, measure_cls in measures.items():
            flag_timeout = False
            for num_tuples in num_tuples_this_dataset:
                if flag_timeout:
                    print("Skipping iteration due to timeout.")
                    results[measure_name]["mean"].append(np.nan)
                    results[measure_name]["min"].append(np.nan)
                    results[measure_name]["max"].append(np.nan)
                    continue

                runtimes_rep = []
                for _ in range(repetitions):
                    n = min(num_tuples, len(data))
                    sample = get_sample(df=data, n=n, fairness_criteria=criteria)
                    m = measure_cls(data=sample)
                    start_time = time.time()
                    with ThreadPoolExecutor() as executor:
                        try:
                            _ = executor.submit(m.calculate, criteria, epsilon=epsilon).result(
                                timeout=timeout_seconds)
                            elapsed_time = time.time() - start_time
                            runtimes_rep.append(elapsed_time)
                        except TimeoutError:
                            print("Skipping the iteration due to timeout.")
                            runtimes_rep.append(np.nan)
                            flag_timeout = True
                            break

                vals = np.array(runtimes_rep, dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    mean_v = min_v = max_v = np.nan
                else:
                    mean_v = vals.mean()
                    min_v = vals.min()
                    max_v = vals.max()
                results[measure_name]["mean"].append(mean_v)
                results[measure_name]["min"].append(min_v)
                results[measure_name]["max"].append(max_v)

        xs = np.arange(len(num_tuples_this_dataset))
        tick_labels = []

        for num_tuples in num_tuples_this_dataset:
            if num_tuples >= 1_000_000:
                tick_labels.append(f"{num_tuples // 1_000_000}M")
            elif num_tuples >= 1_000:
                tick_labels.append(f"{num_tuples // 1_000}K")
            else:
                tick_labels.append(str(num_tuples))

        if ds_name == "IPUMS-CPS":
            if len(num_tuples_this_dataset) >= 7:
                show_idx = [0, 2, 4, 6]
            else:
                show_idx = list(range(len(num_tuples_this_dataset)))
            ax.set_xticks(np.array(show_idx))
            ax.set_xticklabels([tick_labels[i] for i in show_idx])
        else:
            ax.set_xticks(xs)
            ax.set_xticklabels(tick_labels)

        for measure_name, stats in results.items():
            means = np.array(stats["mean"])
            lows  = np.array(stats["min"])
            highs = np.array(stats["max"])
            line, = ax.plot(xs, means, marker="o", linewidth=2, label=measure_name)
            mask = ~np.isnan(means) & ~np.isnan(lows) & ~np.isnan(highs)
            if mask.any():
                ax.fill_between(
                    xs[mask],
                    lows[mask],
                    highs[mask],
                    alpha=0.2,
                    color=line.get_color(),
                    linewidth=0,
                )

        ax.set_xlabel("number of tuples")
        ax.set_yscale('log')
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("runtime (s), log scale")
    fig.suptitle("Runtime as Function of Number of Tuples", y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.show()


def run_experiment_2(
    epsilon=1.0,
    num_tuples=100000,
    repetitions=10,
    outfile="plots/experiment2.png"
):
    """Plot average runtimes over `repetitions` per measure and dataset as function of the number of criteria."""

    plt.rcParams.update({
        "axes.titlesize": 34,
        "axes.labelsize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 34,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(data))
        results = {
            measure_name: {"mean": [], "min": [], "max": []}
            for measure_name in measures.keys()
        }

        for measure_name, measure_cls in measures.items():
            flag_timeout = False

            for num_criteria in range(1, len(criteria) + 1):
                if flag_timeout:
                    print("Skipping iteration due to timeout for smaller number of criteria.")
                    results[measure_name]["mean"].append(np.nan)
                    results[measure_name]["min"].append(np.nan)
                    results[measure_name]["max"].append(np.nan)
                    continue

                runtimes_rep = []
                for _ in range(repetitions):
                    sample = get_sample(df=data, n=n, fairness_criteria=criteria)
                    m = measure_cls(data=sample)
                    start_time = time.time()
                    with ThreadPoolExecutor() as executor:
                        try:
                            _ = executor.submit(
                                m.calculate,
                                criteria[:num_criteria],
                                epsilon=epsilon
                            ).result(timeout=timeout_seconds)
                            elapsed_time = time.time() - start_time
                            runtimes_rep.append(elapsed_time)
                        except TimeoutError:
                            print("Skipping iteration due to timeout.")
                            runtimes_rep.append(np.nan)
                            flag_timeout = True
                            break

                vals = np.array(runtimes_rep, dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    mean_v = min_v = max_v = np.nan
                else:
                    mean_v = vals.mean()
                    min_v = vals.min()
                    max_v = vals.max()
                results[measure_name]["mean"].append(mean_v)
                results[measure_name]["min"].append(min_v)
                results[measure_name]["max"].append(max_v)

        xs = np.arange(1, len(criteria) + 1)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(k)) for k in xs])

        for measure_name, stats in results.items():
            means = np.array(stats["mean"])
            lows  = np.array(stats["min"])
            highs = np.array(stats["max"])
            line, = ax.plot(xs, means, marker="o", linewidth=2, label=measure_name)
            mask = ~np.isnan(means) & ~np.isnan(lows) & ~np.isnan(highs)
            if mask.any():
                ax.fill_between(
                    xs[mask],
                    lows[mask],
                    highs[mask],
                    alpha=0.2,
                    color=line.get_color(),
                    linewidth=0,
                )

        ax.set_xlabel("number of criteria")
        ax.set_yscale('log')
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("runtime (s), log scale")
    fig.suptitle(
        f"Runtime as Function of Number of Criteria, number of tuples at most {round(num_tuples / 1000)}K",
        y=1.02
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.show()


def run_experiment_3(
    epsilons=(0.1, 1, 5, 10),
    num_tuples=100000,
    repetitions=10,
    outfile="plots/experiment3.png"
):
    """Relative L1 error as function of epsilon."""

    def _rel_error(x, y, tiny = 1e-100):
        denom = max(abs(y), tiny)  # ensure we do not divide by 0
        return abs(x - y) / denom

    plt.rcParams.update({
        "axes.titlesize": 34,
        "axes.labelsize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 34,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(data))
        results = {
            measure_name: {"mean": [], "min": [], "max": []}
            for measure_name in measures.keys()
        }

        for measure_name, measure_cls in measures.items():
            flag_timeout = False
            if flag_timeout:
                for _ in epsilons:
                    results[measure_name]["mean"].append(np.nan)
                    results[measure_name]["min"].append(np.nan)
                    results[measure_name]["max"].append(np.nan)
                continue

            errs_per_eps = [[] for _ in epsilons]
            for _ in range(repetitions):
                sample = get_sample(df=data, n=n, fairness_criteria=criteria)
                m = measure_cls(data=sample)
                with ThreadPoolExecutor() as executor:
                    try:
                        non_private_result = executor.submit(
                            m.calculate, criteria, epsilon=None
                        ).result(timeout=timeout_seconds)
                    except TimeoutError:
                        print("Skipping iteration due to timeout.")
                        flag_timeout = True
                        continue

                for j, eps in enumerate(epsilons):
                    if flag_timeout:
                        errs_per_eps[j].append(np.nan)
                        continue
                    with ThreadPoolExecutor() as executor:
                        try:
                            private_result = executor.submit(
                                m.calculate, criteria, epsilon=eps
                            ).result(timeout=timeout_seconds)
                            err = _rel_error(private_result, non_private_result)
                            errs_per_eps[j].append(err)
                        except TimeoutError:
                            print("Skipping iteration due to timeout.")
                            flag_timeout = True
                            continue

            for j in range(len(epsilons)):
                vals = np.array(errs_per_eps[j], dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    mean_v = min_v = max_v = np.nan
                else:
                    mean_v = vals.mean()
                    min_v = vals.min()
                    max_v = vals.max()
                results[measure_name]["mean"].append(mean_v)
                results[measure_name]["min"].append(min_v)
                results[measure_name]["max"].append(max_v)

        x = np.array(epsilons, dtype=float)
        for measure_name, stats in results.items():
            means = np.array(stats["mean"])
            lows  = np.array(stats["min"])
            highs = np.array(stats["max"])
            line, = ax.plot(x, means, marker="o", linewidth=2, label=measure_name)
            mask = ~np.isnan(means) & ~np.isnan(lows) & ~np.isnan(highs)
            if mask.any():
                ax.fill_between(
                    x[mask],
                    lows[mask],
                    highs[mask],
                    alpha=0.2,
                    color=line.get_color(),
                    linewidth=0,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([str(eps) for eps in epsilons])
        ax.set_xlabel("privacy budget ε")
        ax.set_yscale('log')
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("relative L1 error, log scale")
    fig.suptitle(
        f"Relative L1 Error as Function of Privacy Budget, number of tuples at most {round(num_tuples / 1000)}K",
        y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.show()


def run_experiment_4(
        epsilon: float = 1.0,
        num_tuples: int = 100000,
        repetitions: int = 10,
        outfile: str = "plots/experiment4.png",
):
    """
    For each dataset: histogram with X = fairness criteria (indexed 1..k), Y = value.
    For each criterion, show two bars: MutualInformation and its proxy
    ProxyMutualInformationTVD, averaged over `repetitions`.
    """

    plt.rcParams.update({
        "axes.titlesize": 34,
        "axes.labelsize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 34,
    })

    fig, axes = plt.subplots(1, 5, figsize=(30, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(data))
        mi_sums = {}
        tvd_sums = {}
        mi_counts = {}
        tvd_counts = {}

        for criterion in criteria:
            if len(criterion) == 3:
                protected, response, admissible = criterion
                crit_label = f"{protected} , {response} | {admissible}"
            else:
                protected, response = criterion[0], criterion[1]
                crit_label = f"{protected} , {response}"

            mi_sums[crit_label] = 0.0
            tvd_sums[crit_label] = 0.0
            mi_counts[crit_label] = 0
            tvd_counts[crit_label] = 0

        for _ in range(repetitions):
            sample = get_sample(df=data, n=n, fairness_criteria=criteria)
            mi_measure = MutualInformation(data=sample)
            tvd_measure = ProxyMutualInformationTVD(data=sample)

            for criterion in criteria:
                if len(criterion) == 3:
                    protected, response, admissible = criterion
                    crit_label = f"{protected} , {response} | {admissible}"
                else:
                    protected, response = criterion[0], criterion[1]
                    crit_label = f"{protected} , {response}"
                with ThreadPoolExecutor() as executor:
                    try:
                        mi_val = executor.submit(
                            mi_measure.calculate, [criterion], epsilon=epsilon
                        ).result(timeout=timeout_seconds)
                        mi_sums[crit_label] += float(mi_val)
                        mi_counts[crit_label] += 1
                    except TimeoutError:
                        print("Skipping iteration due to timeout.")
                with ThreadPoolExecutor() as executor:
                    try:
                        tvd_val = executor.submit(
                            tvd_measure.calculate, [criterion], epsilon=epsilon
                        ).result(timeout=timeout_seconds)
                        tvd_sums[crit_label] += float(tvd_val)
                        tvd_counts[crit_label] += 1
                    except TimeoutError:
                        print("Skipping iteration due to timeout.")

        crit_labels = sorted(mi_sums.keys())
        x = np.arange(len(crit_labels), dtype=float)
        width = 0.35
        mi_vals = []
        tvd_vals = []
        for cl in crit_labels:
            mi_mean = mi_sums[cl] / mi_counts[cl] if mi_counts[cl] > 0 else np.nan
            tvd_mean = tvd_sums[cl] / tvd_counts[cl] if tvd_counts[cl] > 0 else np.nan
            mi_vals.append(mi_mean)
            tvd_vals.append(tvd_mean)
        tau, _ = kendalltau(mi_vals, tvd_vals)
        print(f"Kendall tau for {ds_name}: {tau}")
        mi_vals = np.array(mi_vals, dtype=float)
        tvd_vals = np.array(tvd_vals, dtype=float)
        mi_bars = ax.bar(x - width / 2, mi_vals, width, label="MutualInformation")
        ax.bar(x + width / 2, tvd_vals, width, label="ProxyMutualInformationTVD")
        ax.set_xlabel("criterion")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(1, len(crit_labels) + 1)])

        for rect, val in zip(mi_bars, mi_vals):
            if not np.isnan(val):
                height = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    height,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=18,
                )

        ax.set_title(ds_name)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(
        f"Comparison of MutualInformation and ProxyMutualInformationTVD, at most "
        f"{round(num_tuples / 1000)}K tuples",
        y=1.03,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.show()


def run_experiment_5(
    num_tuples=100000,
    repetitions=10,
    epsilon=None,
    outfile="plots/experiment5.png",
):
    """TupleContribution value as function of k, sampling separately for each repetition."""

    ks_per_dataset = {
        "Adult": [10, 50, 100, 250, 500, 1000, 5000, 10000, 15000, 30000],
        "IPUMS-CPS": [10, 50, 100, 250, 500, 1000, 5000, 10000, 50000, 100000, 300000, 600000, 1000000],
        "Stackoverflow": [10, 50, 100, 250, 500, 1000, 5000, 10000, 20000, 40000, 60000],
        "Compas": [10, 50, 100, 250, 500, 1000, 1500, 3000, 7000, 10000],
        "Healthcare": [10, 50, 100, 200, 400, 700, 1000],
    }

    plt.rcParams.update({
        "axes.titlesize": 34,
        "axes.labelsize": 29,
        "xtick.labelsize": 16,
        "ytick.labelsize": 20,
        "figure.titlesize": 34,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        ks_this_dataset = ks_per_dataset[ds_name]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(data))
        stats = {"mean": []}

        flag_timeout = False
        for k in ks_this_dataset:
            if flag_timeout:
                print("Skipping the iteration due to timeout.")
                stats["mean"].append(np.nan)
                continue

            values_rep = []
            for _ in range(repetitions):
                sample = get_sample(df=data, n=n, fairness_criteria=criteria)
                m = TupleContribution(data=sample)
                with ThreadPoolExecutor() as executor:
                    try:
                        val = executor.submit(
                            m.calculate,
                            criteria,
                            k=k,
                            epsilon=epsilon,
                        ).result(timeout=timeout_seconds)
                        values_rep.append(float(val))
                    except TimeoutError:
                        print("Skipping the iteration due to timeout.")
                        values_rep.append(np.nan)
                        flag_timeout = True
                        break

            vals = np.array(values_rep, dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                mean_v = np.nan
            else:
                mean_v = vals.mean()
            stats["mean"].append(mean_v)

        xs = np.arange(len(ks_this_dataset))
        means = np.array(stats["mean"], dtype=float)
        means = np.clip(means, 1e-3, None)
        line, = ax.plot(xs, means, marker="o", linewidth=2,
                        label="TupleContribution value")
        tick_labels = []
        for k in ks_this_dataset:
            if k >= 1_000_000:
                tick_labels.append(f"{k / 1_000_000:g}M")
            elif k >= 1_000:
                tick_labels.append(f"{k / 1_000:g}K")
            else:
                tick_labels.append(str(k))
        ax.set_xticks(xs)
        if ds_name != "Healthcare":
            show_idx = list(range(0, len(ks_this_dataset), 2))
            if (len(ks_this_dataset) - 1) not in show_idx:
                show_idx.append(len(ks_this_dataset) - 1)
            ax.set_xticks(np.array(show_idx))
            ax.set_xticklabels([tick_labels[i] for i in show_idx])
        else:
            ax.set_xticklabels(tick_labels)
        ax.set_xlabel("k (top-k tuples)")
        ax.set_yscale('log')
        if ds_name == "Healthcare":
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("TupleContribution, log scale")
    fig.suptitle(
        f"TupleContribution value as function of k, at most {round(num_tuples / 1000)}K tuples",
        y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.show()


def run_experiment_6(
    num_tuples=100000,
    repetitions=50,
    epsilon=1.0,
    outfile="plots/experiment6.png",
):
    """Relative L1 error of TupleContribution as function of k."""

    ks_per_dataset = {
        "Adult": [10, 50, 100, 250, 500, 1000, 5000, 10000, 15000, 30000],
        "IPUMS-CPS": [10, 50, 100, 250, 500, 1000, 5000, 10000, 50000, 100000, 300000, 600000, 1000000],
        "Stackoverflow": [10, 50, 100, 250, 500, 1000, 5000, 10000, 20000, 40000, 60000],
        "Compas": [10, 50, 100, 250, 500, 1000, 1500, 3000, 7000, 10000],
        "Healthcare": [10, 50, 100, 200, 400, 700, 1000],
    }

    def _rel_error(x, y, tiny=1e-100):
        denom = max(abs(y), tiny)  # ensure we do not divide by 0
        return abs(x - y) / denom

    plt.rcParams.update({
        "axes.titlesize": 34,
        "axes.labelsize": 30,
        "xtick.labelsize": 16,
        "ytick.labelsize": 17,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        ks = ks_per_dataset[ds_name]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(data))
        sample = get_sample(df=data, n=n, fairness_criteria=criteria)
        m = TupleContribution(data=sample)
        stats = {"mean": [], "min": [], "max": []}

        for k in ks:
            errs = []

            for _ in range(repetitions):
                with ThreadPoolExecutor() as executor:
                    try:
                        non_private_result = executor.submit(
                            m.calculate,
                            fairness_criteria=criteria,
                            k=k,
                            epsilon=None,
                            encode_and_clean=False
                        ).result(timeout=timeout_seconds)
                    except TimeoutError:
                        print("Skipping iteration due to timeout.")
                        errs.append(np.nan)
                        break
                with ThreadPoolExecutor() as executor:
                    try:
                        private_result = executor.submit(
                            m.calculate,
                            fairness_criteria=criteria,
                            k=k,
                            epsilon=epsilon,
                            encode_and_clean=False
                        ).result(timeout=timeout_seconds)
                        errs.append(_rel_error(private_result, non_private_result))
                    except TimeoutError:
                        print("Skipping iteration due to timeout.")
                        errs.append(np.nan)
                        break

            vals = np.array(errs, dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                mean_v = min_v = max_v = np.nan
            else:
                mean_v = vals.mean()
                min_v = vals.min()
                max_v = vals.max()
            stats["mean"].append(mean_v)
            stats["min"].append(min_v)
            stats["max"].append(max_v)

        xs = np.arange(len(ks))
        means = np.array(stats["mean"])
        lows  = np.array(stats["min"])
        highs = np.array(stats["max"])
        line, = ax.plot(xs, means, marker="o", linewidth=2,
                        label="TupleContribution L1 error")
        mask = ~np.isnan(means) & ~np.isnan(lows) & ~np.isnan(highs)
        if mask.any():
            ax.fill_between(
                xs[mask],
                lows[mask],
                highs[mask],
                alpha=0.2,
                color=line.get_color(),
                linewidth=0,
            )
        tick_labels = []
        for k in ks:
            if k >= 1_000_000:
                tick_labels.append(f"{k / 1_000_000:g}M")
            elif k >= 1_000:
                tick_labels.append(f"{k / 1_000:g}K")
            else:
                tick_labels.append(str(k))
        ax.set_xticks(xs)
        if ds_name != "Healthcare":
            show_idx = list(range(0, len(ks), 2))
            if (len(ks) - 1) not in show_idx:
                show_idx.append(len(ks) - 1)
            ax.set_xticks(np.array(show_idx))
            ax.set_xticklabels([tick_labels[i] for i in show_idx])
        else:
            ax.set_xticklabels(tick_labels)
        ax.set_xlabel("k (top-k tuples)")
        ax.set_yscale('log')
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("relative L1 error, log scale")
    fig.suptitle(
        f"Relative L1 Error of TupleContribution as Function of k, at most {round(num_tuples / 1000)}K tuples, "
        f"ε = {epsilon}",
        y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.show()


def run_experiment_7(
    epsilon: Optional[float]=1.0,
    n_per_sex: int=50000,
    step: float=0.1,                    # switch 10% each iteration
    repetitions: int=10,
    outfile: str = "plots/experiment7.png",
):
    """
    Plot measure values vs growing unfairness on a synthetic dataset with two columns:
      sex in {0,1} and income>50K in {0,1}.
    Start fair: for each sex, 50% income=0 and 50% income=1.
    Then, at each step t = 0, step, 2*step, ..., 1.0:
      - flip a fraction t of (male, income=0) to income=1
      - flip a fraction t of (female, income=1) to income=0
    So at t=1.0: all males income=1, all females income=0.
    """

    # ---- helpers -------------------------------------------------
    def _make_dataset(t: float) -> pd.DataFrame:
        """
        Create synthetic dataset for a given unfairness level t in [0,1].
        sex: 1=male, 0=female (arbitrary, but consistent)
        income: 1=income>50K, 0=otherwise
        """
        n = n_per_sex

        # Base fair allocation:
        # males: n/2 income=0, n/2 income=1
        # females: n/2 income=0, n/2 income=1
        m0 = n // 2
        m1 = n - m0
        f0 = n // 2
        f1 = n - f0

        # How many to flip at level t
        # flip t fraction of male income=0 -> 1, and t fraction of female income=1 -> 0
        flip_m = int(round(t * m0))
        flip_f = int(round(t * f1))

        m0_new = m0 - flip_m
        m1_new = m1 + flip_m
        f1_new = f1 - flip_f
        f0_new = f0 + flip_f

        sex = np.concatenate([
            np.ones(m0_new + m1_new, dtype=int),   # males = 1
            np.zeros(f0_new + f1_new, dtype=int),  # females = 0
        ])
        income = np.concatenate([
            np.concatenate([np.zeros(m0_new, dtype=int), np.ones(m1_new, dtype=int)]),
            np.concatenate([np.zeros(f0_new, dtype=int), np.ones(f1_new, dtype=int)]),
        ])

        df = pd.DataFrame({"sex": sex, "income>50K": income})
        df = df.sample(frac=1.0, replace=False).reset_index(drop=True)
        return df

    def _dp_gap(df: pd.DataFrame) -> float:
        rates = df.groupby("sex")["income>50K"].mean()
        return float(abs(rates.max() - rates.min())) if len(rates) else 0.0

    # ---- build unfairness grid ----------------------------------
    ts = [round(i * step, 10) for i in range(int(1 / step) + 1)]
    criterion = ["sex", "income>50K"]

    # Store per-t aggregates: mean/min/max for each measure and for dp-gap
    results = {
        name: {"mean": [], "min": [], "max": []}
        for name in measures.keys()
    }
    dp_stats = {"mean": [], "min": [], "max": []}

    # ---- run -----------------------------------------------------
    for t in ts:
        vals_rep = {name: [] for name in measures.keys()}
        dp_rep = []

        for _ in range(repetitions):
            df = _make_dataset(t)

            for measure_name, measure_cls in measures.items():
                m = measure_cls(data=df[criterion].copy())
                with ThreadPoolExecutor() as executor:
                    try:
                        v = executor.submit(
                            m.calculate,
                            [criterion],
                            epsilon=epsilon,
                        ).result(timeout=timeout_seconds)
                        vals_rep[measure_name].append(float(v))
                    except TimeoutError:
                        vals_rep[measure_name].append(np.nan)

            dp_rep.append(_dp_gap(df))

        # Aggregate per measure: mean/min/max (ignoring NaNs)
        for measure_name in measures.keys():
            arr = np.asarray(vals_rep[measure_name], dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                results[measure_name]["mean"].append(np.nan)
                results[measure_name]["min"].append(np.nan)
                results[measure_name]["max"].append(np.nan)
            else:
                results[measure_name]["mean"].append(float(arr.mean()))
                results[measure_name]["min"].append(float(arr.min()))
                results[measure_name]["max"].append(float(arr.max()))

        dp_arr = np.asarray(dp_rep, dtype=float)
        dp_arr = dp_arr[~np.isnan(dp_arr)]
        if dp_arr.size == 0:
            dp_stats["mean"].append(np.nan)
            dp_stats["min"].append(np.nan)
            dp_stats["max"].append(np.nan)
        else:
            dp_stats["mean"].append(float(dp_arr.mean()))
            dp_stats["min"].append(float(dp_arr.min()))
            dp_stats["max"].append(float(dp_arr.max()))

        print(
            f"t={t:.1f}  DP-gap≈{dp_stats['mean'][-1]:.3f}  " +
            "  ".join([f"{mn}≈{results[mn]['mean'][-1]:.4f}" for mn in measures.keys()])
        )

    # ---- plot ----------------------------------------------------
    plt.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    fig, ax = plt.subplots(figsize=(8, 4))

    # Use DP-gap mean on x-axis (as you want)
    x = np.asarray(dp_stats["mean"], dtype=float)

    for measure_name in measures.keys():
        y_mean = np.asarray(results[measure_name]["mean"], dtype=float)
        y_min  = np.asarray(results[measure_name]["min"], dtype=float)
        y_max  = np.asarray(results[measure_name]["max"], dtype=float)

        line, = ax.plot(x, y_mean, marker="o", linewidth=2, label=measure_name)

        # Shadow band: min..max (skip NaNs safely)
        mask = ~np.isnan(x) & ~np.isnan(y_min) & ~np.isnan(y_max)
        if mask.any():
            ax.fill_between(
                x[mask],
                y_min[mask],
                y_max[mask],
                alpha=0.2,
                color=line.get_color(),
                linewidth=0,
            )

    ax.set_xlabel("Demographic Parity gap")
    ax.set_yscale('symlog')
    ax.set_ylabel("measure value, symlog")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(y_formatter)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.show()

def run_experiment_8(
    dataset_name: str = "Adult",
    response_col: str = "income>50K",
    protected_col: str = "sex",
    num_tuples: int = 100000,
    repetitions: int = 10,
    complexity_grid=(1, 2, 3, 4, 5),          # layers for NN, trees for RF
    nn_epochs: int = 20,                      # fixed training epochs for NN (complexity is layers)
    epsilon: Optional[float] = 10.0,
    outfile: str = "plots/experiment8.png",
):
    """
    Creates THREE plots (with min..max shadow bands over repetitions):

      1) Accuracy vs complexity
      2) Training time (seconds) vs complexity
      3) Demographic Parity gap vs complexity

    X-axis: complexity (NN = number of hidden layers, RF = number of trees)

    Both models are trained privately if epsilon is not None:
      - RF: diffprivlib RandomForestClassifier(epsilon=epsilon)
      - NN: DP-SGD via Opacus PrivacyEngine(target_epsilon=epsilon)

    Saves 3 files:
      - <outfile base>_acc.png
      - <outfile base>_time.png
      - <outfile base>_dp.png
    """

    # ---- locate dataset ----
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset_name='{dataset_name}'. Choose from {list(datasets.keys())}")
    path = datasets[dataset_name]["path"]

    raw_df = pd.read_csv(path)
    data = _encode_and_clean(path, raw_df.columns)

    # ---- validate columns ----
    required = [response_col, protected_col]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Available columns: {list(data.columns)}")

    n = min(num_tuples, len(data))
    if n < 50:
        raise ValueError(f"Too few rows after cleaning (n={n}). Increase num_tuples.")

    # ---- helpers ----
    def _dp_gap(y_pred_bin: np.ndarray, s: np.ndarray) -> float:
        """Demographic parity gap: max_g P(ŷ=1|S=g) - min_g P(ŷ=1|S=g)."""
        df_eval = pd.DataFrame({"yhat": y_pred_bin.astype(float), "s": s})
        rates = df_eval.groupby("s")["yhat"].mean()
        return float(rates.max() - rates.min()) if len(rates) else 0.0

    def _init_stats():
        return {"mean": [], "min": [], "max": []}

    def _push_stats(stats, arr):
        arr = np.asarray(arr, dtype=float)
        stats["mean"].append(float(np.mean(arr)))
        stats["min"].append(float(np.min(arr)))
        stats["max"].append(float(np.max(arr)))

    # ---- stats per model per metric ----
    rf_acc_stats, rf_time_stats, rf_dp_stats = _init_stats(), _init_stats(), _init_stats()
    nn_acc_stats, nn_time_stats, nn_dp_stats = _init_stats(), _init_stats(), _init_stats()

    for k in complexity_grid:
        rf_acc_rep, rf_time_rep, rf_dp_rep = [], [], []
        nn_acc_rep, nn_time_rep, nn_dp_rep = [], [], []

        for _ in range(repetitions):
            sample = get_sample(
                df=data,
                n=n,
                fairness_criteria=datasets[dataset_name]["criteria"],
            ).copy()

            # features / labels / protected
            feature_cols = [c for c in sample.columns if c != response_col]
            if not feature_cols:
                raise ValueError("No feature columns available (all columns equal response_col).")

            X = sample[feature_cols].to_numpy(dtype=float)
            y_raw = sample[response_col].to_numpy()
            s = sample[protected_col].to_numpy()

            le = LabelEncoder()
            y = le.fit_transform(y_raw).astype(np.int64)
            if len(np.unique(y)) < 2:
                # if degenerate label in this sample, skip this repetition
                continue

            X = MinMaxScaler().fit_transform(X)

            strat = y if len(np.unique(y)) > 1 else None
            X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                X, y, s, test_size=0.2, stratify=strat
            )

            # =======================
            # ===== Random Forest ===
            # =======================
            start = time.perf_counter()
            rf = RandomForestClassifier(
                n_estimators=int(k),
                max_depth=15,
                shuffle=True,
                classes=[0, 1],
                epsilon=epsilon,
            )
            rf.fit(X_train, y_train)
            rf_time = time.perf_counter() - start

            rf_pred = rf.predict(X_test)
            rf_acc = accuracy_score(y_test, rf_pred)
            rf_dp = _dp_gap((rf_pred == 1).astype(int), s_test)

            rf_time_rep.append(rf_time)
            rf_acc_rep.append(rf_acc)
            rf_dp_rep.append(rf_dp)

            # =======================
            # ===== Neural Network ==
            # =======================
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.long)
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

            train_ds = TensorDataset(X_train_t, y_train_t)
            batch_size = min(256, len(train_ds))
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            num_classes = int(len(np.unique(y)))

            class LayerNN(nn.Module):
                def __init__(self, d_in: int, n_layers: int, d_hidden: int = 64, n_out: int = 2):
                    super().__init__()
                    layers = []
                    # first hidden
                    layers.append(nn.Linear(d_in, d_hidden))
                    layers.append(nn.ReLU())
                    # additional hidden layers
                    for _ in range(max(0, n_layers - 1)):
                        layers.append(nn.Linear(d_hidden, d_hidden))
                        layers.append(nn.ReLU())
                    # output
                    layers.append(nn.Linear(d_hidden, n_out))
                    self.net = nn.Sequential(*layers)

                def forward(self, x):
                    return self.net(x)

            model = LayerNN(d_in=X_train.shape[1], n_layers=int(k), d_hidden=64, n_out=num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

            if epsilon is not None:
                privacy_engine = PrivacyEngine()
                model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    target_epsilon=float(epsilon),
                    target_delta=1e-5,
                    epochs=int(nn_epochs),
                    max_grad_norm=1.0,
                )

            start = time.perf_counter()
            model.train()
            for _ep in range(int(nn_epochs)):
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
            nn_time = time.perf_counter() - start

            model.eval()
            with torch.no_grad():
                nn_logits = model(X_test_t)
                nn_pred = torch.argmax(nn_logits, dim=1).cpu().numpy()

            nn_acc = accuracy_score(y_test, nn_pred)
            nn_dp = _dp_gap((nn_pred == 1).astype(int), s_test)

            nn_time_rep.append(nn_time)
            nn_acc_rep.append(nn_acc)
            nn_dp_rep.append(nn_dp)

        # aggregate
        if len(rf_acc_rep) == 0 or len(nn_acc_rep) == 0:
            print(f"k={k}: skipped (degenerate labels or empty reps).")
            _push_stats(rf_acc_stats, [np.nan]); _push_stats(rf_time_stats, [np.nan]); _push_stats(rf_dp_stats, [np.nan])
            _push_stats(nn_acc_stats, [np.nan]); _push_stats(nn_time_stats, [np.nan]); _push_stats(nn_dp_stats, [np.nan])
            continue

        _push_stats(rf_acc_stats, rf_acc_rep)
        _push_stats(rf_time_stats, rf_time_rep)
        _push_stats(rf_dp_stats, rf_dp_rep)

        _push_stats(nn_acc_stats, nn_acc_rep)
        _push_stats(nn_time_stats, nn_time_rep)
        _push_stats(nn_dp_stats, nn_dp_rep)

        print(
            f"k={k} | "
            f"RF acc={np.mean(rf_acc_rep):.4f}, time={np.mean(rf_time_rep):.2f}s, dp={np.mean(rf_dp_rep):.3f} | "
            f"NN acc={np.mean(nn_acc_rep):.4f}, time={np.mean(nn_time_rep):.2f}s, dp={np.mean(nn_dp_rep):.3f}"
        )

    # ---- plotting ----
    plt.rcParams.update({
        "axes.titlesize": 20,
        "axes.labelsize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    x = np.asarray(complexity_grid, dtype=float)

    base, ext = os.path.splitext(outfile)
    out_acc  = f"{base}_acc.png"
    out_time = f"{base}_time.png"
    out_dp   = f"{base}_dp.png"

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)

    def _plot_one(y_rf, y_rf_min, y_rf_max, y_nn, y_nn_min, y_nn_max, ylabel, title, outpath):
        fig, ax = plt.subplots(figsize=(8, 5))

        # RF (purple)
        ax.plot(x, y_rf, marker="o", linewidth=2, color="purple", label="Random Forest")
        ax.fill_between(x, y_rf_min, y_rf_max, alpha=0.2, color="purple")

        # NN (grey)
        ax.plot(x, y_nn, marker="o", linewidth=2, color="grey", label="Neural Network")
        ax.fill_between(x, y_nn_min, y_nn_max, alpha=0.2, color="grey")

        ax.set_xlabel("complexity (trees for RF / layers for NN)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=16)

        plt.tight_layout()
        plt.savefig(outpath, dpi=600, bbox_inches="tight")
        plt.show()

    # 1) Accuracy
    _plot_one(
        np.asarray(rf_acc_stats["mean"]),
        np.asarray(rf_acc_stats["min"]),
        np.asarray(rf_acc_stats["max"]),
        np.asarray(nn_acc_stats["mean"]),
        np.asarray(nn_acc_stats["min"]),
        np.asarray(nn_acc_stats["max"]),
        ylabel="test accuracy",
        title=f"Accuracy vs Complexity ({dataset_name}, ε={epsilon})",
        outpath=out_acc,
    )

    # 2) Training time
    _plot_one(
        np.asarray(rf_time_stats["mean"]),
        np.asarray(rf_time_stats["min"]),
        np.asarray(rf_time_stats["max"]),
        np.asarray(nn_time_stats["mean"]),
        np.asarray(nn_time_stats["min"]),
        np.asarray(nn_time_stats["max"]),
        ylabel="training time (s)",
        title=f"Training Time vs Complexity ({dataset_name}, ε={epsilon})",
        outpath=out_time,
    )

    # 3) Demographic parity gap
    _plot_one(
        np.asarray(rf_dp_stats["mean"]),
        np.asarray(rf_dp_stats["min"]),
        np.asarray(rf_dp_stats["max"]),
        np.asarray(nn_dp_stats["mean"]),
        np.asarray(nn_dp_stats["min"]),
        np.asarray(nn_dp_stats["max"]),
        ylabel="Demographic Parity gap",
        title=f"DP Gap vs Complexity ({dataset_name}, ε={epsilon})",
        outpath=out_dp,
    )


def run_experiment_9(
    dataset_name: str = "IPUMS-CPS",
    num_tuples: int = 100000,
    repetitions: int = 10,                  # default 10
    epsilon: Optional[float] = 10.0,         # default 10 (set None for NO NOISE)
    outfile_csv_queries: str = "plots/experiment9_census_queries.csv",
    outfile_csv_measures: str = "plots/experiment9_census_measures.csv",
):
    """
    Experiment 9 (Census) — averaged analysis queries + averaged measure scores.

    Part A (analysis queries), averaged across repetitions:
        1) avg INCTOT (overall)
        2) avg INCTOT | EDUC
        3) avg INCTOT | SEX
        4) avg INCTOT | RACE
        5) avg INCTOT | (EDUC, SEX)
        6) avg INCTOT | (EDUC, RACE)

    Output: outfile_csv_queries (long format, averaged):
        query, group_a, group_b, avg_inctot

        Conventions:
          - overall row uses group_a="ALL", group_b="ALL"
          - single conditioning uses group_b=""
          - (EDUC, SEX): group_a=EDUC, group_b=SEX
          - (EDUC, RACE): group_a=EDUC, group_b=RACE

    Part B (measure scores), averaged across repetitions:
        MI / TVD / Repair / TupleContribution for:
            - [SEX, INCTOT]
            - [RACE, INCTOT]
            - [SEX, INCTOT, EDUC]
            - [RACE, INCTOT, EDUC]

    Output: outfile_csv_measures:
        criterion, MutualInformation, ProxyMutualInformationTVD, ProxyRepairMaxSat, TupleContribution
    """
    import os
    import numpy as np
    import pandas as pd

    # ---------------- helpers ----------------
    def _safe_mean(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else np.nan

    def _write_csv_strict(df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            df.to_csv(path, index=False)
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied writing '{path}'. "
                f"Close the file in Excel/other programs, or choose a different output path."
            ) from e

    def _densify_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Make every column dense-coded 0..k-1 to prevent np.bincount explosions
        inside TupleContribution / similar measures.
        """
        out = df.copy()
        for c in out.columns:
            codes, _ = pd.factorize(out[c], sort=False)
            out[c] = codes.astype(np.int64)
        return out

    # ---------------- load data ----------------
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset_name='{dataset_name}'. Choose from {list(datasets.keys())}")

    path = datasets[dataset_name]["path"]
    if os.path.basename(path) != "census.csv":
        raise ValueError(
            f"Expected census.csv for this experiment, but got path='{path}'. "
            f"Use dataset_name='IPUMS-CPS' (or update datasets[...] mapping)."
        )

    raw_df = pd.read_csv(path)
    data = _encode_and_clean(path, list(raw_df.columns))

    required = ["INCTOT", "SEX", "RACE", "EDUC"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Available columns: {list(data.columns)}")

    n = min(num_tuples, len(data))
    if n < 50:
        raise ValueError(f"Too few rows after cleaning (n={n}). Increase num_tuples.")

    # =========================
    # Part A: analysis queries (collect per-rep, then average)
    # Output columns: query, EDUC, SEX, RACE, avg_inctot
    # =========================
    rows_q = []

    for rep in range(repetitions):
        sample = get_sample(
            df=data,
            n=n,
            fairness_criteria=datasets[dataset_name]["criteria"],
        ).copy()

        sample["INCTOT"] = sample["INCTOT"].astype(float)

        def add_row(query: str, educ, sex, race, avg_val: float):
            rows_q.append({
                "query": query,
                "EDUC": educ,
                "SEX": sex,
                "RACE": race,
                "avg_inctot": float(avg_val),
            })

        # 1) overall
        avg_all = _safe_mean(sample["INCTOT"].to_numpy())
        add_row("avg INCTOT (overall)", "ALL", "ALL", "ALL", avg_all)

        # 2) avg INCTOT | EDUC
        for educ_val, grp in sample.groupby("EDUC"):
            add_row("avg INCTOT | EDUC", educ_val, "", "", _safe_mean(grp["INCTOT"].to_numpy()))

        # 3) avg INCTOT | SEX
        for sex_val, grp in sample.groupby("SEX"):
            add_row("avg INCTOT | SEX", "", sex_val, "", _safe_mean(grp["INCTOT"].to_numpy()))

        # 4) avg INCTOT | RACE
        for race_val, grp in sample.groupby("RACE"):
            add_row("avg INCTOT | RACE", "", "", race_val, _safe_mean(grp["INCTOT"].to_numpy()))

        # 5) avg INCTOT | (EDUC, SEX)
        for (educ_val, sex_val), grp in sample.groupby(["EDUC", "SEX"]):
            add_row("avg INCTOT | (EDUC, SEX)", educ_val, sex_val, "", _safe_mean(grp["INCTOT"].to_numpy()))

        # 6) avg INCTOT | (EDUC, RACE)
        for (educ_val, race_val), grp in sample.groupby(["EDUC", "RACE"]):
            add_row("avg INCTOT | (EDUC, RACE)", educ_val, "", race_val, _safe_mean(grp["INCTOT"].to_numpy()))

        print(f"rep={rep + 1}/{repetitions} | avg_all(INCTOT)={avg_all:.2f}")

    df_queries_raw = pd.DataFrame(rows_q)

    # Average across repetitions for identical groups (same query + EDUC/SEX/RACE)
    df_queries = (
        df_queries_raw
        .groupby(["query", "EDUC", "SEX", "RACE"], as_index=False)
        .agg(avg_inctot=("avg_inctot", "mean"))
        .sort_values(["query", "EDUC", "SEX", "RACE"], kind="stable")
        .reset_index(drop=True)
    )

    # Print (shows the group columns explicitly)
    print("\n=== Experiment 9: averaged query results (first 40 rows) ===")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 20, "display.width", 200):
        print(df_queries.head(40))

    _write_csv_strict(df_queries, outfile_csv_queries)
    print(f"Saved: {outfile_csv_queries}")

    # =========================
    # Part B: measure scores (averaged; table only)
    # =========================
    criteria = [
        ["SEX", "INCTOT"],
        ["RACE", "INCTOT"],
        ["SEX", "INCTOT", "EDUC"],
        ["RACE", "INCTOT", "EDUC"],
    ]

    rows_m = []
    for crit in criteria:
        crit_label = f"{crit[0]}, {crit[1]}" if len(crit) == 2 else f"{crit[0]}, {crit[1]} | {crit[2]}"

        mi_vals, tvd_vals, repair_vals, tc_vals = [], [], [], []
        for _ in range(repetitions):
            sample = get_sample(
                df=data,
                n=n,
                fairness_criteria=datasets[dataset_name]["criteria"],
            ).copy()

            # critical: densify before measures to avoid huge bincount allocations
            sub = _densify_columns(sample[crit].copy())

            mi_vals.append(float(MutualInformation(data=sub).calculate([crit], epsilon=epsilon)))
            tvd_vals.append(float(ProxyMutualInformationTVD(data=sub).calculate([crit], epsilon=epsilon)))
            repair_vals.append(float(ProxyRepairMaxSat(data=sub).calculate([crit], epsilon=epsilon)))
            tc_vals.append(float(TupleContribution(data=sub).calculate([crit], epsilon=epsilon)))

        rows_m.append({
            "criterion": crit_label,
            "MutualInformation": float(np.mean(mi_vals)),
            "ProxyMutualInformationTVD": float(np.mean(tvd_vals)),
            "ProxyRepairMaxSat": float(np.mean(repair_vals)),
            "TupleContribution": float(np.mean(tc_vals)),
        })

    df_measures = pd.DataFrame(rows_m)
    _write_csv_strict(df_measures, outfile_csv_measures)


def run_experiment_10(
    step: float = 0.1,
    n_per_sex: int = 100000,
    repetitions: int = 10,
    outfile: str = "plots/experiment10.png",
):
    """
    Experiment 10 (synthetic, NO-NOISE):
    Measure *true* MutualInformation (MI) vs ProxyMutualInformationTVD (TVD proxy)
    on the same synthetic 2x2 dataset used in run_experiment_7, while we increase unfairness.

    - We keep the same unfairness construction:
        Start fair: P(income=1|sex=0)=0.5, P(income=1|sex=1)=0.5
        Increase t in [0,1] by 'step':
            flip t fraction of (male, income=0) -> income=1
            flip t fraction of (female, income=1) -> income=0
      At t=1: males always income=1, females always income=0.

    - For each t, we compute over 'repetitions':
        * MI(sex ; income>50K) with epsilon=None
        * TVD proxy on [sex, income>50K] with epsilon=None
        * DP-gap (as a sanity x-axis / interpretability)
      We plot MI and TVD vs DP-gap, with min..max shadow bands.
    """

    # --- helpers -------------------------------------------------
    def _make_dataset(t: float) -> pd.DataFrame:
        """
        Create synthetic dataset for a given unfairness level t in [0,1].
        sex: 1=male, 0=female
        income: 1=income>50K, 0=otherwise
        """
        n = int(n_per_sex)

        # Base fair allocation:
        m0 = n // 2
        m1 = n - m0
        f0 = n // 2
        f1 = n - f0

        flip_m = int(round(t * m0))  # male: 0 -> 1
        flip_f = int(round(t * f1))  # female: 1 -> 0

        m0_new = m0 - flip_m
        m1_new = m1 + flip_m
        f1_new = f1 - flip_f
        f0_new = f0 + flip_f

        sex = np.concatenate([
            np.ones(m0_new + m1_new, dtype=int),   # males = 1
            np.zeros(f0_new + f1_new, dtype=int),  # females = 0
        ])
        income = np.concatenate([
            np.concatenate([np.zeros(m0_new, dtype=int), np.ones(m1_new, dtype=int)]),
            np.concatenate([np.zeros(f0_new, dtype=int), np.ones(f1_new, dtype=int)]),
        ])

        df = pd.DataFrame({"sex": sex, "income>50K": income})
        df = df.sample(frac=1.0, replace=False).reset_index(drop=True)
        return df

    def _dp_gap(df: pd.DataFrame) -> float:
        rates = df.groupby("sex")["income>50K"].mean()
        return float(abs(rates.max() - rates.min())) if len(rates) else 0.0

    # --- grid ----------------------------------------------------
    ts = [round(i * step, 10) for i in range(int(1 / step) + 1)]
    criterion = ["sex", "income>50K"]

    # store per-t stats
    mi_stats = {"mean": [], "min": [], "max": []}
    tvd_stats = {"mean": [], "min": [], "max": []}
    dp_stats = {"mean": [], "min": [], "max": []}

    # --- run -----------------------------------------------------
    for t in ts:
        mi_rep = []
        tvd_rep = []
        dp_rep = []

        for _ in range(repetitions):
            df = _make_dataset(t)

            # NO NOISE: epsilon=None
            mi_val = MutualInformation(data=df[criterion].copy()).calculate([criterion], epsilon=None)
            tvd_val = ProxyMutualInformationTVD(data=df[criterion].copy()).calculate([criterion], epsilon=None)

            mi_rep.append(float(mi_val))
            tvd_rep.append(float(tvd_val))
            dp_rep.append(_dp_gap(df))

        mi_arr = np.asarray(mi_rep, dtype=float)
        tvd_arr = np.asarray(tvd_rep, dtype=float)
        dp_arr = np.asarray(dp_rep, dtype=float)

        mi_stats["mean"].append(float(np.nanmean(mi_arr)))
        mi_stats["min"].append(float(np.nanmin(mi_arr)))
        mi_stats["max"].append(float(np.nanmax(mi_arr)))

        tvd_stats["mean"].append(float(np.nanmean(tvd_arr)))
        tvd_stats["min"].append(float(np.nanmin(tvd_arr)))
        tvd_stats["max"].append(float(np.nanmax(tvd_arr)))

        dp_stats["mean"].append(float(np.nanmean(dp_arr)))
        dp_stats["min"].append(float(np.nanmin(dp_arr)))
        dp_stats["max"].append(float(np.nanmax(dp_arr)))

        print(
            f"t={t:.1f}  DP-gap≈{dp_stats['mean'][-1]:.3f}  "
            f"MI≈{mi_stats['mean'][-1]:.6f}  TVD≈{tvd_stats['mean'][-1]:.6f}"
        )

    # --- plot ----------------------------------------------------
    plt.rcParams.update({
        "axes.titlesize": 20,
        "axes.labelsize": 30,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.asarray(dp_stats["mean"], dtype=float)

    # MI line + shadow
    mi_mean = np.asarray(mi_stats["mean"], dtype=float)
    mi_min  = np.asarray(mi_stats["min"], dtype=float)
    mi_max  = np.asarray(mi_stats["max"], dtype=float)

    line_mi, = ax.plot(x, mi_mean, marker="o", linewidth=2, label="MutualInformation (no noise)")
    mask_mi = ~np.isnan(x) & ~np.isnan(mi_min) & ~np.isnan(mi_max)
    if mask_mi.any():
        ax.fill_between(
            x[mask_mi], mi_min[mask_mi], mi_max[mask_mi],
            alpha=0.2, color=line_mi.get_color(), linewidth=0
        )

    # TVD proxy line + shadow
    tvd_mean = np.asarray(tvd_stats["mean"], dtype=float)
    tvd_min  = np.asarray(tvd_stats["min"], dtype=float)
    tvd_max  = np.asarray(tvd_stats["max"], dtype=float)

    line_tvd, = ax.plot(x, tvd_mean, marker="o", linewidth=2,
                        label="ProxyMutualInformationTVD (no noise)")
    mask_tvd = ~np.isnan(x) & ~np.isnan(tvd_min) & ~np.isnan(tvd_max)
    if mask_tvd.any():
        ax.fill_between(
            x[mask_tvd], tvd_min[mask_tvd], tvd_max[mask_tvd],
            alpha=0.2, color=line_tvd.get_color(), linewidth=0
        )

    ax.set_xlabel("Demographic Parity gap")
    ax.set_ylabel("measure value")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("MI vs TVD proxy on synthetic data (no noise)")

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # create_plot_0()
    # create_plot_1()
    # create_plot_2()
    # create_plot_3()
    # create_plot_4()
    # plot_legend()
    # run_experiment_1()
    # run_experiment_2()
    # run_experiment_3()
    # run_experiment_4()
    # run_experiment_5()
    # run_experiment_6()
    # run_experiment_7()
    # run_experiment_8()
    run_experiment_9()
    run_experiment_8()
    # run_experiment_10()
