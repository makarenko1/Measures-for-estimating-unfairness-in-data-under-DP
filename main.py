import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D as MplLine2D
from matplotlib.ticker import LogLocator, FuncFormatter
from opacus import PrivacyEngine
from diffprivlib.models import RandomForestClassifier
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
        data_full = _encode_and_clean(path, df.columns)
        n = min(num_tuples, len(data_full))

        sum_tvd = 0.0
        sum_repair = 0.0
        sum_tc = 0.0
        sum_dp_rf = 0.0
        sum_dp_nn = 0.0
        sum_loss_rf = 0.0
        sum_loss_nn = 0.0

        for rep in range(repetitions):
            sample = data_full.sample(n=n, replace=False)
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
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
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
    plt.savefig(dp_outfile_rf, dpi=256, bbox_inches="tight")
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
    plt.savefig(dp_outfile_nn, dpi=256, bbox_inches="tight")
    plt.show()


def create_plot_1():
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

    plt.savefig("plots/plot1.png", dpi=220)
    plt.show()


def create_plot_2():
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

    plt.savefig("plots/plot2.png", dpi=220)
    plt.show()


######################################### Experiments ##########################################

measures = {
    "Proxy Mutual Information TVD": ProxyMutualInformationTVD,
    "Proxy RepairMaxSat": ProxyRepairMaxSat,
    "Tuple Contribution": TupleContribution,
}

timeout_seconds = 1 * 60 * 60

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
    df = df.replace(["NA", "N/A", ""], pd.NA).dropna(subset=cols).copy()

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
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))

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
    plt.savefig(outfile, dpi=180, bbox_inches="tight")
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
                if (flag_timeout or (measure_name == "Proxy RepairMaxSat" and
                                    ((path == "data/census.csv" and num_tuples > 300000) or
                                    (path == "data/stackoverflow.csv" and num_tuples > 40000)))):
                    # Skip because Stackoverflow times out on more than 40K tuples and Census more than 300K
                    print("Skipping iteration due to timeout.")
                    results[measure_name]["mean"].append(np.nan)
                    results[measure_name]["min"].append(np.nan)
                    results[measure_name]["max"].append(np.nan)
                    continue

                runtimes_rep = []
                for _ in range(repetitions):
                    n = min(num_tuples, len(data))
                    sample = data.sample(n=n, replace=False)
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
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_2(
    epsilon=1.0,
    num_tuples=100000,
    repetitions=3,
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
        results = {
            measure_name: {"mean": [], "min": [], "max": []}
            for measure_name in measures.keys()
        }

        for measure_name, measure_cls in measures.items():
            flag_timeout = False

            for num_criteria in range(1, len(criteria) + 1):
                if flag_timeout or (measure_name == "Proxy RepairMaxSat" and
                                    (path == "data/stackoverflow.csv" and num_tuples > 40000 and num_criteria > 3)):
                    # Skip because Stackoverflow times out on more than 40K tuples
                    print("Skipping next iterations because got timeout for smaller number of criteria.")
                    results[measure_name]["mean"].append(np.nan)
                    results[measure_name]["min"].append(np.nan)
                    results[measure_name]["max"].append(np.nan)
                    continue

                runtimes_rep = []
                for _ in range(repetitions):
                    n = min(num_tuples, len(data))
                    sample = data.sample(n=n, replace=False)
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
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_3(
    epsilons=(0.1, 1, 5, 10),
    num_tuples=100000,
    repetitions=3,
    outfile="plots/experiment3.png"
):
    """Relative L1 error as function of epsilon."""

    def _rel_error(x, y, tiny=1e-100):
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
        data_full = _encode_and_clean(path, cols_list)
        results = {
            measure_name: {"mean": [], "min": [], "max": []}
            for measure_name in measures.keys()
        }

        for measure_name, measure_cls in measures.items():
            flag_timeout = False
            if flag_timeout or (measure_name == "Proxy RepairMaxSat" and
                                (path == "data/stackoverflow.csv" and num_tuples > 40000)):
                # Skip because Stackoverflow times out on more than 40K tuples
                for _ in epsilons:
                    results[measure_name]["mean"].append(np.nan)
                    results[measure_name]["min"].append(np.nan)
                    results[measure_name]["max"].append(np.nan)
                continue

            errs_per_eps = [[] for _ in epsilons]
            n = min(num_tuples, len(data_full))
            for _ in range(repetitions):
                if n < len(data_full):
                    sample = data_full.sample(n=n, replace=False)
                else:
                    sample = data_full
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
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
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
        df_full = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(df_full))
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
            if n < len(df_full):
                df = df_full.sample(n=n, replace=False)
            else:
                df = df_full
            mi_measure = MutualInformation(data=df)
            tvd_measure = ProxyMutualInformationTVD(data=df)

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
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_5(
    num_tuples=100000,
    repetitions=10,
    epsilon=None,
    outfile="plots/experiment5.png",
):
    """TupleContribution value as function of k, sampling separately for each repetition."""

    ks_per_dataset = {
        "Adult": [100, 500, 1000, 5000, 10000, 15000, 30000],
        "IPUMS-CPS": [1000, 5000, 10000, 50000, 100000, 300000, 600000, 1000000],
        "Stackoverflow": [1000, 5000, 10000, 20000, 40000, 60000],
        "Compas": [100, 500, 1000, 1500, 3000, 7000, 10000],
        "Healthcare": [100, 200, 400, 700, 1000],
    }

    plt.rcParams.update({
        "axes.titlesize": 34,
        "axes.labelsize": 28,
        "xtick.labelsize": 19,
        "ytick.labelsize": 22,
        "figure.titlesize": 34,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    tiny = 1e-100  # to avoid issues with log(0)

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        ks_this_dataset = ks_per_dataset[ds_name]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)
        stats = {"mean": [], "min": [], "max": []}

        flag_timeout = False
        for k in ks_this_dataset:
            if flag_timeout:
                print("Skipping the iteration due to timeout.")
                stats["mean"].append(np.nan)
                stats["min"].append(np.nan)
                stats["max"].append(np.nan)
                continue

            values_rep = []
            for _ in range(repetitions):
                n = min(num_tuples, len(data))
                sample = data.sample(n=n, replace=False)
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
                mean_v = min_v = max_v = np.nan
            else:
                mean_v = vals.mean()
                min_v = vals.min()
                max_v = vals.max()
            stats["mean"].append(mean_v)
            stats["min"].append(min_v)
            stats["max"].append(max_v)

        xs = np.arange(len(ks_this_dataset))
        means = np.array(stats["mean"], dtype=float)
        lows  = np.array(stats["min"], dtype=float)
        highs = np.array(stats["max"], dtype=float)
        means = np.clip(means, tiny, None)
        lows  = np.clip(lows, tiny, None)
        highs = np.clip(highs, tiny, None)
        line, = ax.plot(xs, means, marker="o", linewidth=2,
                        label="TupleContribution value")
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
        for k in ks_this_dataset:
            if k >= 1_000_000:
                tick_labels.append(f"{k // 1_000_000}M")
            elif k >= 1_000:
                tick_labels.append(f"{k // 1_000}K")
            else:
                tick_labels.append(str(k))
        if ds_name == "IPUMS-CPS":
            show_idx = [0, 2, 4, 6] if len(ks_this_dataset) >= 7 else list(range(len(ks_this_dataset)))
            ax.set_xticks(np.array(show_idx))
            ax.set_xticklabels([tick_labels[i] for i in show_idx])
        else:
            ax.set_xticks(xs)
            ax.set_xticklabels(tick_labels)
        ax.set_xlabel("k (top-k tuples)")
        ax.set_yscale('log')
        if ds_name == "Healthcare":
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("TupleContribution (log scale)")
    fig.suptitle(
        f"TupleContribution value as function of k, at most {round(num_tuples / 1000)}K tuples",
        y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_6(
    num_tuples=100000,
    repetitions=10,
    epsilon=1.0,
    outfile="plots/experiment6.png",
):
    """Relative L1 error of TupleContribution as function of k."""

    ks_per_dataset = {
        "Adult": [500, 1000, 5000, 10000, 15000, 30000],
        "IPUMS-CPS": [1000, 5000, 10000, 50000, 100000, 300000, 600000, 1000000],
        "Stackoverflow": [1000, 5000, 10000, 20000, 40000, 60000],
        "Compas": [500, 1000, 1500, 3000, 7000, 10000],
        "Healthcare": [100, 200, 400, 700, 1000],
    }

    def _rel_error(x, y, tiny=1e-100):
        denom = max(abs(y), tiny)  # ensure we do not divide by 0
        return abs(x - y) / denom

    plt.rcParams.update({
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
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
        stats = {"mean": [], "min": [], "max": []}

        for k in ks:
            errs = []

            for _ in range(repetitions):
                sample = data.sample(n=n, replace=False)
                m = TupleContribution(data=sample)
                with ThreadPoolExecutor() as executor:
                    try:
                        non_private_result = executor.submit(
                            m.calculate,
                            criteria,
                            k=k,
                            epsilon=None
                        ).result(timeout=timeout_seconds)
                    except TimeoutError:
                        print("Skipping iteration due to timeout.")
                        errs.append(np.nan)
                with ThreadPoolExecutor() as executor:
                    try:
                        private_result = executor.submit(
                            m.calculate,
                            criteria,
                            k=k,
                            epsilon=epsilon
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

        x = np.arange(len(ks))
        means = np.array(stats["mean"])
        lows  = np.array(stats["min"])
        highs = np.array(stats["max"])
        line, = ax.plot(x, means, marker="o", linewidth=2,
                        label="TupleContribution L1 error")
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
        full_tick_labels = [str(k) if k % 1000 != 0 else f"{k // 1000}K" for k in ks]
        if ds_name == "IPUMS-CPS":
            show_idx = [0, 2, 4, 6] if len(ks) >= 7 else list(range(len(ks)))
            ax.set_xticks(np.array(show_idx))
            ax.set_xticklabels([full_tick_labels[i] for i in show_idx])
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(full_tick_labels)
        ax.set_xlabel("k (top-k tuples)")
        ax.set_yscale('log')
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("relative L1 error (log scale)")
    fig.suptitle(
        f"Relative L1 Error of TupleContribution as Function of k, at most {round(num_tuples / 1000)}K tuples, "
        f"ε = {epsilon}",
        y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_7(
        epsilon: float = 1.0,
        num_tuples: int = 1000,
        repetitions: int = 10,
        outfile: str = "plots/experiment7.png",
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
        df_full = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(df_full))
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
            if n < len(df_full):
                df = df_full.sample(n=n, replace=False)
            else:
                df = df_full
            repair_regular = ProxyRepairMaxSat(data=df)
            repair_with_chunks = ProxyRepairMaxSat(data=df)

            for criterion in criteria:
                if len(criterion) == 3:
                    protected, response, admissible = criterion
                    crit_label = f"{protected} , {response} | {admissible}"
                else:
                    protected, response = criterion[0], criterion[1]
                    crit_label = f"{protected} , {response}"
                with ThreadPoolExecutor() as executor:
                    try:
                        if path == "data/stackoverflow.csv":  # Stackoverflow times out
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
        f"Comparison of regular RepairMaxSAT and RepairMaxSAT with chunks of size 100, {round(num_tuples / 1000)}K tuples",
        y=1.03,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_8(
    epsilon: Optional[float] = None,
    n_per_sex: int = 500,                 # 500 males, 500 females
    step: float = 0.1,                    # switch 10% each iteration
    repetitions: int = 10,
    outfile: str = "plots/experiment8.png",
):
    """
    Plot measure values vs growing unfairness on a synthetic dataset with two columns:
      sex in {0,1} and income>50K in {0,1}.
    Start fair: for each sex, 50% income=0 and 50% income=1.
    Then, at each step t = 0, step, 2*step, ..., 1.0:
      - flip a fraction t of (male, income=0) to income=1
      - flip a fraction t of (female, income=1) to income=0
    So at t=1.0: all males income=1, all females income=0.

    Uses your global `measures` dict (ProxyMutualInformationTVD, ProxyRepairMaxSat, TupleContribution).
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

        # After flipping:
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

        # Shuffle rows (important for any measure that might be order-sensitive)
        df = df.sample(frac=1.0, replace=False).reset_index(drop=True)
        return df

    def _dp_gap(df: pd.DataFrame) -> float:
        """Demographic parity gap on the raw labels (no model): |P(y=1|male)-P(y=1|female)|."""
        rates = df.groupby("sex")["income>50K"].mean()
        return float(abs(rates.max() - rates.min())) if len(rates) else 0.0

    # ---- build unfairness grid ----------------------------------
    # If you want exactly: start + 10 flips until fully polarized, use t in {0.0,0.1,...,1.0}
    ts = [round(i * step, 10) for i in range(int(1 / step) + 1)]
    # Optionally truncate/override with num_steps if you want a different count:
    # ts = [round(i * step, 10) for i in range(num_steps + 1)]

    criterion = ["sex", "income>50K"]

    # Store results: measure_name -> list over t
    results = {name: [] for name in measures.keys()}
    dp_gaps = []

    # ---- run -----------------------------------------------------
    for t in ts:
        vals_rep = {name: [] for name in measures.keys()}
        dp_rep = []

        for _ in range(repetitions):
            df = _make_dataset(t)

            # All your measures take `data=...` and `calculate([criterion], epsilon=...)`
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

        # Aggregate
        for measure_name in measures.keys():
            arr = np.asarray(vals_rep[measure_name], dtype=float)
            arr = arr[~np.isnan(arr)]
            results[measure_name].append(float(arr.mean()) if arr.size else np.nan)

        dp_arr = np.asarray(dp_rep, dtype=float)
        dp_gaps.append(float(dp_arr.mean()) if dp_arr.size else np.nan)

        print(f"t={t:.1f}  DP-gap≈{dp_gaps[-1]:.3f}  " +
              "  ".join([f"{mn}≈{results[mn][-1]:.4f}" for mn in measures.keys()]))

    # ---- plot ----------------------------------------------------
    plt.rcParams.update({
        "axes.titlesize": 25,
        "axes.labelsize": 28,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "figure.titlesize": 34,
        "legend.fontsize": 18,
    })

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.asarray(ts, dtype=float)
    for measure_name in measures.keys():
        y = np.asarray(results[measure_name], dtype=float)
        ax.plot(x, y, marker="o", linewidth=2, label=measure_name)

    # Optional: also show DP-gap (ground-truth unfairness) on a second axis
    ax2 = ax.twinx()
    ax2.plot(x, np.asarray(dp_gaps, dtype=float), marker="s", linewidth=2, linestyle="--", label="DP-gap")
    ax2.set_ylabel("DP-gap")
    ax2.yaxis.set_major_formatter(y_formatter)

    # ax.set_title("Measure Values vs Growing Unfairness (Synthetic sex × income)")
    ax.set_xlabel("flip fraction (0 = fair, 1 = fully unfair)")
    ax.set_yscale('log')
    ax.set_ylabel("measure value (log scale)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(y_formatter)

    # Merge legends from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # ax.legend(h1 + h2, l1 + l2, loc="lower right", frameon=True)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # create_plot_0()
    # create_plot_1()
    # create_plot_2()
    # plot_legend()
    # run_experiment_1()
    # run_experiment_2()
    # run_experiment_3()
    # run_experiment_4()
    # run_experiment_5()
    # run_experiment_6()
    # run_experiment_7()
    run_experiment_8()
