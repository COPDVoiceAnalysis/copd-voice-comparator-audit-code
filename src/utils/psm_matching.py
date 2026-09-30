import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold


def compute_propensity_scores(
    df: pd.DataFrame,
    covariate_cols: list[str],
    use_age: bool = True,
    include_age_squared: bool = True,
    random_state: int = 42,
    individual_id_col: str | None = None,
) -> tuple[pd.DataFrame, LogisticRegression]:
    """Fit sparse logistic (L1) model to estimate P(label=1|X).

    Adds:
      ps        : predicted probability
      logit_ps  : logit transformation (used for caliper distance)
    Robustness tweaks:
      - Ignore covariates not present.
      - Replace non-finite logits with median.

    ``include_age_squared`` (default True) controls whether a quadratic age term
    (``age_z2``) is added alongside the linear ``age_z``. The Charité age-only
    PSM uses ``include_age_squared=False`` (linear age only): with the small
    control group, the quadratic term changes the propensity ranking enough to
    alter which cases match, so it is disabled there for a monotonic-in-age PS.
    """
    use_cols = [c for c in covariate_cols if c in df.columns]
    if use_age:
        if not "age_z" in df.columns:
            df["age_z"] = (df["age"] - df["age"].mean()) / (
                df["age"].std(ddof=0) + 1e-9
            )
        use_cols += ["age_z"]
        if include_age_squared:
            if not "age_z2" in df.columns:
                df["age_z2"] = df["age_z"] ** 2
            use_cols += ["age_z2"]
    df_original = df.copy()
    X_original = df_original[use_cols].copy()
    if individual_id_col is not None:
        df = df.drop_duplicates(subset=individual_id_col).reset_index(drop=True).copy()
    X = df[use_cols].copy()
    for c in use_cols:
        X[c] = X[c].fillna(0).astype(float)
    y = (df["label"] == 1).astype(int)

    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=1.0,
        max_iter=2000,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X, y)

    ps = clf.predict_proba(X_original)[:, 1]
    z = np.log(ps / (1 - ps))
    z[~np.isfinite(z)] = np.median(z[np.isfinite(z)])

    out = df_original.copy()
    out["ps"] = ps
    out["logit_ps"] = z
    return out, clf


def trim_common_support(g: pd.DataFrame) -> pd.DataFrame:
    """Restrict stratum rows to overlapping propensity score range between groups.

    Prevents extrapolation when one group has extreme PS values not shared by the other.
    """
    lo_t = g.loc[g.label == 1, "ps"].min()
    hi_t = g.loc[g.label == 1, "ps"].max()
    lo_c = g.loc[g.label == 0, "ps"].min()
    hi_c = g.loc[g.label == 0, "ps"].max()
    lo, hi = max(lo_t, lo_c), min(hi_t, hi_c)
    return g[(g.ps >= lo) & (g.ps <= hi)]


def match_within_strata(
    df_with_ps: pd.DataFrame,
    exact_keys: list[str],
    k: int,
    caliper_coef: float,
    age_caliper_years: int | None = None,
    target_col: str = "label",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Greedy 1:k matching within exact strata on logit-PS with optional age constraint.

    Process overview (per stratum):
      1. (Optional) Build stratum id from exact_keys (e.g. sex|language) else single pool.
      2. Trim to *common support* in raw propensity score space to avoid extrapolation.
      3. Compute a local logit-PS standard deviation (fallback to global) to scale caliper.
      4. For each treated (label=1) row:
         a. Filter controls by absolute age difference if age_caliper_years set.
         b. Fit k-NN on remaining controls using logit_ps (1-D distance).
         c. Traverse nearest controls in order; retain those within distance <= caliper.
         d. Enforce without-replacement inside stratum (control can appear once).
         e. If no control within caliper: increment drop reason and continue.
      5. Accumulate pair_id for each treated + its matched controls.

    Drop reasons recorded in drop_report:
      no_opposite_label            : Stratum has only cases or only controls.
      no_support                   : After common support trim still lacks both groups.
      empty_after_filter           : Defensive catch (should be rare) when either side empty.
      no_ctrl_within_age_caliper   : Treated unit had no controls inside age window.
      no_match_within_ps_caliper   : Controls exist but none within PS distance caliper.

    Returns
    -------
    matched_df : Long DataFrame of treated + matched controls with pair_id.
    drop_report: Dict[str,int] counts per drop category for diagnostics.
    """
    df = df_with_ps.copy()
    n_copd = len(df.loc[df.label == 1])

    # Build stratum identifier string (e.g., F|en); single pseudo-stratum if no exact keys.
    if exact_keys:
        df["stratum"] = df[exact_keys].astype(str).agg("|".join, axis=1)
    else:
        df["stratum"] = "_all_"

    out_rows: list[pd.Series] = []
    drop_report: dict[str, int] = defaultdict(int)
    pair_counter = 0

    # Global SD used when a local stratum SD is zero/NaN (stability for caliper sizing).
    global_sd = df["logit_ps"].std() if "logit_ps" in df else 1.0
    if not np.isfinite(global_sd) or global_sd == 0:
        global_sd = 1.0

    # Iterate each exact-match stratum independently.
    for s_name, s_df in df.groupby("stratum", dropna=False):
        # Need at least one treated and one control.
        if s_df[target_col].nunique() < 2:
            drop_report["no_opposite_label"] += len(s_df)
            continue

        # Restrict to overlapping propensity score range (common support).
        s_df = trim_common_support(s_df)
        if s_df.empty or s_df[target_col].nunique() < 2:
            drop_report["no_support"] += len(s_df)
            continue

        # Local SD for dynamic caliper width (Rosenbaum & Rubin style); fallback to global.
        local_sd = s_df["logit_ps"].std()
        if not np.isfinite(local_sd) or local_sd == 0:
            local_sd = global_sd
        cal = (
            caliper_coef * local_sd
            if np.isfinite(local_sd) and local_sd > 0
            else caliper_coef * global_sd
        )

        tdf = s_df[s_df.label == 1].copy()  # Treated (cases)
        cdf_full = s_df[s_df.label == 0].copy()  # Candidate controls
        if len(tdf) == 0 or len(cdf_full) == 0:
            drop_report["empty_after_filter"] += len(s_df)
            continue

        used_ctrl: set[Any] = (
            set()
        )  # Track controls already matched (no reuse within stratum).

        # Greedy single pass over treated units
        for _, tr in tdf.iterrows():
            # Optional absolute age difference pruning prior to k-NN.
            if age_caliper_years is not None:
                cdf = cdf_full[
                    np.abs(cdf_full["age"] - tr["age"]) <= age_caliper_years
                ].copy()
                if cdf.empty:
                    drop_report["no_ctrl_within_age_caliper"] += 1
                    continue
            else:
                cdf = cdf_full

            # Fit k-NN (k limited by available controls); distance in logit-PS space.
            nn = NearestNeighbors(
                n_neighbors=min(k * n_copd + len(used_ctrl), len(cdf))
            ).fit(cdf[["logit_ps"]])
            dists, inds = nn.kneighbors(
                pd.DataFrame([[tr["logit_ps"]]], columns=["logit_ps"]),
                return_distance=True,
            )

            chosen: list[Any] = []
            for d, j in zip(dists.ravel(), inds.ravel()):
                if d > cal:  # Outside caliper threshold.
                    continue
                ci = cdf.iloc[j].name
                if ci in used_ctrl:  # Already used in another pair for this stratum.
                    continue
                chosen.append(ci)
                used_ctrl.add(ci)
                if len(chosen) == k:  # Reached desired ratio.
                    break
            if not chosen:  # No acceptable matches inside caliper.
                drop_report["no_match_within_ps_caliper"] += 1
                continue

            # Assign a pair identifier; include stratum for easier post-hoc debugging.
            pid = f"{s_name}|{pair_counter}"
            tro = tr.copy()
            tro["pair_id"] = pid
            tro["match_ratio_k"] = len(chosen)
            out_rows.append(tro)
            for ci in chosen:
                cro = df.loc[ci].copy()
                cro["pair_id"] = pid
                cro["match_ratio_k"] = len(chosen)
                out_rows.append(cro)
            pair_counter += 1

    if not out_rows:
        raise RuntimeError("No pairs constructed; check keys/caliper.")
    matched = pd.DataFrame(out_rows)
    matched["match_ratio_k"] = matched["match_ratio_k"].astype(int)
    return matched, dict(drop_report)


def smd_cont(x1: pd.Series, x0: pd.Series) -> float:
    """Standardized mean difference for continuous features.

    Formula: (mean1 - mean0) / sqrt( (Var1 + Var0)/2 )
    A tiny constant is added to denominator to guard against division by zero.
    """
    s = np.sqrt(((x1.var() + x0.var()) / 2) + 1e-12)
    return float((x1.mean() - x0.mean()) / s) if s > 0 else 0.0


def smd_bin(p1: float, p0: float) -> float:
    """Standardized mean difference for a binary variable given proportions.

    Formula: (p1 - p0) / sqrt( p*(1-p) ), where p = (p1 + p0)/2 (pooled Bernoulli variance).
    """
    p = (p1 + p0) / 2
    return float((p1 - p0) / np.sqrt(max(p * (1 - p), 1e-12)))


def smd_table(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    bin_cols: list[str],
    cont_cols: list[str],
) -> pd.DataFrame:
    """Compute SMD before/after matching and order by absolute post-match imbalance.

    Steps:
      1. For each listed binary column: compute treated/control proportions then SMD.
      2. For each listed continuous column: compute SMD using pooled SD.
      3. Assemble a DataFrame with smd_before / smd_after.
      4. Sort descending by absolute smd_after to highlight worst residual imbalance.
    Missing columns are skipped silently (robust to differing availability pre/post).
    """

    def calc(tab: pd.DataFrame) -> pd.Series:
        t, c = tab[tab.label == 1], tab[tab.label == 0]
        out: dict[str, float] = {}
        # Binary covariates
        for col in bin_cols:
            if col not in tab.columns:
                continue
            p1 = (t[col].fillna(0).astype(int) == 1).mean()
            p0 = (c[col].fillna(0).astype(int) == 1).mean()
            out[col] = smd_bin(p1, p0)
        # Continuous covariates
        for col in cont_cols:
            if col not in tab.columns:
                continue
            out[col] = smd_cont(t[col].astype(float), c[col].astype(float))
        return pd.Series(out)

    smd = pd.DataFrame({"smd_before": calc(before_df), "smd_after": calc(after_df)})
    return smd.reindex(smd["smd_after"].abs().sort_values(ascending=False).index)


def make_folds_simple(
    matched_df: pd.DataFrame, n_folds: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Assign cross-validation folds stratified on (label × language).

    Rules:
      - Always creates a 'fold' column in [0, n_folds-1].
      - Derives a simple 'set' column: one fold -> all train; two folds -> fold 0 test; >=3 folds -> fold0 test, fold1 val, rest train.
    Deterministic due to fixed random_state.
    """
    dfm = matched_df.copy()
    y_strat = dfm["label"].astype(str) + "_" + dfm["language"].astype(str)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    dfm["fold"] = -1
    for f, (_, test_idx) in enumerate(skf.split(np.zeros(len(dfm)), y_strat)):
        dfm.loc[dfm.index[test_idx], "fold"] = f
    return dfm


def make_reference_folds(
    matched_df: pd.DataFrame, n_folds: int = 5, random_state: int = 42
):
    """Assign cross-validation folds stratified on (label × language).

    Rules:
      - Always creates a 'fold' column in [0, n_folds-1].
      - Derives a simple 'set' column: one fold -> all train; two folds -> fold 0 test; >=3 folds -> fold0 test, fold1 val, rest train.
    """
    strat_cols = ["label", "language", "sex"]
    dfm = matched_df.copy()
    y_strat = dfm[strat_cols].astype(str).agg("|".join, axis=1)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    dfm["fold"] = -1

    fold_dict = {}
    for f, (_, test_idx) in enumerate(skf.split(np.zeros(len(dfm)), y_strat)):
        for idx in test_idx:
            rec_idx = dfm.iloc[idx].recording_identifier
            try:
                fold_dict[int(rec_idx)] = f
            except TypeError:
                print("stop")
    return fold_dict


def use_reference_folds(
    matched_df: pd.DataFrame,
    reference_folds: dict,
    n_folds: int = 5,
    random_state: int = 42,
):
    """Assign cross-validation folds stratified on (label × language).

    Rules:
      - Always creates a 'fold' column in [0, n_folds-1].
      - Derives a simple 'set' column: one fold -> all train; two folds -> fold 0 test; >=3 folds -> fold0 test, fold1 val, rest train.
    """
    strat_cols = ["label", "language", "sex"]
    dfm = matched_df.copy()
    y_strat = dfm[strat_cols].astype(str).agg("|".join, axis=1)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    dfm["fold"] = -1

    fold_dict = {}
    for f, (_, test_idx) in enumerate(skf.split(np.zeros(len(dfm)), y_strat)):
        fold_dict[test_idx] = f
    return fold_dict
