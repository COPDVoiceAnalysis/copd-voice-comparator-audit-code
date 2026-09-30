"""LMM sensitivity analysis: refit on logit-transformed wAUROC and apply BH correction.

Reproduces the LMM specification used in `notebooks/lung_disease_runs_analysis.py`
(lines 833-842), then refits with logit-transformed wAUROC as the response, and
prints a side-by-side table of fixed-effect coefficients, p-values, BH-adjusted
p-values, and rank-correlation between the two model parameterisations.

Cohorts: defaults to PSM-matched Charité (primary per CLAUDE.md). Pass
``--cohort complete`` to rerun on the full-Charité sensitivity cohort.

Run from project root:
    uv run python scripts/lmm_logit_sensitivity.py                  # PSM (primary)
    uv run python scripts/lmm_logit_sensitivity.py --cohort complete  # full Charité
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.special import logit
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

COHORT_PARQUETS = {
    "psm": "data/experiment_runs/copd_classification_charite_psm_2025-08-13-v2.parquet",
    "complete": "data/experiment_runs/copd_classification_charite_only_2025-08-13-v2.parquet",
}

FORMULA = (
    "{response} ~ C(recording_type, Treatment(reference='a'))"
    " + C(feature_family, Treatment(reference='egemaps'))"
    " + C(classifier_name, Treatment(reference='LogisticRegression'))"
    " + C(norm, Treatment(reference='norm_off'))"
    " + C(scope, Treatment(reference='voice_only'))"
)

DEMO_SCOPES = {"age_sex_only", "comorbidity_only", "age_sex_comorbidity"}


def _strip_prefixes(runs_raw: pd.DataFrame) -> pd.DataFrame:
    """Mirror the column-renaming step from notebooks/lung_disease_runs_analysis.py."""
    bare = set(runs_raw.columns)
    param_renames = {
        c: c.replace("params.", "")
        for c in runs_raw.columns
        if c.startswith("params.") and c.replace("params.", "") not in bare
    }
    metric_renames = {
        c: c.replace("metrics.", "")
        for c in runs_raw.columns
        if c.startswith("metrics.") and c.replace("metrics.", "") not in bare
    }
    df = runs_raw.rename(columns={**param_renames, **metric_renames})
    df = df.rename(columns={"tags.mlflow.runName": "run_name"})
    return df


def _parse_run_name(runs: pd.DataFrame) -> pd.DataFrame:
    """Split run_name into norm/scope/feat_variant/splitting (notebook lines 50-66)."""
    parts = runs["run_name"].str.split("__", expand=True)
    n_parts = parts.shape[1]
    is_demo = parts[0].isin(DEMO_SCOPES)

    runs = runs.copy()
    runs["norm"] = parts[0].where(~is_demo, other="")
    runs["scope"] = parts[1].where(~is_demo, other=parts[0])
    runs["feat_variant"] = parts[2].where(~is_demo, other=parts[1])
    if n_parts > 5:
        runs["splitting"] = parts[5].where(~is_demo, other="")
    return runs


def _parse_feat_variant(feat_variant: str, splitting: str) -> tuple[str | None, str | None]:
    feat_variant = str(feat_variant)
    splitting = str(splitting)
    if feat_variant.startswith("wav2vec_"):
        rec = feat_variant.replace("wav2vec_", "")
        if rec == "poem" and splitting == "split":
            rec = "poem_chunked"
        return rec, "wav2vec2"
    if feat_variant.startswith("parsel_"):
        rest = feat_variant[len("parsel_"):]
        for fam in ("all_features", "basic_acoustics", "egemaps", "spectral", "temporal"):
            if rest.endswith(fam):
                rec = rest[: -(len(fam) + 1)]
                if rec == "poem" and splitting == "split":
                    rec = "poem_chunked"
                return rec, fam
    return None, None


def _explode_fold_cache(runs: pd.DataFrame) -> pd.DataFrame:
    wide = runs[["run_id", "fold_aurocs", "fold_weighted_aurocs"]].copy()
    wide = wide[wide["fold_aurocs"].apply(lambda x: x is not None and len(x) > 0)]
    fold = wide.explode(["fold_aurocs", "fold_weighted_aurocs"]).reset_index(drop=True)
    fold = fold.rename(columns={"fold_aurocs": "auroc", "fold_weighted_aurocs": "weighted_auroc"})
    fold["auroc"] = pd.to_numeric(fold["auroc"], errors="coerce")
    fold["weighted_auroc"] = pd.to_numeric(fold["weighted_auroc"], errors="coerce")
    return fold


def build_lmm_frame(parquet: str) -> pd.DataFrame:
    raw = pd.read_parquet(parquet)
    runs = _strip_prefixes(raw)
    runs = _parse_run_name(runs)

    # Filter to FINISHED + extended hyperparam, dedupe by run_name (newest), as in notebook.
    fin_all = runs[(runs["status"] == "FINISHED") & (runs["tags.hyperparam"] == "extended")].copy()
    fin = (
        fin_all.sort_values("start_time", ascending=False)
        .drop_duplicates(subset=["run_name"], keep="first")
    )

    fold = _explode_fold_cache(fin)

    meta_cols = ["run_id", "feat_variant", "scope", "norm", "splitting", "classifier_name"]
    meta = fin[meta_cols].copy()
    long = fold.merge(meta, on="run_id", how="left")

    parsed = long.apply(
        lambda r: pd.Series(
            _parse_feat_variant(r["feat_variant"], r.get("splitting", "")),
            index=["recording_type", "feature_family"],
        ),
        axis=1,
    )
    long = pd.concat([long, parsed], axis=1)

    # Restrict to the canonical 6-level recording_type x 6-level feature_family
    # design used in the primary LMM (Table 4 / sec:results_lmm). Composite
    # wav2vec+parsel feature runs (e.g. wav2vec_poem_parsel_o_basic_acoustics)
    # parse to a degenerate recording_type level and are excluded from the
    # primary analysis; we mirror that exclusion here for an apples-to-apples
    # sensitivity check.
    _VALID_RECTYPES = {"a", "i", "o", "all_vowels", "poem", "poem_chunked"}
    _VALID_FAMILIES = {"all_features", "basic_acoustics", "egemaps",
                        "spectral", "temporal", "wav2vec2"}
    df = long[
        (long["scope"].isin(["voice_only", "voice_plus_age_sex_comorbidity"]))
        & long["recording_type"].isin(_VALID_RECTYPES)
        & long["feature_family"].isin(_VALID_FAMILIES)
    ].copy()
    df = df.dropna(subset=["weighted_auroc"])

    # Clamp wAUROC away from {0,1} so logit is finite.
    eps = 1e-3
    df["weighted_auroc"] = df["weighted_auroc"].clip(eps, 1 - eps)
    df["weighted_auroc_logit"] = logit(df["weighted_auroc"].to_numpy())

    for col in ["recording_type", "feature_family", "classifier_name", "norm", "scope"]:
        df[col] = df[col].astype("category")

    return df


def fit(df: pd.DataFrame, response: str):
    formula = FORMULA.format(response=response)
    model = mixedlm(formula, df, groups=df["run_id"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(method="lbfgs", reml=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LMM logit-sensitivity analysis on a canonical Charité parquet.",
    )
    parser.add_argument(
        "--cohort",
        choices=sorted(COHORT_PARQUETS.keys()),
        default="psm",
        help="Cohort to analyse: 'psm' (primary, default) or 'complete' (full-Charité sensitivity).",
    )
    args = parser.parse_args()
    parquet = COHORT_PARQUETS[args.cohort]

    print(f"=== Cohort: {args.cohort} ({parquet}) ===")
    df = build_lmm_frame(parquet)
    n_obs, n_runs = len(df), df["run_id"].nunique()
    print(f"LMM dataset: {n_obs:,} fold observations from {n_runs} runs")
    print(f"wAUROC range: [{df['weighted_auroc'].min():.4f}, {df['weighted_auroc'].max():.4f}]")
    print()

    res_raw = fit(df, "weighted_auroc")
    res_logit = fit(df, "weighted_auroc_logit")

    raw_table = pd.DataFrame({
        "estimate_raw": res_raw.params,
        "p_raw": res_raw.pvalues,
    })
    logit_table = pd.DataFrame({
        "estimate_logit": res_logit.params,
        "p_logit": res_logit.pvalues,
    })
    out = raw_table.join(logit_table, how="inner")
    fx_mask = ~out.index.isin(["Intercept", "Group Var"])
    out_fx = out.loc[fx_mask].copy()

    out_fx["p_raw_BH"] = multipletests(out_fx["p_raw"], method="fdr_bh")[1]
    out_fx["p_logit_BH"] = multipletests(out_fx["p_logit"], method="fdr_bh")[1]
    out_fx["sign_match"] = np.sign(out_fx["estimate_raw"]) == np.sign(out_fx["estimate_logit"])

    print("Fixed-effect coefficients (raw vs logit-transformed wAUROC):")
    fmt = pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:.4g}".format,
    )
    with fmt:
        print(out_fx.to_string())

    print()
    rank_corr = out_fx["estimate_raw"].rank().corr(out_fx["estimate_logit"].rank(), method="spearman")
    print(f"Spearman rank correlation of estimates (raw vs logit): {rank_corr:.4f}")
    print(f"All signs match: {out_fx['sign_match'].all()}")
    print()
    p_threshold = 0.05
    n_sig_raw = (out_fx["p_raw"] < p_threshold).sum()
    n_sig_logit = (out_fx["p_logit"] < p_threshold).sum()
    n_sig_raw_bh = (out_fx["p_raw_BH"] < p_threshold).sum()
    n_sig_logit_bh = (out_fx["p_logit_BH"] < p_threshold).sum()
    print(f"# coefficients with p < 0.05 (raw):   uncorrected={n_sig_raw}, BH-adjusted={n_sig_raw_bh}")
    print(f"# coefficients with p < 0.05 (logit): uncorrected={n_sig_logit}, BH-adjusted={n_sig_logit_bh}")
    print()
    print("Variance components:")
    print(f"  raw model: Group Var = {res_raw.cov_re.iloc[0,0]:.6f}, Residual = {res_raw.scale:.6f}")
    print(f"  logit model: Group Var = {res_logit.cov_re.iloc[0,0]:.6f}, Residual = {res_logit.scale:.6f}")
    icc_raw = res_raw.cov_re.iloc[0,0] / (res_raw.cov_re.iloc[0,0] + res_raw.scale)
    icc_logit = res_logit.cov_re.iloc[0,0] / (res_logit.cov_re.iloc[0,0] + res_logit.scale)
    print(f"  ICC raw: {icc_raw:.4f}, ICC logit: {icc_logit:.4f}")


if __name__ == "__main__":
    main()
