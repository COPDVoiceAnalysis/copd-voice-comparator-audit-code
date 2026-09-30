"""
Thesis Results: COPD vs Control Classification (v2: chunk normalization + person-level eval)

Loads experiment runs from a local parquet snapshot (no MLFlow server needed).
Generates all tables/plots needed for the Results chapter of the master thesis.

Start from project root:
  uv run marimo edit notebooks/lung_disease_runs_analysis_psm.py
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import altair as alt
    from pathlib import Path

    return Path, alt, mo, np, pd


@app.cell
def _(Path, mo, pd):
    EXPERIMENT_NAME = "copd_classification_charite_psm_2025-08-13-v2"
    _PARQUET_PATH = Path("data/experiment_runs") / f"{EXPERIMENT_NAME}.parquet"

    _runs_raw = pd.read_parquet(_PARQUET_PATH)

    # Rename columns: strip prefixes, skip conflicts with bare columns
    _bare_cols = {_c for _c in _runs_raw.columns if not _c.startswith(("params.", "metrics.", "tags."))}
    _param_renames = {
        _c: _c.replace("params.", "")
        for _c in _runs_raw.columns
        if _c.startswith("params.") and _c.replace("params.", "") not in _bare_cols
    }
    _metric_renames = {
        _c: _c.replace("metrics.", "")
        for _c in _runs_raw.columns
        if _c.startswith("metrics.") and _c.replace("metrics.", "") not in _bare_cols
    }
    runs_all = _runs_raw.rename(columns={**_param_renames, **_metric_renames})
    runs_all = runs_all.rename(columns={"tags.mlflow.runName": "run_name"})

    # Parse run name components
    # Voice runs: norm__scope__feature__hyperparam__task__splitting__classifier (7 parts)
    # Demographics-only runs: scope__feature__hyperparam__task__classifier (5 parts)
    _parts = runs_all["run_name"].str.split("__", expand=True)
    _n_parts = _parts.shape[1]

    _DEMO_SCOPES = {"age_sex_only", "comorbidity_only", "age_sex_comorbidity"}
    _is_demo = _parts[0].isin(_DEMO_SCOPES)

    runs_all["norm"] = _parts[0].where(~_is_demo, other="")
    runs_all["scope"] = _parts[1].where(~_is_demo, other=_parts[0])
    runs_all["feat_variant"] = _parts[2].where(~_is_demo, other=_parts[1])
    if _n_parts > 5:
        runs_all["splitting"] = _parts[5].where(~_is_demo, other="")
    if _n_parts > 6:
        runs_all["clf_short"] = _parts[6].where(~_is_demo, other=(_parts[4] if _n_parts > 4 else ""))

    # FINISHED runs only; deduplicate by run_name (keep latest per config)
    _fin_all = runs_all[(runs_all["status"] == "FINISHED") & (runs_all["tags.hyperparam"] == "extended")].copy()
    _n_before_dedup = len(_fin_all)
    fin = (
        _fin_all
        .sort_values("start_time", ascending=False)
        .drop_duplicates(subset=["run_name"], keep="first")
    )
    _n_dupes = _n_before_dedup - len(fin)

    n_total = len(runs_all)
    n_finished = len(fin)
    n_running = (runs_all["status"] == "RUNNING").sum()

    mo.md(f"""
    # Thesis Results: COPD vs Control Classification

    **Experiment:** `{EXPERIMENT_NAME}`

    | Status | Count |
    |--------|-------|
    | FINISHED | {n_finished} |
    | RUNNING | {n_running} |
    | Duplicates removed | {_n_dupes} |
    | **Total (raw)** | **{n_total}** |
    """)
    MLFLOW_URI = "http://localhost:5002"
    return EXPERIMENT_NAME, MLFLOW_URI, fin


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Tabelle 1: COPD vs Control — Experiment Matrix

    Matrix of dataset/task rows x feature family columns (voice_only scope).
    Corresponds to Table 1 in the Results chapter.
    """)
    return


@app.cell
def _(mo):
    metric_choice = mo.ui.dropdown(
        label="Metric",
        options=[
            "weighted_auroc_mean",
            "auroc_mean",
            "weighted_balanced_accuracy_mean",
            "balanced_accuracy_mean",
            "weighted_mcc_mean",
            "mcc_mean",
        ],
        value="weighted_auroc_mean",
    )
    agg_choice = mo.ui.dropdown(
        label="Aggregation",
        options=["best", "mean", "mean_std"],
        value="best",
    )
    scope_choice = mo.ui.dropdown(
        label="Scope",
        options=["voice_only", "voice_plus_age_sex_comorbidity"],
        value="voice_only",
    )
    mo.hstack([metric_choice, agg_choice, scope_choice])
    return agg_choice, metric_choice, scope_choice


@app.cell
def _(agg_choice, fin, metric_choice, mo, pd, scope_choice):
    _metric = metric_choice.value
    _agg = agg_choice.value
    _scope = scope_choice.value

    # --- Feature variant -> column mapping ---
    _col_map = {
        "spectral": "Spectral",
        "basic_acoustics": "Basic acoustics",
        "egemaps": "eGeMAPS",
        "temporal": "Temporal",
        "all_features": "All acoustic",
    }

    def _map_col(feat_variant):
        _f = str(feat_variant)
        if _f.startswith("wav2vec_"):
            return "wav2vec2"
        for _suffix, _col in _col_map.items():
            if _f.endswith(_suffix):
                return _col
        return None

    # --- Feature variant -> row mapping ---
    def _map_row(feat_variant, splitting):
        _f = str(feat_variant)
        _s = str(splitting)
        if "poem" in _f:
            return "poem chunked" if _s == "split" else "poem unchunked"
        if "all_vowels" in _f:
            return "all vowels"
        if _f.startswith("parsel_a_") or _f == "wav2vec_a":
            return "vowel /a/"
        if _f.startswith("parsel_i_") or _f == "wav2vec_i":
            return "vowel /i/"
        if _f.startswith("parsel_o_") or _f == "wav2vec_o":
            return "vowel /o/"
        return None

    _df = fin[fin["scope"] == _scope].copy()
    _df["table_col"] = _df["feat_variant"].map(_map_col)
    _df["table_row"] = _df.apply(
        lambda r: _map_row(r["feat_variant"], r.get("splitting", "")), axis=1
    )
    _df = _df[_df["table_col"].notna() & _df["table_row"].notna()].copy()
    _df["_val"] = pd.to_numeric(_df[_metric], errors="coerce")

    # Also grab std column if available
    _std_col = _metric.replace("_mean", "_std")
    if _std_col in _df.columns:
        _df["_std"] = pd.to_numeric(_df[_std_col], errors="coerce")

    if _df.empty:
        results_matrix = pd.DataFrame()
    else:
        if _agg == "best":
            _agg_df = (
                _df.groupby(["table_row", "table_col"])["_val"]
                .max()
                .reset_index()
            )
            _agg_df["display"] = _agg_df["_val"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "")
        elif _agg == "mean":
            _agg_df = (
                _df.groupby(["table_row", "table_col"])["_val"]
                .mean()
                .reset_index()
            )
            _agg_df["display"] = _agg_df["_val"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "")
        else:  # mean_std
            _g = _df.groupby(["table_row", "table_col"])["_val"]
            _mean = _g.mean().reset_index(name="_val")
            _std = _g.std().reset_index(name="_s")
            _agg_df = _mean.merge(_std, on=["table_row", "table_col"])
            _agg_df["display"] = _agg_df.apply(
                lambda r: f"{r['_val']:.3f} +/- {r['_s']:.3f}" if pd.notna(r["_val"]) else "",
                axis=1,
            )

        results_matrix = _agg_df.pivot(index="table_row", columns="table_col", values="display")
        _col_order = ["Spectral", "Basic acoustics", "eGeMAPS", "Temporal", "All acoustic", "wav2vec2"]
        for _c in _col_order:
            if _c not in results_matrix.columns:
                results_matrix[_c] = ""
        results_matrix = results_matrix[[_c for _c in _col_order if _c in results_matrix.columns]]
        _row_order = [
            "vowel /a/", "vowel /i/", "vowel /o/", "all vowels",
            "poem unchunked", "poem chunked",
        ]
        _existing = [_r for _r in _row_order if _r in results_matrix.index]
        _extra = [_r for _r in results_matrix.index if _r not in _row_order]
        results_matrix = results_matrix.reindex(_existing + _extra)

    if results_matrix.empty:
        _out = mo.md("No data for this selection.")
    else:
        _out = mo.ui.table(results_matrix.fillna("").reset_index().rename(columns={"table_row": "Task"}))
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Demographics-Only Baselines (Non-Voice)

    How separable are the cohorts without any voice features?
    These baselines contextualize the voice-based results.
    """)
    return


@app.cell
def _(fin, mo, pd):
    _demo = fin[fin["demographics_only"] == "True"].copy()
    _demo["auroc_mean_num"] = pd.to_numeric(_demo["auroc_mean"], errors="coerce")

    if _demo.empty:
        _out = mo.md("No demographics-only runs found.")
    else:
        # Deduplicate: demographics runs are identical across norm/split
        _demo_dedup = (
            _demo.dropna(subset=["auroc_mean_num"])
            .sort_values("auroc_mean_num", ascending=False)
            .drop_duplicates(subset=["scope", "classifier_name"], keep="first")
        )
        _demo_cols = ["scope", "classifier_name", "auroc_mean_num", "auroc_std",
                      "weighted_auroc_mean", "weighted_auroc_std",
                      "balanced_accuracy_mean", "weighted_balanced_accuracy_mean",
                      "mcc_mean", "n_features"]
        _demo_cols = [_c for _c in _demo_cols if _c in _demo_dedup.columns]
        _tbl = _demo_dedup[_demo_cols].copy()
        _tbl.columns = ["Scope", "Classifier", "AUROC", "AUROC std",
                         "wAUROC", "wAUROC std",
                         "BA", "wBA",
                         "MCC", "n_features"][:len(_demo_cols)]
        for _c in [_col for _col in _tbl.columns if _col not in ("Scope", "Classifier", "n_features")]:
            _tbl[_c] = pd.to_numeric(_tbl[_c], errors="coerce").round(3)
        _tbl["n_features"] = pd.to_numeric(_tbl["n_features"], errors="coerce").astype("Int64")
        _out = mo.ui.table(_tbl.reset_index(drop=True))
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Voice-Only vs Voice + Age/Sex/Comorbidity

    Paired comparison: does adding age, sex, and comorbidity features improve upon voice-only models?
    Each pair is matched on feature_variant + classifier + normalization + splitting.
    """)
    return


@app.cell
def _(alt, fin, mo, pd):
    _voice = fin[fin["scope"].isin(["voice_only", "voice_plus_age_sex_comorbidity"])].copy()
    _voice["wauroc_num"] = pd.to_numeric(_voice["weighted_auroc_mean"], errors="coerce")
    _voice["key"] = (
        _voice["feat_variant"] + "__" + _voice["classifier_name"] + "__"
        + _voice["norm"] + "__" + _voice["splitting"]
    )
    _pivot = _voice.pivot_table(
        index="key", columns="scope", values="wauroc_num", aggfunc="first"
    ).dropna()

    _COMBO = "voice_plus_age_sex_comorbidity"
    if _pivot.empty or "voice_only" not in _pivot.columns or _COMBO not in _pivot.columns:
        _out = mo.md("Not enough paired data.")
    else:
        _pivot["delta"] = _pivot[_COMBO] - _pivot["voice_only"]
        _n = len(_pivot)
        _mean_d = _pivot["delta"].mean()
        _std_d = _pivot["delta"].std()
        _pos = (_pivot["delta"] > 0).sum()
        _neg = (_pivot["delta"] < 0).sum()

        _chart = alt.Chart(_pivot.reset_index()).mark_bar().encode(
            alt.X("delta:Q", bin=alt.Bin(maxbins=25), title="wAUROC delta (voice+demographics minus voice_only)"),
            alt.Y("count():Q", title="Count"),
        ).properties(width=500, height=250)
        _rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="red", strokeDash=[4, 4]).encode(x="x:Q")

        _out = mo.vstack([
            mo.md(f"""
            **{_n} matched pairs** | Mean wAUROC delta: **{_mean_d:+.4f}** +/- {_std_d:.4f}
            | voice+demographics better: {_pos}/{_n} | voice_only better: {_neg}/{_n}
            """),
            _chart + _rule,
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Feature-Family Comparison (voice_only scope)

    wAUROC distribution across feature variants. Grouped by Parselmouth vs Wav2Vec2.
    """)
    return


@app.cell
def _(alt, fin, mo, pd):
    _vo = fin[fin["scope"] == "voice_only"].copy()
    _vo["wauroc_num"] = pd.to_numeric(_vo["weighted_auroc_mean"], errors="coerce")
    _vo = _vo.dropna(subset=["wauroc_num"])
    _vo["family"] = _vo["feat_variant"].apply(
        lambda f: "wav2vec2" if str(f).startswith("wav2vec_") else "parselmouth"
    )

    if _vo.empty:
        _out = mo.md("No voice_only runs.")
    else:
        _chart = alt.Chart(_vo).mark_boxplot(extent="min-max").encode(
            x=alt.X("wauroc_num:Q", title="wAUROC (strata-weighted, mean over CV folds)", scale=alt.Scale(domain=[0.3, 0.9])),
            y=alt.Y("feat_variant:N", title="Feature Variant", sort="-x"),
            color=alt.Color("family:N", title="Family"),
        ).properties(width=550, height=500)

        # Best per feature variant table
        _best = _vo.loc[_vo.groupby("feat_variant")["wauroc_num"].idxmax()]
        _best_tbl = _best[["feat_variant", "classifier_name", "splitting", "wauroc_num", "weighted_auroc_std", "weighted_balanced_accuracy_mean", "n_features"]].copy()
        _best_tbl.columns = ["Feature", "Classifier", "Split", "wAUROC", "wAUROC std", "wBA", "n_features"]
        for _c in ["wAUROC", "wAUROC std", "wBA"]:
            _best_tbl[_c] = pd.to_numeric(_best_tbl[_c], errors="coerce").round(3)
        _best_tbl["n_features"] = pd.to_numeric(_best_tbl["n_features"], errors="coerce").astype("Int64")
        _best_tbl = _best_tbl.sort_values("wAUROC", ascending=False).reset_index(drop=True)

        _out = mo.vstack([
            _chart,
            mo.md("### Best run per feature variant"),
            mo.ui.table(_best_tbl),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Wav2Vec2 vs Parselmouth: Poem Task Detail

    All poem-task runs (voice_only scope), comparing wav2vec2 embeddings
    against the best Parselmouth feature sets on the same task.
    """)
    return


@app.cell
def _(fin, mo, pd):
    _poem_feats = [
        "wav2vec_poem", "parsel_poem_egemaps", "parsel_poem_spectral",
        "parsel_poem_basic_acoustics", "parsel_poem_all_features", "parsel_poem_temporal",
    ]
    _vo = fin[
        (fin["scope"] == "voice_only") & (fin["feat_variant"].isin(_poem_feats))
    ].copy()
    _vo["wauroc_num"] = pd.to_numeric(_vo["weighted_auroc_mean"], errors="coerce")

    if _vo.empty:
        _out = mo.md("No poem runs found.")
    else:
        _vo["_wba"] = pd.to_numeric(_vo["weighted_balanced_accuracy_mean"], errors="coerce")
        _poem_cols = ["feat_variant", "classifier_name", "splitting", "norm", "wauroc_num", "weighted_auroc_std", "_wba", "auroc_mean"]
        _poem_labels = ["Feature", "Classifier", "Split", "Norm", "wAUROC", "wAUROC std", "wBA", "AUROC"]
        _tbl = _vo[_poem_cols].copy()
        _tbl.columns = _poem_labels
        for _c in [_l for _l in _poem_labels if _l not in ("Feature", "Classifier", "Split", "Norm")]:
            _tbl[_c] = pd.to_numeric(_tbl[_c], errors="coerce").round(3)
        _tbl = _tbl.sort_values("wAUROC", ascending=False).reset_index(drop=True)
        _out = mo.ui.table(_tbl)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Split (Chunked) vs Unsplit Comparison

    Chunked poem recordings provide more training samples but may introduce variability.
    Only wav2vec_poem has both split and unsplit runs.
    """)
    return


@app.cell
def _(fin, mo, pd):
    _voice = fin[fin["scope"].isin(["voice_only", "voice_plus_age_sex_comorbidity"])].copy()
    _voice["wauroc_num"] = pd.to_numeric(_voice["weighted_auroc_mean"], errors="coerce")

    _rows = []
    for _scope in _voice["scope"].unique():
        for _split_val in ["split", "unsplit"]:
            _sub = _voice[(_voice["feat_variant"].str.contains("poem")) & (_voice["scope"] == _scope) & (_voice["splitting"] == _split_val)]
            if not _sub.empty:
                _rows.append({
                    "Scope": _scope,
                    "Splitting": _split_val,
                    "n_runs": len(_sub),
                    "wAUROC mean": round(_sub["wauroc_num"].mean(), 4),
                    "wAUROC std": round(_sub["wauroc_num"].std(), 4),
                    "wAUROC best": round(_sub["wauroc_num"].max(), 4),
                })
    _tbl = pd.DataFrame(_rows)
    if _tbl.empty:
        _out = mo.md("No data.")
    else:
        _out = mo.vstack([
            mo.ui.table(_tbl),
            mo.md("""
            **Note:** Split runs are only available for wav2vec_poem. The large difference reflects
            both more training samples (chunks) and that wav2vec_poem is the strongest feature variant.
            """),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Classifier Comparison
    """)
    return


@app.cell
def _(alt, fin, mo, pd):
    _voice = fin[fin["scope"].isin(["voice_only", "voice_plus_age_sex_comorbidity"])].copy()
    _voice["wauroc_num"] = pd.to_numeric(_voice["weighted_auroc_mean"], errors="coerce")
    _voice = _voice.dropna(subset=["wauroc_num"])

    if _voice.empty:
        _out = mo.md("No data.")
    else:
        _chart = alt.Chart(_voice).mark_boxplot(extent="min-max").encode(
            x=alt.X("wauroc_num:Q", title="wAUROC (strata-weighted)", scale=alt.Scale(domain=[0.3, 0.9])),
            y=alt.Y("classifier_name:N", title="Classifier", sort="-x"),
            color="classifier_name:N",
        ).properties(width=500, height=200)

        _rows = []
        for _clf in _voice["classifier_name"].unique():
            _sub = _voice[_voice["classifier_name"] == _clf]
            _rows.append({
                "Classifier": _clf,
                "n_runs": len(_sub),
                "wAUROC mean": round(_sub["wauroc_num"].mean(), 4),
                "wAUROC std": round(_sub["wauroc_num"].std(), 4),
                "wAUROC best": round(_sub["wauroc_num"].max(), 4),
            })
        _tbl = pd.DataFrame(_rows).sort_values("wAUROC mean", ascending=False).reset_index(drop=True)

        _out = mo.vstack([_chart, mo.ui.table(_tbl)])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Amplitude Normalization: norm_on vs norm_off
    """)
    return


@app.cell
def _(fin, mo, pd):
    _voice = fin[fin["scope"].isin(["voice_only", "voice_plus_age_sex_comorbidity"])].copy()
    _voice["wauroc_num"] = pd.to_numeric(_voice["weighted_auroc_mean"], errors="coerce")

    _rows = []
    for _n in ["norm_on", "norm_off"]:
        _sub = _voice[_voice["norm"] == _n]
        if not _sub.empty:
            _rows.append({
                "Normalization": _n,
                "n_runs": len(_sub),
                "wAUROC mean": round(_sub["wauroc_num"].mean(), 4),
                "wAUROC std": round(_sub["wauroc_num"].std(), 4),
            })
    _tbl = pd.DataFrame(_rows)

    # Paired delta
    _voice["key"] = (
        _voice["scope"] + "__" + _voice["feat_variant"] + "__"
        + _voice["classifier_name"] + "__" + _voice["splitting"]
    )
    _piv = _voice.pivot_table(index="key", columns="norm", values="wauroc_num", aggfunc="first").dropna()
    if "norm_on" in _piv.columns and "norm_off" in _piv.columns:
        _piv["delta"] = _piv["norm_on"] - _piv["norm_off"]
        _mean_d = _piv["delta"].mean()
        _note = f"Paired wAUROC delta (norm_on - norm_off): **{_mean_d:+.4f}** over {len(_piv)} pairs"
    else:
        _note = ""

    mo.vstack([
        mo.ui.table(_tbl) if not _tbl.empty else mo.md("No data."),
        mo.md(_note) if _note else mo.md(""),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Top 10 Runs: All Metrics

    Primary: wAUROC (strata-weighted) | Secondary: wBA, AUROC, BA, MCC |
    Deployment: PPV/NPV at pi_deploy=0.2, expected FP per 1000 |
    Shortcut indicator: score-age correlation among controls
    """)
    return


@app.cell
def _(fin, mo, pd):
    _voice = fin[fin["scope"].isin(["voice_only", "voice_plus_age_sex_comorbidity"])].copy()
    _voice["wauroc_num"] = pd.to_numeric(_voice["weighted_auroc_mean"], errors="coerce")
    _top = _voice.nlargest(10, "wauroc_num")

    _cols_map = {
        "run_name": "Run",
        "weighted_auroc_mean": "wAUROC",
        "weighted_auroc_std": "wAUROC std",
        "weighted_balanced_accuracy_mean": "wBA",
        "auroc_mean": "AUROC",
        "balanced_accuracy_mean": "BA",
        "mcc_mean": "MCC",
        "ppv_deploy_mean": "PPV",
        "npv_deploy_mean": "NPV",
        "expected_fp_per_1000_mean": "FP/1000",
        "score_age_correlation_controls_mean": "Score-Age r",
        "n_features": "n_feat",
    }
    _avail = {_k: _v for _k, _v in _cols_map.items() if _k in _top.columns}
    _tbl = _top[list(_avail.keys())].copy()
    _tbl.columns = list(_avail.values())

    # Shorten run name for display
    _tbl["Run"] = _tbl["Run"].str.replace("__copd_classification_charite_psm", "", regex=False)

    _num_cols = [_c for _c in _tbl.columns if _c != "Run"]
    for _c in _num_cols:
        _tbl[_c] = pd.to_numeric(_tbl[_c], errors="coerce").round(3)

    mo.ui.table(_tbl.reset_index(drop=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Stratum-Specific AUROCs (Top 5 Runs)

    AUROC broken down by age-band x sex strata.
    **Caution:** Perfect AUROCs (1.000) in small strata are likely artifacts of very small n.
    """)
    return


@app.cell
def _(fin, mo, pd):
    _voice = fin[fin["scope"].isin(["voice_only", "voice_plus_age_sex_comorbidity"])].copy()
    _voice["wauroc_num"] = pd.to_numeric(_voice["weighted_auroc_mean"], errors="coerce")
    _top5 = _voice.nlargest(5, "wauroc_num")

    _strat_cols = [_c for _c in fin.columns if "stratum" in _c and _c.endswith("_mean")]

    _rows = []
    for _, _row in _top5.iterrows():
        _entry = {"Run": str(_row["run_name"]).replace("__copd_classification_charite_psm", "")}
        for _sc in _strat_cols:
            _label = _sc.replace("auroc_stratum_", "").replace("_mean", "")
            _entry[_label] = round(float(_row[_sc]), 3) if pd.notna(_row[_sc]) else None
        _rows.append(_entry)

    _tbl = pd.DataFrame(_rows)
    if _tbl.empty:
        _out = mo.md("No stratum data available.")
    else:
        _out = mo.ui.table(_tbl)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Vowel Comparison (Parselmouth eGeMAPS, voice_only)

    Which sustained vowel carries the most discriminative information?
    """)
    return


@app.cell
def _(alt, fin, mo, pd):
    _vowel_feats = ["parsel_a_egemaps", "parsel_i_egemaps", "parsel_o_egemaps", "parsel_all_vowels_egemaps"]
    _vo = fin[
        (fin["scope"] == "voice_only") & (fin["feat_variant"].isin(_vowel_feats))
    ].copy()
    _vo["wauroc_num"] = pd.to_numeric(_vo["weighted_auroc_mean"], errors="coerce")

    if _vo.empty:
        _out = mo.md("No vowel eGeMAPS runs.")
    else:
        _chart = alt.Chart(_vo).mark_boxplot(extent="min-max").encode(
            x=alt.X("wauroc_num:Q", title="wAUROC (strata-weighted)", scale=alt.Scale(domain=[0.3, 0.8])),
            y=alt.Y("feat_variant:N", title="Vowel", sort="-x"),
        ).properties(width=450, height=180)

        _best = _vo.loc[_vo.groupby("feat_variant")["wauroc_num"].idxmax()]
        _tbl = _best[["feat_variant", "classifier_name", "wauroc_num", "weighted_auroc_std", "weighted_balanced_accuracy_mean"]].copy()
        _tbl.columns = ["Vowel", "Classifier", "wAUROC", "wAUROC std", "wBA"]
        for _c in ["wAUROC", "wAUROC std", "wBA"]:
            _tbl[_c] = pd.to_numeric(_tbl[_c], errors="coerce").round(3)
        _tbl = _tbl.sort_values("wAUROC", ascending=False).reset_index(drop=True)

        _out = mo.vstack([_chart, mo.ui.table(_tbl)])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Sample Weight Strategy: class_only vs ps_overlap
    """)
    return


@app.cell
def _(fin, mo, pd):
    _voice = fin[fin["scope"].isin(["voice_only", "voice_plus_age_sex_comorbidity"])].copy()
    _voice["wauroc_num"] = pd.to_numeric(_voice["weighted_auroc_mean"], errors="coerce")

    _rows = []
    for _strat in ["class_only", "ps_overlap"]:
        _sub = _voice[_voice["sample_weight_strategy"] == _strat]
        if not _sub.empty:
            _rows.append({
                "Strategy": _strat,
                "n_runs": len(_sub),
                "wAUROC mean": round(_sub["wauroc_num"].mean(), 4),
                "wAUROC std": round(_sub["wauroc_num"].std(), 4),
                "wAUROC best": round(_sub["wauroc_num"].max(), 4),
            })
    _tbl = pd.DataFrame(_rows)
    if _tbl.empty:
        _out = mo.md("No data.")
    else:
        _out = mo.ui.table(_tbl)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Linear Mixed Model: Factorial Analysis

    LMM on fold-level **wAUROC** (strata-weighted AUROC, 50 folds per run = 10 repeats x 5 folds)
    to estimate the contribution of each experimental factor while accounting for within-run correlation.
    Strata weights reweight each test fold to match the DMP target population (age × sex).

    **Fixed effects:** recording_type, feature_family, classifier, norm, scope

    **Random effect:** `(1 | run_id)` — random intercept per run

    **recording_type levels:** a, i, o, all_vowels, poem, poem_chunked (split poems as own category)
    """)
    return


@app.cell
def _(fin, mo, pd):
    # Per-fold AUROC values are persisted in the canonical parquet as list
    # columns (`fold_aurocs`, `fold_weighted_aurocs`) by build_canonical_parquets.py.
    # We explode them into long format here so downstream LMM / boxplot code
    # sees the same (run_id, fold_step, auroc, weighted_auroc) shape it always
    # expected. No MLflow round-trip required.
    _wide = fin[["run_id", "fold_aurocs", "fold_weighted_aurocs"]].copy()
    _wide = _wide[_wide["fold_aurocs"].apply(lambda x: x is not None and len(x) > 0)]
    fold_df = _wide.explode(["fold_aurocs", "fold_weighted_aurocs"]).reset_index(drop=True)
    fold_df["fold_step"] = fold_df.groupby("run_id").cumcount()
    fold_df = fold_df.rename(
        columns={"fold_aurocs": "auroc", "fold_weighted_aurocs": "weighted_auroc"}
    )
    fold_df["auroc"] = pd.to_numeric(fold_df["auroc"], errors="coerce")
    fold_df["weighted_auroc"] = pd.to_numeric(fold_df["weighted_auroc"], errors="coerce")
    fold_df = fold_df[["run_id", "fold_step", "auroc", "weighted_auroc"]]

    _status_msg = (
        f"Loaded {len(fold_df)} fold observations from canonical parquet "
        f"({fold_df['run_id'].nunique()} runs)."
    )
    mo.md(f"**Fold data:** {_status_msg}")
    return (fold_df,)


@app.cell
def _(fin, fold_df, mo, pd):
    # Merge fold-level data with run metadata
    _meta_cols = ["run_id", "run_name", "feat_variant", "scope", "norm",
                  "splitting", "classifier_name", "demographics_only"]
    _available = [_c for _c in _meta_cols if _c in fin.columns]
    _run_meta = fin[_available].copy()

    _lmm_raw = fold_df.merge(_run_meta, on="run_id", how="left")

    # Parse feat_variant into recording_type and feature_family
    def _parse_fv(_f, _s):
        _f = str(_f)
        _s = str(_s)
        if _f.startswith("wav2vec_"):
            _rec = _f.replace("wav2vec_", "")
            if _rec == "poem" and _s == "split":
                _rec = "poem_chunked"
            return _rec, "wav2vec2"
        if _f.startswith("parsel_"):
            _rest = _f[len("parsel_"):]
            for _fam in ["all_features", "basic_acoustics", "egemaps", "spectral", "temporal"]:
                if _rest.endswith(_fam):
                    _rec = _rest[:-(len(_fam) + 1)]
                    if _rec == "poem" and _s == "split":
                        _rec = "poem_chunked"
                    return _rec, _fam
        return None, None

    _parsed = _lmm_raw.apply(
        lambda _r: pd.Series(
            _parse_fv(_r["feat_variant"], _r.get("splitting", "")),
            index=["recording_type", "feature_family"],
        ),
        axis=1,
    )
    _lmm_raw = pd.concat([_lmm_raw, _parsed], axis=1)

    # Filter: voice scopes only, valid parsing, not demographics-only
    lmm_df = _lmm_raw[
        (_lmm_raw["scope"].isin(["voice_only", "voice_plus_age_sex_comorbidity"]))
        & (_lmm_raw["recording_type"].notna())
        & (_lmm_raw["feature_family"].notna())
    ].copy()

    # Remove NaN weighted_auroc values (primary metric for LMM)
    lmm_df = lmm_df.dropna(subset=["weighted_auroc"])

    # Convert to categorical for statsmodels
    for _col in ["recording_type", "feature_family", "classifier_name", "norm", "scope"]:
        lmm_df[_col] = lmm_df[_col].astype("category")

    _n_runs = lmm_df["run_id"].nunique()
    _n_obs = len(lmm_df)
    _rec_levels = sorted(lmm_df["recording_type"].unique())
    _feat_levels = sorted(lmm_df["feature_family"].unique())

    mo.md(f"""
    **LMM dataset:** {_n_obs:,} fold-level observations from {_n_runs} runs

    **Primary metric:** strata-weighted AUROC (wAUROC) — reweighted to DMP age×sex target population

    | Factor | Levels |
    |--------|--------|
    | recording_type | {', '.join(_rec_levels)} |
    | feature_family | {', '.join(_feat_levels)} |
    | classifier | {', '.join(sorted(lmm_df['classifier_name'].unique()))} |
    | norm | {', '.join(sorted(lmm_df['norm'].unique()))} |
    | scope | {', '.join(sorted(lmm_df['scope'].unique()))} |
    """)
    return (lmm_df,)


@app.cell
def _(lmm_df, mo, pd):
    from statsmodels.formula.api import mixedlm as _mixedlm

    _formula = (
        "weighted_auroc ~ C(recording_type, Treatment(reference='a'))"
        " + C(feature_family, Treatment(reference='egemaps'))"
        " + C(classifier_name, Treatment(reference='LogisticRegression'))"
        " + C(norm, Treatment(reference='norm_off'))"
        " + C(scope, Treatment(reference='voice_only'))"
    )

    _model = _mixedlm(_formula, lmm_df, groups=lmm_df["run_id"])
    lmm_result = _model.fit(method="lbfgs", reml=True)

    # Build coefficient table
    _coef_df = pd.DataFrame({
        "Term": lmm_result.params.index,
        "Estimate": lmm_result.params.values,
        "SE": lmm_result.bse.values,
        "z": lmm_result.tvalues.values,
        "p": lmm_result.pvalues.values,
        "CI_lower": lmm_result.conf_int().iloc[:, 0].values,
        "CI_upper": lmm_result.conf_int().iloc[:, 1].values,
    })

    # Clean up term names
    _coef_df["Term"] = (
        _coef_df["Term"]
        .str.replace(r"C\(recording_type, Treatment\(reference='a'\)\)\[T\.", "rec: ", regex=True)
        .str.replace(r"C\(feature_family, Treatment\(reference='egemaps'\)\)\[T\.", "feat: ", regex=True)
        .str.replace(r"C\(classifier_name, Treatment\(reference='LogisticRegression'\)\)\[T\.", "clf: ", regex=True)
        .str.replace(r"C\(norm, Treatment\(reference='norm_off'\)\)\[T\.", "norm: ", regex=True)
        .str.replace(r"C\(scope, Treatment\(reference='voice_only'\)\)\[T\.", "scope: ", regex=True)
        .str.rstrip("]")
    )

    _coef_df["Sig"] = _coef_df["p"].apply(
        lambda _p: "***" if _p < 0.001 else "**" if _p < 0.01 else "*" if _p < 0.05 else ""
    )

    for _c in ["Estimate", "SE", "z", "CI_lower", "CI_upper"]:
        _coef_df[_c] = _coef_df[_c].round(4)
    _coef_df["p"] = _coef_df["p"].apply(lambda _v: f"{_v:.1e}" if _v < 0.001 else f"{_v:.4f}")

    # Random effects info
    _re_var = lmm_result.cov_re.iloc[0, 0]
    _resid_var = lmm_result.scale
    _icc = _re_var / (_re_var + _resid_var)

    lmm_coef_df = _coef_df

    mo.vstack([
        mo.md(f"""
        **Model:** `wAUROC ~ recording_type + feature_family + classifier + norm + scope + (1 | run_id)`

        | Statistic | Value |
        |-----------|-------|
        | Random intercept variance (run) | {_re_var:.4f} |
        | Residual variance | {_resid_var:.4f} |
        | ICC | {_icc:.3f} |
        | N observations | {int(lmm_result.nobs):,} |
        | N groups (runs) | {lmm_df['run_id'].nunique()} |
        | Log-likelihood | {lmm_result.llf:.1f} |
        | AIC | {lmm_result.aic:.1f} |

        **Reference levels:** recording_type=a, feature_family=eGeMAPS, classifier=LogisticRegression, norm=norm_off, scope=voice_only
        """),
        mo.md("### Fixed Effects Coefficients"),
        mo.ui.table(lmm_coef_df),
    ])
    return (lmm_coef_df,)


@app.cell
def _(alt, lmm_coef_df, mo, pd):
    # Exclude intercept and Group Var for plotting
    _plot_df = lmm_coef_df[
        ~lmm_coef_df["Term"].isin(["Intercept", "Group Var"])
    ].copy()

    # Need numeric columns for Altair
    for _c in ["Estimate", "CI_lower", "CI_upper"]:
        _plot_df[_c] = pd.to_numeric(_plot_df[_c])

    # Color by factor type
    _plot_df["Factor"] = _plot_df["Term"].apply(
        lambda _t: (
            "Recording type" if _t.startswith("rec:")
            else "Feature family" if _t.startswith("feat:")
            else "Classifier" if _t.startswith("clf:")
            else "Normalization" if _t.startswith("norm:")
            else "Scope" if _t.startswith("scope:")
            else "Other"
        )
    )

    _base = alt.Chart(_plot_df).encode(
        y=alt.Y(
            "Term:N",
            sort=alt.EncodingSortField(field="Estimate", order="descending"),
            title="",
        ),
        color=alt.Color("Factor:N"),
    )

    _points = _base.mark_point(size=60, filled=True).encode(
        x=alt.X("Estimate:Q", title="Coefficient (wAUROC change vs reference)"),
    )

    _errorbars = _base.mark_rule(strokeWidth=2).encode(
        x="CI_lower:Q",
        x2="CI_upper:Q",
    )

    _zero = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color="gray", strokeDash=[4, 4])
        .encode(x="x:Q")
    )

    _chart = (_errorbars + _points + _zero).properties(
        width=550,
        height=max(250, len(_plot_df) * 25),
        title="LMM Fixed Effects on wAUROC: Coefficient Estimates with 95% CI",
    )

    mo.vstack([
        _chart,
        mo.md("""
        **Interpretation:** Each coefficient shows the estimated change in fold-level wAUROC
        (strata-weighted to DMP target population) relative to the reference category.
        Reference: vowel /a/, eGeMAPS, LogisticRegression, norm_off, voice_only.
        Error bars = 95% CI; factors whose CI excludes zero are significant.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(EXPERIMENT_NAME, mo):
    mo.md(f"""
    ---
    *Notebook: `notebooks/lung_disease_runs_analysis_psm.py` — Experiment: `{EXPERIMENT_NAME}`*
    """)
    return


if __name__ == "__main__":
    app.run()
