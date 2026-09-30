"""Does training-time covariate weighting explain the Charité comparators below chance?

Reads the published single-seed runs (mlflow_cluster mirror) and the reruns of
scripts/comparator_weighting_sensitivity.py (Charité complete and PSM, COVID-19 Sounds; each
rerun is paired with that cohort's published voice run) and writes, under
outputs/paper_checks/comparator_weighting/:

  published_runs.csv   per published run: weighting, grid, seed, number of
                       folds with a constant prediction, fold-level wAUROC and
                       AUROC (mean, SD over the 50 folds), and person-pooled
                       wAUROC / AUROC on the same out-of-fold predictions
  reruns_by_seed.csv   per rerun (task x scope x weighting x grid x seed): person
                       bootstrap wAUROC with 95% CI, paired difference to the
                       cohort's voice run with 95% CI, unweighted AUROC, fold stats
  reruns_summary.csv   the same, summarised over seeds per tuple
  uk_eval_weights.json COVID-19 Sounds evaluation weights: ESS per class and
                       share of total weight on participants over 65

and outputs/_tmp_boot/fig2_comparators.json, which Figure 2 reads: the
seed-42 flat comparators (extended grid) for Charité complete and COVID-19
Sounds with their paired differences to voice, and raw age as the Charité PSM
comparator. Raw age is scored on the 55 voice_psm participants with that run's
strata weights, as paper_bootstrap_uk.py does for COVID-19 Sounds.

Person-level scores follow scripts/paper_bootstrap_charite.py: each person's
out-of-fold probability and strata weight are averaged over the 10 repeats;
bootstrap resamples persons within class (B = 5000). The bootstrap uses the
closed form of the weighted Mann-Whitney statistic, which equals
sklearn's roc_auc_score with sample_weight (ties count one half).

Run from the repository root:
    PYTHONPATH=. uv run python scripts/paper_comparator_weighting_analysis.py
"""
import glob
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

B = 5000
OUT = Path("outputs/paper_checks/comparator_weighting")
MF = "data/2025-08-13/mlflow_cluster"
CHA, PSM, UK = (f"{MF}/457012274041432773", f"{MF}/644578817518142221",
                f"{MF}/271009020022062271")
RERUNS = "outputs/comparator_weighting/mlruns"
UKMETA = "data/uk_matched_metadata.csv"
FIG2_JSON = Path("outputs/_tmp_boot/fig2_comparators.json")
REF_YEAR = 2025      # Charité age = REF_YEAR - birth_year, as everywhere else

PUBLISHED = {
    "charite_complete/voice": [f"{CHA}/00a61b66628b45e8906c95097abd7348"],
    "charite_complete/age_only": ["outputs/_tmp_age_only_mlruns/821226550920393827/a0b5bd9da0384ecdb66d70c5c0206cac"],
    "charite_complete/age_sex": [f"{CHA}/b9dc00497d77462e8a76f9515b348af2"],
    "charite_complete/age_sex_comorb": [f"{CHA}/56f48804a8674b4ab6a5ed2098886c19"],
    "charite_psm/voice": [f"{PSM}/884ea1802c0d44278090ae4860de5288"],
    "charite_psm/age_only": [f"{PSM}/8f8cfddc0ae14767b5a72d18e7e5f4db"],
    "uk/voice": [f"{UK}/de720542f5974ee8998d966ca75a536f"],
    "uk/age_sex": [f"{UK}/629aed426f2f4a4b9df891fc8b08b402"],
    # paper_bootstrap_uk.py pools the two identical-config runs
    "uk/age_sex_comorb": [f"{UK}/2569dbc2b5444f50b6187910b263c8ff",
                          f"{UK}/4906d6f489c24893a804cb14cdd7be68"],
}


def read(path):
    try:
        return Path(path).read_text().strip()
    except FileNotFoundError:
        return None


def fold_predictions(run_dir):
    files = glob.glob(f"{run_dir}/artifacts/fold_predictions/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.assign(run_id=Path(run_dir).name)


def persons(df):
    g = df.groupby("audio_id")
    return pd.DataFrame({"y": g["true"].first().astype(int),
                         "p": g["proba_copd"].mean(),
                         "w": g["strata_weight"].mean()})


def fold_values(df):
    """Per (run, fold): wAUROC, AUROC, whether the prediction is constant."""
    rows = {}
    for key, g in df.groupby(["run_id", "fold"]):
        rows[key] = (roc_auc_score(g["true"], g["proba_copd"], sample_weight=g["strata_weight"]),
                     roc_auc_score(g["true"], g["proba_copd"]),
                     g["proba_copd"].nunique() == 1)
    return pd.DataFrame(rows, index=["wauroc", "auroc", "constant"]).T


def fold_stats(df):
    f = fold_values(df)
    return {"n_folds": len(f),
            "n_constant_folds": int(f["constant"].sum()),
            "fold_wauroc_mean": f["wauroc"].mean(), "fold_wauroc_sd": f["wauroc"].std(ddof=1),
            "fold_auroc_mean": f["auroc"].mean(), "fold_auroc_sd": f["auroc"].std(ddof=1)}


def fold_diff_sd(voice_df, cmp_df):
    """SD over folds of wAUROC(voice) - wAUROC(comparator), if both runs used
    the same outer folds (same seed and persons); NaN otherwise."""
    members = lambda d: d.groupby("fold")["audio_id"].apply(frozenset)
    cmp_one = cmp_df[cmp_df["run_id"] == cmp_df["run_id"].iloc[0]]
    if not members(voice_df).equals(members(cmp_one)):
        return np.nan
    v = fold_values(voice_df)["wauroc"].droplevel(0)
    c = fold_values(cmp_one)["wauroc"].droplevel(0)
    return float((v - c).std(ddof=1))


def pair_matrix(p_pos, p_neg):
    return (p_pos[:, None] > p_neg[None, :]) + 0.5 * (p_pos[:, None] == p_neg[None, :])


def boot(person, rng, paired=None):
    """Class-stratified person bootstrap of wAUROC (and of a paired difference)."""
    y = person["y"].to_numpy()
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    cp = rng.multinomial(len(pos), np.full(len(pos), 1 / len(pos)), size=B)
    cn = rng.multinomial(len(neg), np.full(len(neg), 1 / len(neg)), size=B)

    def wauc(df):
        p, w = df["p"].to_numpy(), df["w"].to_numpy()
        C = pair_matrix(p[pos], p[neg])
        a, b = cp * w[pos], cn * w[neg]
        return (((a @ C) * b).sum(1) / (a.sum(1) * b.sum(1)),
                roc_auc_score(y, p, sample_weight=w))

    dist, point = wauc(person)
    r = {"wauroc": point, "wauroc_lo": np.quantile(dist, .025),
         "wauroc_hi": np.quantile(dist, .975),
         "auroc": roc_auc_score(y, person["p"])}
    if paired is not None:
        d2, p2 = wauc(paired)
        d = dist - d2
        r.update(diff=point - p2, diff_lo=np.quantile(d, .025),
                 diff_hi=np.quantile(d, .975), frac_diff_gt0=(d > 0).mean())
    return r


def run_info(run_dir):
    prm = lambda k: read(f"{run_dir}/params/{k}")
    grid = "lr_baseline" if prm("classifier_hyperparams.clf__solver") == "['lbfgs']" else "extended"
    log = glob.glob(f"{run_dir}/artifacts/logs/*.log")
    covs = None
    if log:
        m = re.search(r"covariates used: (\[.*?\])", Path(log[0]).read_text())
        covs = m.group(1) if m else None
    return {"classifier": prm("classifier_name"), "weighting": prm("sample_weight_strategy"),
            "grid": grid if prm("classifier_name") == "LogisticRegression" else "extended",
            "C_grid": prm("classifier_hyperparams.clf__C"),
            "penalty_grid": prm("classifier_hyperparams.clf__penalty"),
            "seed": prm("random_seed"), "demographic_columns": prm("demographic_feature_columns"),
            "include_voice": prm("include_voice_features"),
            "n_features": (read(f"{run_dir}/metrics/n_features") or " nan").split()[1],
            "run_name": read(f"{run_dir}/tags/mlflow.runName"),
            "overlap_covariates_logged": covs}


def published():
    rows, P, D = [], {}, {}
    for key, dirs in PUBLISHED.items():
        df = pd.concat([fold_predictions(d) for d in dirs], ignore_index=True)
        D[key] = df
        P[key] = persons(df)
        per = P[key]
        voice = D[key.split("/")[0] + "/voice"]
        rows.append({"run": key, "run_ids": ";".join(Path(d).name for d in dirs),
                     **run_info(dirs[0]), **fold_stats(df),
                     "fold_diff_sd_voice_minus": fold_diff_sd(voice, df) if df is not voice else np.nan,
                     "n_pos": int(per.y.sum()), "n_neg": int((per.y == 0).sum()),
                     "pooled_wauroc": roc_auc_score(per.y, per.p, sample_weight=per.w),
                     "pooled_auroc": roc_auc_score(per.y, per.p)})
    # UK age-only is raw age on the voice_uk person set (paper_bootstrap_uk.py);
    # fold stats use voice_uk's fold assignment and test-fold strata weights.
    vdf = fold_predictions(PUBLISHED["uk/voice"][0])
    age = pd.to_numeric(pd.read_csv(UKMETA).drop_duplicates("audio_id")
                        .set_index("audio_id")["age"], errors="coerce")
    adf = vdf.assign(proba_copd=vdf["audio_id"].map(age).astype(float)).dropna(subset=["proba_copd"])
    per = persons(adf)
    P["uk/age_only"] = per
    rows.append({"run": "uk/age_only", "run_ids": "raw age on voice_uk persons",
                 **fold_stats(adf), "fold_diff_sd_voice_minus": fold_diff_sd(vdf, adf),
                 "n_pos": int(per.y.sum()), "n_neg": int((per.y == 0).sum()),
                 "pooled_wauroc": roc_auc_score(per.y, per.p, sample_weight=per.w),
                 "pooled_auroc": roc_auc_score(per.y, per.p)})
    # Charité PSM age comparator: raw age on the voice_psm participants, with
    # that run's fold assignment and test-fold strata weights.
    pdf = D["charite_psm/voice"]
    birth = pd.read_csv("data/2025-08-13/metadata.csv").groupby("audio_id")["birth_year"].first()
    rdf = pdf.assign(proba_copd=(REF_YEAR - pdf["audio_id"].map(birth)).astype(float))
    per = persons(rdf)
    P["charite_psm/age_raw"] = per
    rows.append({"run": "charite_psm/age_raw", "run_ids": "raw age on voice_psm persons",
                 **fold_stats(rdf), "fold_diff_sd_voice_minus": fold_diff_sd(pdf, rdf),
                 "n_pos": int(per.y.sum()), "n_neg": int((per.y == 0).sum()),
                 "pooled_wauroc": roc_auc_score(per.y, per.p, sample_weight=per.w),
                 "pooled_auroc": roc_auc_score(per.y, per.p)})
    return pd.DataFrame(rows), P, D, vdf, age


def reruns(voices):
    """voices: {task: (voice persons, voice fold predictions)} for pairing."""
    rows = []
    for d in sorted(glob.glob(f"{RERUNS}/*/*/")):
        name = read(f"{d}tags/mlflow.runName")
        if not name or not glob.glob(f"{d}artifacts/fold_predictions/*.parquet"):
            continue
        scope, grid, task, weighting_seed = name.split("__", 3)
        weighting, seed = weighting_seed.rsplit("__seed", 1)
        voice, voice_df = voices[task]
        df = fold_predictions(d)
        per = persons(df)
        common = voice.index.intersection(per.index)
        assert len(common) == len(voice) == len(per)
        v, c = voice.loc[common], per.loc[common]
        assert (v.y.to_numpy() == c.y.to_numpy()).all()
        fdsd = fold_diff_sd(voice_df, df)
        rng = np.random.default_rng([int(seed), zlib.crc32(name.encode())])
        r = boot(v, rng, paired=c)  # paired: voice - comparator, same resamples
        cmp_only = boot(c, np.random.default_rng(int(seed)))
        rows.append({"task": task, "scope": scope, "grid": grid, "weighting": weighting,
                     "seed": int(seed),
                     "wauroc": cmp_only["wauroc"], "wauroc_lo": cmp_only["wauroc_lo"],
                     "wauroc_hi": cmp_only["wauroc_hi"], "auroc": cmp_only["auroc"],
                     "diff_voice_minus": r["diff"], "diff_lo": r["diff_lo"], "diff_hi": r["diff_hi"],
                     "frac_diff_gt0": r["frac_diff_gt0"], **fold_stats(df),
                     "fold_diff_sd_voice_minus": fdsd})
    return pd.DataFrame(rows)


def uk_eval_weights(vdf, age):
    """ESS per class and weight share on age > 65, person level and per fold."""
    def summarise(y, w, a):
        out = {}
        for lab, cls in ((1, "copd"), (0, "control")):
            m = y == lab
            out[f"ess_{cls}"] = float(w[m].sum() ** 2 / (w[m] ** 2).sum())
            out[f"n_{cls}"] = int(m.sum())
            out[f"share_weight_over65_{cls}"] = float(w[m & (a > 65)].sum() / w[m].sum())
            out[f"share_persons_over65_{cls}"] = float((a[m] > 65).mean())
        out["share_weight_over65_all"] = float(w[a > 65].sum() / w.sum())
        out["share_persons_over65_all"] = float((a > 65).mean())
        return out

    per = persons(vdf)
    a = per.index.map(age).to_numpy(dtype=float)
    res = {"person_level_mean_over_repeats": summarise(per.y.to_numpy(), per.w.to_numpy(), a)}
    folds = []
    for _, g in vdf.groupby("fold"):
        folds.append(summarise(g["true"].to_numpy(), g["strata_weight"].to_numpy(),
                               g["audio_id"].map(age).to_numpy(dtype=float)))
    f = pd.DataFrame(folds)
    res["per_test_fold_median"] = f.median().to_dict()
    res["per_test_fold_min"] = f.min().to_dict()
    res["per_test_fold_max"] = f.max().to_dict()
    return res


def fig2_estimates(P, rr):
    """Figure 2's new comparator rows, unrounded (the figure rounds once)."""
    est = lambda r: {"point": r["wauroc"], "ci": [r["wauroc_lo"], r["wauroc_hi"]],
                     "auroc": r["auroc"]}
    pair = lambda r: {"diff_point": r["diff"], "diff_ci": [r["diff_lo"], r["diff_hi"]],
                      "frac_diff_gt0": r["frac_diff_gt0"]}
    out = {}
    flat = rr[(rr.seed == 42) & (rr.weighting == "flat") & (rr.grid == "extended")]
    names = {"age_only": "age_only", "age_sex_only": "age_sex",
             "age_sex_comorbidity_individual": "age_sexcomorb"}
    for task, coh in [("copd_classification_charite_complete", "complete"),
                      ("copd_classification_uk_only", "uk")]:
        for _, r in flat[flat.task == task].iterrows():
            key = f"{names[r.scope]}_{coh}_flat"
            out[key] = {**est(r), "source": "reruns_by_seed.csv"}
            out[f"paired_{coh}_voice_minus_{names[r.scope]}_flat"] = {
                "diff_point": r["diff_voice_minus"], "diff_ci": [r["diff_lo"], r["diff_hi"]],
                "frac_diff_gt0": r["frac_diff_gt0"], "source": "reruns_by_seed.csv"}

    # PSM raw age, same bootstrap and RNG convention as the reruns (seed 42)
    v, a = P["charite_psm/voice"], P["charite_psm/age_raw"]
    assert v.index.equals(a.index) and (v.y == a.y).all()
    only = boot(a, np.random.default_rng(42))
    paired = boot(v, np.random.default_rng([42, zlib.crc32(b"charite_psm/age_raw")]), paired=a)
    out["age_raw_psm"] = {**est(only), "n_pos": int(a.y.sum()), "n_neg": int((a.y == 0).sum()),
                          "source": "raw age, voice_psm persons and strata weights"}
    out["paired_psm_voice_minus_age_raw"] = {**pair(paired), "voice_point": paired["wauroc"]}
    FIG2_JSON.write_text(json.dumps(out, indent=2))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pub, P, D, vdf, age = published()
    pub.to_csv(OUT / "published_runs.csv", index=False)
    print(pub[["run", "weighting", "grid", "seed", "n_features", "n_constant_folds",
               "fold_wauroc_mean", "fold_wauroc_sd", "fold_diff_sd_voice_minus",
               "pooled_wauroc", "pooled_auroc"]]
          .round(3).to_string(index=False))

    uk = uk_eval_weights(vdf, age)
    (OUT / "uk_eval_weights.json").write_text(json.dumps(uk, indent=2))
    print(json.dumps(uk["person_level_mean_over_repeats"], indent=2))

    rr = reruns({
        "copd_classification_charite_complete": (P["charite_complete/voice"],
                                                 D["charite_complete/voice"]),
        "copd_classification_charite_psm": (P["charite_psm/voice"], D["charite_psm/voice"]),
        "copd_classification_uk_only": (P["uk/voice"], D["uk/voice"]),
    })
    if rr.empty:
        return
    keys = ["task", "grid", "scope", "weighting"]
    rr = rr.sort_values([*keys, "seed"])
    rr.to_csv(OUT / "reruns_by_seed.csv", index=False)
    s42 = rr[rr.seed == 42].set_index(keys)
    g = rr.groupby(keys)
    summ = pd.DataFrame({
        "n_seeds": g.size(),
        "seed42_wauroc": s42["wauroc"], "seed42_lo": s42["wauroc_lo"], "seed42_hi": s42["wauroc_hi"],
        "seed42_auroc": s42["auroc"],
        "seed42_diff": s42["diff_voice_minus"], "seed42_diff_lo": s42["diff_lo"],
        "seed42_diff_hi": s42["diff_hi"],
        "wauroc_mean": g["wauroc"].mean(), "wauroc_min": g["wauroc"].min(),
        "wauroc_max": g["wauroc"].max(), "n_seeds_below_0.5": g["wauroc"].apply(lambda x: int((x < .5).sum())),
        "auroc_mean": g["auroc"].mean(), "diff_mean": g["diff_voice_minus"].mean(),
        "n_seeds_diff_ci_excl_0": g["diff_lo"].apply(lambda x: int((x > 0).sum())),
        "fold_wauroc_mean": g["fold_wauroc_mean"].mean(), "fold_wauroc_sd": g["fold_wauroc_sd"].mean(),
        "fold_diff_sd_seed42": s42["fold_diff_sd_voice_minus"],
        "constant_folds_mean": g["n_constant_folds"].mean(),
    }).reset_index()
    summ.to_csv(OUT / "reruns_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    f2 = fig2_estimates(P, rr)
    print(f"\nwrote {FIG2_JSON}")
    for k in ("age_raw_psm", "paired_psm_voice_minus_age_raw"):
        print(k, json.dumps(f2[k]))


if __name__ == "__main__":
    main()
