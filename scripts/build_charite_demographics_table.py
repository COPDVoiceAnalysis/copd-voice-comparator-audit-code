"""
Build the Charité demographics table (tab:charite-demographics) reproducibly.

Root-cause fix: this table was previously hand-authored inline in
methodology.tex (N=101/79) with a comorbidity grouping not defined in any
committed script. This script regenerates it from committed data on the
*exact* analysis cohort used by the headline analyses, so the thesis is
internally consistent and the table is reproducible.

Cohort: reconstructed via the same ExperimentConfig + Metadata
filter_for_experiment path the bootstrap/training use (task
copd_classification_charite_complete, complete-case a/i/o/poem) — yields
the same N as every headline number.

Group definitions (explicit, documented; raw columns from
comorbidities_2026_01_28.xlsx joined via voice_audio_id.xlsx, matching
scripts/build_ps_covariates_csv.py; respiratory-non-COPD from the
secondary lung-disease metadata fields):
  Metabolic      = Arterielle Hypertonie OR Diabetes Mellitus II
  Cardiovascular = KHK OR Herzinsuffizienz OR VHF
  Renal          = Niereninsuffizienz
  Depression     = Depression
  Dyslipidemia   = Dyslipidämie (single raw condition; 0% controls)
  Hepatic steatosis = Steatosis hepatis (single raw condition; 0% controls)
  Respiratory (non-COPD) = any of lung_disease_2/3/4 present and != copd

Continuous: mean ± SD, Welch t-test. Categorical: n (%), Fisher exact.

Usage: uv run python scripts/build_charite_demographics_table.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, fisher_exact

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import yaml  # noqa: E402
from src.models.config_models import ExperimentConfig  # noqa: E402
from src.models.metadata import Metadata  # noqa: E402

CFG = ("outputs/_tmp_age_only_configs/"
       "baseline_age_only__copd_classification_charite_complete__"
       "ps_overlap__seed_sweep_config.yml")
META = "data/2025-08-13/metadata.csv"
XLSX = "data/2025-08-13/comorbidities_2026_01_28.xlsx"
VID = "data/2025-08-13/voice_audio_id.xlsx"
OUT = "outputs/thesis_tables/tab_charite_demographics.tex"


def cohort_audio_ids() -> tuple[set[int], dict[int, int]]:
    """Exact N analysis cohort via the same Metadata filter as the headline
    analyses. Returns (audio_ids, {audio_id: y} with y=1 COPD / 0 control)."""
    cfg = ExperimentConfig(**yaml.safe_load(open(CFG)))
    md = Metadata(csv_path=cfg.metadata_file,
                  ps_covariates_file=cfg.ps_covariates_file)
    md.filter_for_experiment(task_config=cfg.task_config,
                             feature_config=cfg.feature_config)
    rids = md.get_recording_identifier_sets(feature_config=cfg.feature_config)
    ids = md.extract_audio_ids(rids)
    y, _ = md.get_class_labels_for_sets(rids)
    ids = [int(i) for i in ids]
    return set(ids), {int(i): int(yy) for i, yy in zip(ids, y)}


def _bin(s: pd.Series) -> pd.Series:
    return s.fillna(0).astype(float).clip(0, 1).astype(int)


def comorbidity_groups(audio_ids: set[int]) -> pd.DataFrame:
    x = pd.read_excel(XLSX)
    x.columns = x.columns.str.strip()
    v = pd.read_excel(VID)
    v.columns = v.columns.str.strip()
    x["sid"] = x["Studien-ID"].astype(str).str.strip().str.lower()
    v["sid"] = v["Studien-ID"].astype(str).str.strip().str.lower()
    g = pd.DataFrame({
        "sid": x["sid"],
        "Metabolic": ((_bin(x.get("Arterielle Hypertonie", 0))
                       + _bin(x.get("Diabetes Mellitus II", 0))) > 0).astype(int),
        "Cardiovascular": ((_bin(x.get("KHK", 0)) + _bin(x.get("Herzinsuffizienz", 0))
                            + _bin(x.get("VHF", 0))) > 0).astype(int),
        "Renal": _bin(x.get("Niereninsuffizienz", 0)),
        "Depression": _bin(x.get("Depression", 0)),
        "Dyslipidemia": _bin(x.get("Dyslipidämie", 0)),
        "HepaticSteatosis": _bin(x.get("Steatosis hepatis", 0)),
    })
    vl = v[["sid", "Audio-ID"]].drop_duplicates("sid")
    g = g.merge(vl, on="sid", how="left").rename(columns={"Audio-ID": "audio_id"})
    g["audio_id"] = pd.to_numeric(g["audio_id"], errors="coerce").astype("Int64")
    g = g.dropna(subset=["audio_id"]).copy()
    g["audio_id"] = g["audio_id"].astype(int)
    return g[g["audio_id"].isin(audio_ids)].set_index("audio_id")


def main() -> None:
    ids, ymap = cohort_audio_ids()
    m = pd.read_csv(META)
    m = m[m["audio_id"].isin(ids)].drop_duplicates("audio_id").set_index("audio_id")
    m["y"] = m.index.map(ymap)  # 1 COPD / 0 control
    m["age"] = pd.to_datetime(m["date"]).dt.year - pd.to_numeric(m["birth_year"], errors="coerce")
    for _nc in ("bmi", "pack_years"):
        m[_nc] = pd.to_numeric(m[_nc], errors="coerce")
    resp_cols = ["lung_disease_2", "lung_disease_3", "lung_disease_4"]
    resp = m[resp_cols].apply(
        lambda r: any(isinstance(z, str) and z.strip() != "" and z.strip().lower() != "copd"
                      for z in r), axis=1).astype(int)
    m["Respiratory_nonCOPD"] = resp
    cg = comorbidity_groups(ids)
    for c in ["Metabolic", "Cardiovascular", "Renal", "Depression",
              "Dyslipidemia", "HepaticSteatosis"]:
        m[c] = m.index.map(cg[c]).fillna(0).astype(int)

    copd = m[m["y"] == 1]
    ctrl = m[m["y"] == 0]
    n_c, n_k = len(copd), len(ctrl)
    print(f"COHORT: N={len(m)}  COPD={n_c}  control={n_k}")

    def cont(col: str):
        a, b = copd[col].dropna(), ctrl[col].dropna()
        p = ttest_ind(a, b, equal_var=False).pvalue if len(a) > 1 and len(b) > 1 else np.nan
        return (a.mean(), a.std(ddof=1), len(a), b.mean(), b.std(ddof=1), len(b), p)

    def cat(col: str):
        ca, cb = int(copd[col].sum()), int(ctrl[col].sum())
        tbl = [[ca, n_c - ca], [cb, n_k - cb]]
        p = fisher_exact(tbl)[1]
        return ca, ca / n_c * 100, cb, cb / n_k * 100, p

    print("\n--- continuous (COPD mean±sd | control mean±sd | Welch p) ---")
    for c in ["age", "bmi", "pack_years"]:
        am, asd, an, bm, bsd, bn, p = cont(c)
        print(f"  {c:11s} {am:6.1f}±{asd:5.1f} (n={an})  |  {bm:6.1f}±{bsd:5.1f} (n={bn})  |  p={p:.3g}")

    print("\n--- sex (male=?) raw values:", sorted(m['sex'].dropna().unique())[:6])
    male = m['sex'].astype(str).str.lower().isin(["1", "m", "male", "männlich", "maennlich"])
    m["_male"] = male.astype(int)
    ca, pa, cb, pb, p = (int(m.loc[m.y == 1, "_male"].sum()),
                         m.loc[m.y == 1, "_male"].mean() * 100,
                         int(m.loc[m.y == 0, "_male"].sum()),
                         m.loc[m.y == 0, "_male"].mean() * 100,
                         fisher_exact([[int(m.loc[m.y==1,'_male'].sum()), n_c-int(m.loc[m.y==1,'_male'].sum())],
                                       [int(m.loc[m.y==0,'_male'].sum()), n_k-int(m.loc[m.y==0,'_male'].sum())]])[1])
    print(f"  Sex(male)  {ca} ({pa:.1f}%)  |  {cb} ({pb:.1f}%)  |  p={p:.3g}")

    print("\n--- GOLD (COPD only) ---")
    gold = copd["copd_gold_stage"].astype(str).str.lower().value_counts()
    for k in ["i", "ii", "iii", "iv", "unbekannt"]:
        v = int(gold.get(k, 0))
        print(f"  GOLD {k:9s} {v} ({v/n_c*100:.1f}%)")

    print("\n--- comorbidity groups (COPD n(%) | control n(%) | Fisher p) ---")
    for c in ["Respiratory_nonCOPD", "Cardiovascular", "Metabolic", "Depression",
              "Renal", "Dyslipidemia", "HepaticSteatosis"]:
        ca, pa, cb, pb, p = cat(c)
        print(f"  {c:20s} {ca} ({pa:.1f}%)  |  {cb} ({pb:.1f}%)  |  p={p:.3g}")

    # ---- emit LaTeX table (reproducible). "Respiratory (non-COPD)" is
    # deliberately omitted: its original hand-authored definition is not
    # recoverable from committed data (original reported 27.3% in controls,
    # which contradicts the control-recruitment criterion) and it is not
    # load-bearing for the thesis argument. -------------------------------
    def C(col):
        return cont(col)

    def K(col):
        return cat(col)

    age = C("age"); bmi = C("bmi"); pky = copd["pack_years"].dropna()
    smale_c = int(m.loc[m.y == 1, "_male"].sum()); smale_k = int(m.loc[m.y == 0, "_male"].sum())
    sp = fisher_exact([[smale_c, n_c - smale_c], [smale_k, n_k - smale_k]])[1]
    gold = copd["copd_gold_stage"].astype(str).str.lower().value_counts()
    cv, mb, dp, rn = K("Cardiovascular"), K("Metabolic"), K("Depression"), K("Renal")
    dys, hst = K("Dyslipidemia"), K("HepaticSteatosis")

    def f2(m_, s_):
        return f"${m_:.1f} \\pm {s_:.1f}$"

    def pv(p):
        return "$<$0.001" if p < 0.001 else f"{p:.2f}"

    L = [
        "% AUTO-GENERATED by scripts/build_charite_demographics_table.py — do not edit by hand.",
        f"% Cohort N={len(m)} (COPD={n_c}, control={n_k}), reconstructed via the same",
        "% Metadata.filter_for_experiment path as the headline analyses.",
        "\\begin{table}[htbp]", "\\centering",
        (f"\\caption{{Demographic and clinical characteristics of the Charit\\'e cohort "
         f"($N = {len(m)}$; complete-case cohort, all four recording categories, "
         f"see Section~\\ref{{sec:complete-case-restriction}}). Continuous variables: "
         f"mean $\\pm$ SD; categorical: $n$ (\\%). Reproducibly generated.}}"),
        "\\label{tab:charite-demographics}", "\\small",
        "\\begin{tabular}{lrrr}", "\\toprule",
        f"& \\textbf{{COPD}} ($n = {n_c}$) & \\textbf{{Control}} ($n = {n_k}$) & $p$ \\\\",
        "\\midrule",
        "\\textit{Demographics} & & & \\\\",
        f"\\quad Age (years) & {f2(age[0],age[1])} & {f2(age[3],age[4])} & {age[6]:.2f} \\\\",
        f"\\quad Sex (male) & {smale_c} ({smale_c/n_c*100:.1f}\\%) & {smale_k} ({smale_k/n_k*100:.1f}\\%) & {sp:.2f} \\\\",
        f"\\quad BMI (kg/m$^{{2}}$) & {f2(bmi[0],bmi[1])} & {f2(bmi[3],bmi[4])} & {bmi[6]:.2f} \\\\",
        "\\midrule",
        "\\textit{COPD severity (GOLD stage)} & & & \\\\",
    ]
    for k, lab in [("i", "I"), ("ii", "II"), ("iii", "III"), ("iv", "IV"), ("unbekannt", "Unknown")]:
        vv = int(gold.get(k, 0))
        L.append(f"\\quad {lab} & {vv} ({vv/n_c*100:.1f}\\%) & --- & \\\\")
    L += [
        "\\midrule",
        f"\\textit{{Pack-years}} & {f2(pky.mean(),pky.std(ddof=1))} ($n = {pky.notna().sum()}$) & --- & \\\\",
        "\\midrule",
        "\\textit{Comorbidities} & & & \\\\",
        f"\\quad Cardiovascular & {cv[0]} ({cv[1]:.1f}\\%) & {cv[2]} ({cv[3]:.1f}\\%) & {pv(cv[4])}$^{{\\dagger}}$ \\\\",
        f"\\quad Metabolic & {mb[0]} ({mb[1]:.1f}\\%) & {mb[2]} ({mb[3]:.1f}\\%) & {pv(mb[4])}$^{{\\dagger}}$ \\\\",
        f"\\quad Depression & {dp[0]} ({dp[1]:.1f}\\%) & {dp[2]} ({dp[3]:.1f}\\%) & {pv(dp[4])}$^{{\\dagger}}$ \\\\",
        f"\\quad Renal & {rn[0]} ({rn[1]:.1f}\\%) & {rn[2]} ({rn[3]:.1f}\\%) & {pv(rn[4])}$^{{\\dagger}}$ \\\\",
        f"\\quad Dyslipidemia & {dys[0]} ({dys[1]:.1f}\\%) & {dys[2]} ({dys[3]:.1f}\\%) & {pv(dys[4])}$^{{\\dagger}}$ \\\\",
        f"\\quad Hepatic steatosis & {hst[0]} ({hst[1]:.1f}\\%) & {hst[2]} ({hst[3]:.1f}\\%) & {pv(hst[4])}$^{{\\dagger}}$ \\\\",
        "\\bottomrule", "\\end{tabular}", "\\par\\smallskip", "\\footnotesize",
        ("$p$-values from Welch's $t$-test (continuous) or Fisher's exact test "
         "($^{\\dagger}$, categorical). Grouped indicators: Cardiovascular = KHK / "
         "Herzinsuffizienz / VHF; Metabolic = arterial hypertension / diabetes~II; "
         "Renal = chronic kidney disease; Depression. Dyslipidemia and hepatic "
         "steatosis are single raw conditions, both $0\\%$ in controls and omitted "
         "from the propensity model as perfect-separation predictors "
         "(Section~\\ref{sec:psm-protocol}). GOLD = Global Initiative for "
         "Chronic Obstructive Lung Disease."),
        "\\end{table}",
    ]
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text("\n".join(L) + "\n")
    print(f"\nWrote reproducible LaTeX table -> {OUT}")


if __name__ == "__main__":
    main()
