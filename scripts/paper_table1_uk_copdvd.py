"""Table 1 columns for COVID-19 Sounds and COPDVD, reproducibly.

COVID-19 Sounds: the PSM-matched cohort (data/uk_matched_metadata.csv,
100 COPD / 806 controls), and for reference the analysis sample, i.e. the
participants with out-of-fold predictions in the published voice_uk run
(99 / 803; paper_bootstrap_uk.py).
COPDVD: the 48-participant balanced subset (24 greedy +/-5 y same-gender
HC-COPD pairs), built exactly as in scripts/copdvd_holdout_sim.py (mode pair48).

Continuous: mean +/- SD, Welch t-test. Categorical: n (%), Fisher exact.
Writes outputs/paper_checks/table1_uk_copdvd.csv.

Run from the repository root:
    uv run python scripts/paper_table1_uk_copdvd.py
"""
import glob
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact, ttest_ind

UKMETA = "data/uk_matched_metadata.csv"
UK_VOICE = ("data/2025-08-13/mlflow_cluster/271009020022062271/"
            "de720542f5974ee8998d966ca75a536f/artifacts/fold_predictions/*.parquet")
COPDVD = "data/copdvd/AnonymDataSet_ForBinaryClassificationOf_COPD.xlsx"
OUT = Path("outputs/paper_checks/table1_uk_copdvd.csv")


def continuous(cohort, var, a, b):
    a, b = a.dropna(), b.dropna()
    return {"cohort": cohort, "variable": var,
            "copd": f"{a.mean():.1f} ± {a.std():.1f}", "control": f"{b.mean():.1f} ± {b.std():.1f}",
            "n_copd": len(a), "n_control": len(b),
            "p": ttest_ind(a, b, equal_var=False).pvalue}


def binary(cohort, var, a, b):
    a, b = a.dropna().astype(bool), b.dropna().astype(bool)
    table = [[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]]
    return {"cohort": cohort, "variable": var,
            "copd": f"{a.sum()} ({100 * a.mean():.1f}%)", "control": f"{b.sum()} ({100 * b.mean():.1f}%)",
            "n_copd": len(a), "n_control": len(b), "p": fisher_exact(table)[1]}


def uk_rows(cohort, per):
    copd, ctrl = per[per.lung_disease_main == "copd"], per[per.lung_disease_main == "control"]
    rows = [continuous(cohort, "age", copd.age, ctrl.age),
            binary(cohort, "male", copd.sex == "m", ctrl.sex == "m")]
    for c in ["asthma", "hbp", "diabetes"]:
        rows.append(binary(cohort, c, copd[c], ctrl[c]))
    return rows


def greedy_pairs(person):
    """Greedy nearest-age 1:1 HC<->COPD pairing within gender, +/-5y
    (verbatim from scripts/copdvd_holdout_sim.py)."""
    pairs = {'F': [], 'M': []}
    for g in ['F', 'M']:
        copd = person[(person.Label == 1) & (person.Gender == g)].sort_values('Age')
        hc   = person[(person.Label == 0) & (person.Gender == g)].sort_values('Age')
        used = set()
        for cid, crow in copd.iterrows():
            cand = hc[(~hc.index.isin(used)) & (hc.Age.between(crow.Age-5, crow.Age+5))]
            if len(cand):
                best = (cand.Age - crow.Age).abs().idxmin()
                pairs[g].append((cid, best)); used.add(best)
    return pairs


def copdvd_rows():
    df = pd.read_excel(COPDVD)
    person = df.groupby("ID").agg(Label=("Label", "first"), Age=("Age", "first"),
                                  Gender=("Gender", "first"))
    pairs = greedy_pairs(person)
    ids = [i for g in pairs for pr in pairs[g] for i in pr]
    per = person.loc[ids]
    rec = df[df.ID.isin(ids)].groupby("ID").size()
    copd, hc = per[per.Label == 1], per[per.Label == 0]
    rows = [continuous("COPDVD pair48", "age", copd.Age, hc.Age),
            binary("COPDVD pair48", "male", copd.Gender == "M", hc.Gender == "M")]
    rows.append({"cohort": "COPDVD pair48", "variable": "recordings",
                 "copd": str(int(rec.loc[copd.index].sum())), "control": str(int(rec.loc[hc.index].sum())),
                 "n_copd": len(copd), "n_control": len(hc), "p": float("nan")})
    return rows


def main():
    meta = pd.read_csv(UKMETA).drop_duplicates("audio_id").set_index("audio_id")
    analysed = pd.concat([pd.read_parquet(f) for f in glob.glob(UK_VOICE)])["audio_id"].unique()
    rows = (uk_rows("COVID-19 Sounds PSM cohort", meta)
            + uk_rows("COVID-19 Sounds analysis sample", meta.loc[analysed])
            + copdvd_rows())
    t = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)
    with pd.option_context("display.width", 160):
        print(t.assign(p=t.p.round(3)).to_string(index=False))


if __name__ == "__main__":
    main()
