"""Single source of truth for the revision's model runs ("arms").

Two subcommands:

  make-configs  Write one training config per voice arm (and optionally the
                layer sweep) with the pipeline's own config generator, plus
                train_list.txt (one config path per line) for the Slurm array.
  resolve       After training, find each arm's finished MLflow run by run
                name and write the arm files that paired_bootstrap.py,
                operating_points.py and fold_prediction_figure.py read.

Voice arms reuse the published selected configurations:
  Charité  norm_on__voice_only__wav2vec_poem__extended__copd_classification_charite_complete__split__SVC
  UK       norm_on__voice_only__wav2vec_poem__extended__copd_classification_uk_only__unsplit__LogisticRegression
and add the same configuration with age+sex and with age+sex+individual
comorbidities (Reviewer 1, point 1). Demographic comparators come from
scripts/comparator_weighting_sensitivity.py, trained once with the
ps_overlap weights (as published) and once without weights ("flat").

The MLflow run name of a training run is its config file name without
"__config.yml" (src/train.py), and a comparator run is named
"<scope>__extended__<task>__<weighting>__seed42" (src/train_seed_sweep.py).

Usage (from the repository root):
  uv run python scripts/revision/arms.py make-configs --preproc norm_on \
      --data-dir data/2025-08-13 --out-dir $OUT/configs [--layers] [--smoke]
  uv run python scripts/revision/arms.py resolve --preproc norm_on \
      --mlruns $OUT/mlruns --comparator-mlruns $OUT/comparators/mlruns \
      --out-dir $OUT/arms
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.generate_experiment_configs import (  # noqa: E402
    ConfigurationGenerator,
    DirectoryConfig,
    ExperimentParams,
)

EXPERIMENT_VERSION = "revision_r1"
COHORTS = {
    "charite": {"task": "copd_classification_charite_complete", "splitting": "split", "classifier": "SVC"},
    "uk": {"task": "copd_classification_uk_only", "splitting": "unsplit", "classifier": "LogisticRegression"},
}
VOICE_SCOPES = {
    "voice": "voice_only",
    "voice+age+sex": "voice_plus_age_sex",
    "voice+age+sex+comorb": "voice_plus_age_sex_comorbidity_individual",
}
# label -> (scope, weighting) of the comparator runs, per cohort
COMPARATORS = {
    "charite": {"age": "age_only", "age+sex": "age_sex_only", "age+sex+comorb": "age_sex_comorbidity_individual"},
    "uk": {"age+sex": "age_sex_only", "age+sex+comorb": "age_sex_comorbidity_individual"},
}
LAYERS = list(range(25))
# Published runs (paper numbers): label -> [(MLflow experiment id, run id), ...],
# from scripts/paper_bootstrap_charite.py and scripts/paper_bootstrap_uk.py
# (the UK age+sex+comorbidity comparator pools two runs, as in the paper). The
# comparators here were trained with ps_overlap weights.
PUBLISHED = {
    "charite_voice": [("457012274041432773", "00a61b66628b45e8906c95097abd7348")],
    "charite_age+sex (weighted)": [("457012274041432773", "b9dc00497d77462e8a76f9515b348af2")],
    "charite_age+sex+comorb (weighted)": [("457012274041432773", "56f48804a8674b4ab6a5ed2098886c19")],
    "uk_voice": [("271009020022062271", "de720542f5974ee8998d966ca75a536f")],
    "uk_age+sex (weighted)": [("271009020022062271", "629aed426f2f4a4b9df891fc8b08b402")],
    "uk_age+sex+comorb (weighted)": [("271009020022062271", "2569dbc2b5444f50b6187910b263c8ff"),
                                     ("271009020022062271", "4906d6f489c24893a804cb14cdd7be68")],
}


def voice_id(preproc, cohort, scope):
    c = COHORTS[cohort]
    return f"{preproc}__{scope}__wav2vec_poem__extended__{c['task']}__{c['splitting']}__{c['classifier']}"


def layer_id(preproc, layer, stat="mean"):
    base = voice_id(preproc, "charite", "voice_only")
    return f"{base}__layer{layer:02d}" + ("" if stat == "mean" else f"__{stat}")


def smoke(cfg):
    """Shrink a config so one run takes minutes: for testing the chain, never for results."""
    t = cfg["training_config"]
    t.update(n_repeats_outer=1, n_repeats_inner=1, n_splits_inner=3)
    t["classifier_hyperparams"] = {k: v[:1] for k, v in t["classifier_hyperparams"].items()}
    t["pca_hyperparams"] = {k: v[:1] for k, v in t["pca_hyperparams"].items()}
    return cfg


def make_configs(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = Path(a.data_dir)
    dirs = lambda path: DirectoryConfig(str(d / f"parselmouth__{a.preproc}"), str(d / f"wav2vec2__{a.preproc}"),
                                        str(d / f"preprocessed__{a.preproc}" / "metadata.csv"), str(path))
    gen = ConfigurationGenerator(experiment_version=EXPERIMENT_VERSION)
    # config/snakemake_matrix.yml has voice_plus_age_sex commented out for the
    # published grid; allow the revision's scopes in memory, leave the file as is.
    gen.matrix_config["scope_variants"] = sorted(set(gen.matrix_config.get("scope_variants") or [])
                                                 | set(VOICE_SCOPES.values()))
    written = []
    for cohort, c in COHORTS.items():
        for scope in VOICE_SCOPES.values():
            cid = voice_id(a.preproc, cohort, scope)
            path = out / f"{cid}__config.yml"
            params = ExperimentParams(a.preproc, scope, "wav2vec_poem", "extended", c["task"],
                                      c["classifier"], c["splitting"])
            if not gen.generate_single_config(params, dirs(path)):
                raise RuntimeError(f"Generator skipped {cid}")
            written.append(path)
    if a.layers:
        template = yaml.safe_load(open(out / f"{voice_id(a.preproc, 'charite', 'voice_only')}__config.yml"))
        for layer, stat in [(L, "mean") for L in LAYERS] + [(4, "mean_std")]:
            cfg = json.loads(json.dumps(template))
            cfg["feature_config"]["wav2vec2_layer"] = layer
            cfg["feature_config"]["wav2vec2_embedding_statistics"] = stat
            cid = layer_id(a.preproc, layer, stat)
            cfg["experiment_name"] = cid
            path = out / f"{cid}__config.yml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            written.append(path)
    for path in written:
        cfg = yaml.safe_load(open(path))
        if a.smoke:
            cfg = smoke(cfg)
            cfg["description"] = "SMOKE TEST (reduced CV and grids) -- not a result. " + cfg["description"]
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "train_list.txt").write_text("".join(f"{p}\n" for p in written))
    print(f"{len(written)} configs in {out}; list in {out / 'train_list.txt'}")


def find_runs(mlruns):
    """run name -> run directory, for FINISHED runs; the newest wins when a name repeats."""
    runs = {}
    for meta in Path(mlruns).glob("*/*/meta.yaml"):
        m = yaml.safe_load(open(meta))
        if m.get("status") != 3:  # 3 = FINISHED
            continue
        name_file = meta.parent / "tags" / "mlflow.runName"
        name = name_file.read_text().strip() if name_file.exists() else m.get("run_name")
        if not list((meta.parent / "artifacts" / "fold_predictions").glob("*.parquet")):
            continue
        if name not in runs or m["start_time"] > runs[name][0]:
            runs[name] = (m["start_time"], str(meta.parent))
    return {k: v[1] for k, v in runs.items()}


def resolve(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = {**find_runs(a.mlruns), **find_runs(a.comparator_mlruns)}
    missing = []

    def get(name):
        if name not in runs:
            missing.append(name)
        return runs.get(name)

    for cohort, c in COHORTS.items():
        arms = {label: get(voice_id(a.preproc, cohort, scope)) for label, scope in VOICE_SCOPES.items()}
        for label, scope in COMPARATORS[cohort].items():
            for w, suffix in [("flat", ""), ("ps_overlap", " (weighted)")]:
                arms[label + suffix] = get(f"{scope}__extended__{c['task']}__{w}__seed42")
        contrasts = [["voice+age+sex", "voice"], ["voice+age+sex+comorb", "voice"]]
        for label in COMPARATORS[cohort]:
            contrasts.append(["voice", label])
            contrasts.append(["voice", label + " (weighted)"])
        contrasts += [["voice+age+sex", "age+sex"], ["voice+age+sex+comorb", "age+sex+comorb"]]
        arms = {k: v for k, v in arms.items() if v}
        contrasts = [p for p in contrasts if p[0] in arms and p[1] in arms]
        json.dump({"arms": arms, "contrasts": contrasts}, open(out / f"{cohort}.json", "w"), indent=1)
        fig = {k: arms[k] for k in ["voice", "age+sex (weighted)", "age+sex"] if k in arms}
        fig = {{"voice": "Voice", "age+sex (weighted)": "Age+sex\n(weighted)",
                "age+sex": "Age+sex\n(unweighted)"}[k]: v for k, v in fig.items()}
        json.dump({"arms": fig}, open(out / f"{cohort}_figure.json", "w"), indent=1)
    layers = {f"layer{L:02d}": runs.get(layer_id(a.preproc, L)) for L in LAYERS}
    layers["layer04_mean_std"] = runs.get(layer_id(a.preproc, 4, "mean_std"))
    layers = {k: v for k, v in layers.items() if v}
    if layers:
        json.dump({"arms": layers, "contrasts": [[k, "layer04"] for k in layers if k != "layer04" and "layer04" in layers]},
                  open(out / "layers.json", "w"), indent=1)
    print(f"arm files in {out}; {len(layers)} layer runs found")
    if missing:
        print("MISSING runs (not finished or not found):\n  " + "\n  ".join(missing))
        if a.strict:
            sys.exit(1)


def published_files(mlflow, label):
    """Fold-prediction files of a published arm, or [] if any of its runs is unreadable."""
    files = []
    for exp, run in PUBLISHED[label]:
        try:
            f = sorted((Path(mlflow) / exp / run / "artifacts" / "fold_predictions").glob("*.parquet"))
        except OSError:  # unreadable directory of a former account
            f = []
        if not f:
            return []
        files += f
    return files


def published(a):
    """Arm files for the published runs (paper numbers), when their fold predictions are readable.

    Each arm gets a directory <out>/published_runs/<label>/artifacts/fold_predictions/
    with symlinks to the published parquet files, so pooled arms load like one run.
    Nothing is copied.
    """
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for cohort in ("charite", "uk"):
        arms = {}
        for label in PUBLISHED:
            c, _, arm = label.partition("_")
            files = published_files(a.mlflow, label) if c == cohort else []
            if not files:
                continue
            d = out / "published_runs" / label.replace(" ", "_").replace("(", "").replace(")", "")
            fp = d / "artifacts" / "fold_predictions"
            fp.mkdir(parents=True, exist_ok=True)
            for f in files:
                link = fp / f"{f.parent.parent.parent.name}_{f.name}"
                if not link.exists():
                    link.symlink_to(f)
            arms[arm] = str(d)
        if len(arms) < 2:
            print(f"published {cohort}: fewer than two runs readable, skipped")
            continue
        contrasts = [["voice", k] for k in arms if k != "voice" and "voice" in arms]
        json.dump({"arms": arms, "contrasts": contrasts}, open(out / f"published_{cohort}.json", "w"), indent=1)
        print(f"published {cohort}: {sorted(arms)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-configs")
    m.add_argument("--preproc", default="norm_on")
    m.add_argument("--data-dir", default="data/2025-08-13")
    m.add_argument("--out-dir", required=True)
    m.add_argument("--layers", action="store_true", help="Also write the layer sweep (Reviewer 1, point 6)")
    m.add_argument("--smoke", action="store_true", help="Tiny CV and grids, to test the chain only")
    r = sub.add_parser("resolve")
    r.add_argument("--preproc", default="norm_on")
    r.add_argument("--mlruns", required=True)
    r.add_argument("--comparator-mlruns", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--strict", action="store_true", help="Exit 1 if any voice or comparator run is missing")
    v = sub.add_parser("voice-run-name", help="Print the run name of a cohort's voice arm")
    v.add_argument("--preproc", default="norm_on")
    v.add_argument("--cohort", choices=list(COHORTS), required=True)
    v.add_argument("--scope", default="voice_only")
    d = sub.add_parser("run-dir", help="Print the directory of the newest finished run with this name")
    d.add_argument("--mlruns", required=True)
    d.add_argument("--name", required=True)
    p = sub.add_parser("published", help="Write arm files for the published runs, if readable")
    p.add_argument("--mlflow", required=True, help="MLflow directory of the published analysis")
    p.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "make-configs":
        make_configs(a)
    elif a.cmd == "resolve":
        resolve(a)
    elif a.cmd == "voice-run-name":
        print(voice_id(a.preproc, a.cohort, a.scope))
    elif a.cmd == "run-dir":
        runs = find_runs(a.mlruns)
        if a.name not in runs:
            sys.exit(f"no finished run named {a.name} in {a.mlruns}")
        print(runs[a.name])
    else:
        published(a)


if __name__ == "__main__":
    main()
