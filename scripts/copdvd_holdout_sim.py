"""COPDVD: repeated 12-participant holdout simulation (paper §3.6).

Reproducibility script for the comparator audit on the Idrisoglu et al. (2024)
COPDVD cohort. Lives in the repo (not /tmp) so results survive and the analysis
is reproducible alongside the manuscript.

Three sampling modes:
  - 'pair48' (FAITHFUL primary): cohort = the reproduced 48-pt balanced subset
        (24 greedy +/-5y same-gender pairs built from all 68; 24F + 24M,
        24 COPD + 24 HC, 20 excluded -- composition confirmed identical to the
        published subset). Per holdout: 3 F-pairs + 3 M-pairs -> 12 test, train
        on the remaining 36. Matches the published 48 -> 12-test/36-train protocol.
  - 'pair'  : pairs sampled as test, train pool = 60-pt coverage cohort minus test
        (intermediate; superseded by pair48, kept only for comparison).
  - 'gxc'   (SENSITIVITY): 60-pt coverage cohort; independent stratified sampling
        of 3 F-COPD + 3 F-HC + 3 M-COPD + 3 M-HC per holdout.

Models use Idrisoglu's reported best hyperparameters. Their inner grid search
optimised ACCURACY (sklearn GridSearchCV default .score()), confirmed by the
first author in correspondence (2026-06); we apply their reported best config
directly rather than re-tuning.

NOTE ON INTERPRETATION: the author's actual 12-pt test set was selected to be
"representative" (age/gender-balanced), NOT randomly drawn. This script
characterises the *protocol's* sampling distribution; it is not a hypothesis
test on the author's specific reported value, which cannot be reproduced
because the public release is irreversibly anonymised.

Usage: python scripts/copdvd_holdout_sim.py {pair48|pair|gxc} [out_dir]
"""
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

MODE = sys.argv[1] if len(sys.argv) > 1 else 'pair48'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else 'outputs/copdvd_sim'
assert MODE in ('pair48', 'pair', 'gxc')
B = 500
N_WORKERS = 8
SEED_BASE = 20260530

import os
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_excel('data/copdvd/AnonymDataSet_ForBinaryClassificationOf_COPD.xlsx')
person_all = df.groupby('ID').agg(Label=('Label', 'first'), Age=('Age', 'first'),
                                  Gender=('Gender', 'first'))


def greedy_pairs(person):
    """Greedy nearest-age 1:1 HC<->COPD pairing within gender, +/-5y."""
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


if MODE == 'pair48':
    pairs = greedy_pairs(person_all)
    paired_ids = [i for g in pairs for pr in pairs[g] for i in pr]
    m = df[df['ID'].isin(paired_ids)].copy()
else:
    keep = [pid for pid, row in person_all.iterrows()
            if len(person_all[(person_all.Label != row.Label) &
                              (person_all.Gender == row.Gender) &
                              person_all.Age.between(row.Age-5, row.Age+5)]) > 0]
    m = df[df['ID'].isin(keep)].copy()
    pairs = greedy_pairs(m.groupby('ID').agg(
        Label=('Label', 'first'), Age=('Age', 'first'), Gender=('Gender', 'first')))

m['gender_m'] = (m['Gender'] == 'M').astype(int)
META = {'ID', 'Label', 'Age', 'Gender', 'gender_m', 'Cold', 'Pain', 'Slimy', 'Other'}
voice_cols = [c for c in m.columns if c not in META]
# Idrisoglu's full 107-feature set = 101 voice (BLA + MFCC) + Age + Gender +
# 4 symptom flags. CatBoost on this set is the model that produced the 0.82
# headline (Age is their #1 SHAP feature); voice-only CatBoost is NOT.
full_cols = voice_cols + ['Age', 'gender_m', 'Cold', 'Pain', 'Slimy', 'Other']
per = m.groupby('ID').agg(Label=('Label', 'first'), Age=('Age', 'first'),
                          Gender=('Gender', 'first'))

print(f'mode={MODE}  B={B}  workers={N_WORKERS}')
print(f'cohort: {len(per)} pts ({(per.Label==1).sum()} COPD / {(per.Label==0).sum()} HC), '
      f'{len(m)} recordings, {len(voice_cols)} voice features')
print(f'pairs: F={len(pairs["F"])} M={len(pairs["M"])}  '
      f'(train pool per holdout = {len(per) - 12} pts)')


def sample_holdout(rng, mode, per, pairs):
    if mode in ('pair48', 'pair'):
        f_idx = rng.choice(len(pairs['F']), 3, replace=False)
        m_idx = rng.choice(len(pairs['M']), 3, replace=False)
        ids = []
        for i in f_idx: ids.extend(pairs['F'][i])
        for i in m_idx: ids.extend(pairs['M'][i])
        return np.array(ids)
    pools = {(g, l): per.index[(per.Gender == g) & (per.Label == l)].to_numpy()
             for g in ['F', 'M'] for l in [0, 1]}
    return np.concatenate([rng.choice(pools[k], 3, replace=False) for k in pools])


def fit_and_score(model, Xtr, ytr, Xte, yte, ids_te):
    pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc', StandardScaler()), ('clf', model)])
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    auc_rec = roc_auc_score(yte, p)
    pt = (pd.DataFrame({'g': ids_te, 'y': yte, 'p': p})
          .groupby('g').agg(y=('y', 'first'), p=('p', 'mean')))
    return auc_rec, roc_auc_score(pt['y'], pt['p'])


def one_holdout(b, mode, m, per, pairs, voice_cols):
    rng = np.random.default_rng(SEED_BASE + b)
    test_ids = sample_holdout(rng, mode, per, pairs)
    tr = m[~m['ID'].isin(test_ids)]; te = m[m['ID'].isin(test_ids)]
    Xtr_v = tr[voice_cols].values.astype(float); Xte_v = te[voice_cols].values.astype(float)
    Xtr_f = tr[full_cols].values.astype(float);  Xte_f = te[full_cols].values.astype(float)
    Xtr_a = tr[['Age']].values.astype(float);    Xte_a = te[['Age']].values.astype(float)
    ytr = tr['Label'].values; yte = te['Label'].values; ids_te = te['ID'].values

    def cb_clf():
        return CatBoostClassifier(iterations=300, learning_rate=0.2, depth=4,
                                  l2_leaf_reg=3, auto_class_weights='Balanced',
                                  random_seed=42, verbose=False, thread_count=1,
                                  allow_writing_files=False)
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42,
                                    class_weight='balanced', n_jobs=1)
        lr = LogisticRegression(max_iter=2000, class_weight='balanced')
        r = fit_and_score(rf,       Xtr_v, ytr, Xte_v, yte, ids_te)  # RF voice-only
        c = fit_and_score(cb_clf(), Xtr_v, ytr, Xte_v, yte, ids_te)  # CatBoost voice-only
        f = fit_and_score(cb_clf(), Xtr_f, ytr, Xte_f, yte, ids_te)  # CatBoost full 107 (THEIR model)
        a = fit_and_score(lr,       Xtr_a, ytr, Xte_a, yte, ids_te)  # age-only LR
        return (r[0], r[1], c[0], c[1], a[0], a[1], f[0], f[1])
    except ValueError:
        return None


print(f'\nlaunching {B} holdouts across {N_WORKERS} workers...')
results = Parallel(n_jobs=N_WORKERS, backend='loky', verbose=5)(
    delayed(one_holdout)(b, MODE, m, per, pairs, voice_cols) for b in range(B))
results = [r for r in results if r is not None]
print(f'completed: {len(results)}/{B}')

arr = np.array(results)
(rf_rec, rf_part, cb_rec, cb_part, ag_rec, ag_part,
 cbf_rec, cbf_part) = (arr[:, i] for i in range(8))
out = f'{OUT_DIR}/copdvd_holdout_sim_{MODE}.npz'
np.savez(out, rf_rec=rf_rec, rf_part=rf_part, cb_rec=cb_rec, cb_part=cb_part,
         ag_rec=ag_rec, ag_part=ag_part, cbf_rec=cbf_rec, cbf_part=cbf_part)
print(f'saved -> {out}')


def summary(name, x):
    q = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])
    print(f'  {name:22s}  med={q[2]:.3f}  IQR=[{q[1]:.3f}, {q[3]:.3f}]  '
          f'90%CI=[{q[0]:.3f}, {q[4]:.3f}]  mean={x.mean():.3f}')

print(f'\n=== Test-AUROC over B={len(rf_rec)} ({MODE}) ===')
print('-- recording-level --')
summary('CatBoost FULL (107, theirs)', cbf_rec)
summary('CatBoost voice-only (101)', cb_rec)
summary('RF voice-only (101)', rf_rec)
summary('age-only LR (1)', ag_rec)
print('-- participant-level --')
summary('CatBoost FULL (107)', cbf_part)
summary('CatBoost voice-only', cb_part)
summary('RF voice-only', rf_part)
summary('age-only LR', ag_part)

print('\n=== Headline placement (vs their CatBoost test AUROC = 0.82) ===')
for a_, lbl in [(cbf_rec, 'CB-full(107)'), (cb_rec, 'CB-voice(101)'),
                (rf_rec, 'RF-voice'), (ag_rec, 'age-only')]:
    print(f'  {lbl:14s}  {float((a_<=0.82).mean())*100:5.1f} pctile   '
          f'P(>=0.82)={float((a_>=0.82).mean()):.3f}')

print('\n=== Decomposition: paired deltas (rec-level, same holdout) ===')
for name, d in [('CB-full   - CB-voice (demographic add-on)', cbf_rec - cb_rec),
                ('CB-voice  - age-only (voice over age)',     cb_rec - ag_rec),
                ('CB-full   - age-only (full over age)',      cbf_rec - ag_rec)]:
    q = np.quantile(d, [0.05, 0.50, 0.95])
    print(f'  {name:42s}  median={q[1]:+.3f}  90%CI=[{q[0]:+.3f}, {q[2]:+.3f}]  '
          f'frac(>0)={(d>0).mean():.3f}')
