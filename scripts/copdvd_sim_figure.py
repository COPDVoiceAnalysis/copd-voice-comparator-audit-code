"""Figure 3: interval plot of the COPDVD 12-participant holdout simulation.

Shows median + 90 % interval of test-AUROC per model, at participant level
(primary, clinically meaningful) and recording level (the original's reporting
convention). The intervals show substantial split-to-split variability, including for the
authors' own 107-feature configuration. Paired differences in the saved results
show no consistent advantage of the evaluated voice models over age alone.

Reads outputs/copdvd_sim/copdvd_holdout_sim_pair48.npz (faithful 48-pt protocol),
produced by scripts/copdvd_holdout_sim.py.

Usage: copdvd_sim_figure.py [npz] [figures-dir]
Both default relative to this checkout; the figures dir defaults to the
copd-voice-comparator-audit checkout sitting next to it (override with the
positional arg or COPDVD_AUDIT_FIGURES).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import paper_fig_style as S

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_NPZ = _REPO_ROOT / 'outputs' / 'copdvd_sim' / 'copdvd_holdout_sim_pair48.npz'

# an empty first arg means 'default npz', so the figures dir can be passed alone
NPZ = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else _DEFAULT_NPZ
d = np.load(NPZ)

# top -> bottom display order. Hue matches fig2: amber = the published model,
# navy = a voice model, teal = the demographic comparator.
ROWS = [
    ('CatBoost, 107 features\n(published model)', 'cbf', S.AMBER, True),
    ('CatBoost, voice only',            'cb',  S.SAND,  False),
    ('Random forest, voice only',       'rf',  S.NAVY,  False),
    ('Age only',                        'ag',  S.TEAL,  False),
]
LEVELS = [
    ('part', 'A', 'Participant level (primary)'),
    ('rec',  'B', 'Recording level (as published)'),
]

S.apply()

fig, axes = plt.subplots(1, 2, figsize=(S.TEXT_WIDTH, 2.7), sharey=True)
# bottom leaves room for the shared x-label below the tick row
# Panel A's value column sits in the gutter. At a narrow wspace it nearly
# touches panel B and reads as if it belonged to the right-hand plot, so the
# gutter is set to leave a clear margin AFTER the numbers: they sit ~0.03 in
# from their own axes and ~0.4 in from the next one.
fig.subplots_adjust(wspace=0.86, bottom=0.20, top=0.86)

for ax, (lvl, letter, title) in zip(axes, LEVELS):
    ys = np.arange(len(ROWS))[::-1]
    for y, (label, key, colour, emphasis) in zip(ys, ROWS):
        x = d[f'{key}_{lvl}']
        lo, med, hi = np.quantile(x, [0.05, 0.50, 0.95])
        S.interval(ax, lo, hi, med, y, colour)
        S.value_column(ax, y, f'{S.fmt(med)}  [{S.fmt(lo)}, {S.fmt(hi)}]',
                       color=S.INK if emphasis else S.MUTED,
                       weight='semibold' if emphasis else 'normal')

    S.reference_line(ax, 0.5, 'chance')
    if lvl == 'rec':
        # The published value is captioned along its own line, inside the
        # plot. Above the axes the caption sat between the two reference lines
        # and read as if it belonged to the chance line; centred on its line
        # it would run into the value-column header. Right of 0.82 the upper
        # rows are empty (their bars end below 0.77), so the rotated caption
        # touches nothing.
        S.reference_line(ax, 0.82, color=S.FAINT, ls='-', lw=0.9)
        ax.text(0.835, len(ROWS) - 0.35, 'published 0.82', rotation=90,
                ha='left', va='top', fontsize=7.0, color=S.FAINT)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=8.0)
    for tick, (_lbl, _k, _c, emphasis) in zip(ax.get_yticklabels(), ROWS):
        tick.set_color(S.INK if emphasis else S.MUTED)
        if emphasis:
            tick.set_fontweight('semibold')

    # The bracketed values are 5th-95th percentiles over hold-outs, not CIs;
    # a column header says so, level with the reference-line captions.
    S.value_column(ax, len(ROWS) - 0.3, 'median [90% range]', color=S.FAINT,
                   size=7.0)
    ax.texts[-1].set_va('bottom')

    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(-0.7, len(ROWS) - 0.3)
    S.forest_axes(ax)
    S.panel_title(ax, letter, title)

fig.supxlabel('test AUROC over 500 hold-outs of 12 of the 48 participants', fontsize=8.5,
              color=S.MUTED, y=0.045)

fig_dir = S.resolve_fig_dir(sys.argv[2] if len(sys.argv) > 2 else None)
S.fit_to_width(fig)
S.save(fig, 'fig3_copdvd_holdout_sim', fig_dir)

# console check
for lvl, _letter, _title in LEVELS:
    print(f'-- {lvl} --')
    for label, key, _c, _e in ROWS:
        x = d[f'{key}_{lvl}']
        lo, med, hi = np.quantile(x, [0.05, 0.5, 0.95])
        print(f'   {key:4s} med={med:.3f} [{lo:.3f},{hi:.3f}] width={hi-lo:.3f}')
