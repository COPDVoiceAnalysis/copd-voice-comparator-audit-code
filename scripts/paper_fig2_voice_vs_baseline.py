"""Figure 2: voice vs. demographic comparators across the three cohorts.

Panel A: strata-weighted AUROC with 95 % subject-bootstrap CIs for the voice
model and each demographic comparator, per cohort. Filled symbols are the
comparators of choice, trained without covariate weights (flat, extended grid,
seed 42); in the two matched cohorts the age comparator is raw age. Open
symbols are the published comparators trained with the voice model's overlap
weights, shown only as a degenerate contrast. Panel B: the paired voice - age
difference per cohort, which is the quantity the claim actually rests on.

Every value is read from the bootstrap outputs, so the numbers are not retyped
here:

    outputs/_tmp_boot/charite_singleseed_results.json  -> Charite voice rows,
                                                          overlap-weighted rows
    outputs/_tmp_boot/uk_full_results.json             -> COVID-19 Sounds voice,
                                                          raw age, overlap rows
    outputs/_tmp_boot/fig2_comparators.json            -> flat comparators, PSM
                                                          raw age, their paired
                                                          differences

written unrounded by scripts/paper_bootstrap_charite.py,
scripts/paper_bootstrap_uk.py and scripts/paper_comparator_weighting_analysis.py.
Keep them that way: this script rounds for display, and rounding the files as
well is double rounding (a bound of 0.4147 stored as 0.415 printed as 0.42).

Provenance, verified by recomputing the pooled subject-level bootstrap from
the stored fold predictions (B = 5000, class-stratified over participants):

    Charite complete, voice   run 00a61b66628b
                              norm_on__voice_only__wav2vec_poem__..._split__SVC
    Charite PSM, voice        run 884ea1802c0d, 2026-05-20 16:13
                              norm_on__voice_only__parsel_poem_spectral__..._split__SVC
    Charite PSM, age          raw age on the voice_psm participants, scored
                              with that run's strata weights. The earlier
                              "age only" run 8f8cfddc0ae1 is not used: it
                              resolved no scope and trained on 193 Parselmouth
                              /a/ features plus age, sex and BMI.

Beware the neighbouring outputs/_tmp_boot/subject_bootstrap_results.json: it
is a superseded vintage computed on the 50-participant PSM cohort (37/13) and
disagrees with the published PSM numbers. The PSM cohort went through three
versions -- 50, then 52, then the current 55 participants (43 COPD, 12
controls) in data/2025-08-13/psm_matched_charite.csv -- and only the two
configurations above were re-run against the final one. Runs at n=50 are not
the published analysis.

Usage: paper_fig2_voice_vs_baseline.py [figures-dir]
"""
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

import paper_fig_style as S

import json
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[1] / 'outputs' / '_tmp_boot'
_CHARITE = json.loads((_BOOT / 'charite_singleseed_results.json').read_text())
_UK = json.loads((_BOOT / 'uk_full_results.json').read_text())
_FLAT = json.loads((_BOOT / 'fig2_comparators.json').read_text())


def est(store, key):
    """Point estimate and 95 % CI for one row."""
    v = store[key]
    return v['point'], v['ci'][0], v['ci'][1]


def diff(store, key):
    """Paired difference: point, CI, and share of resamples above zero.

    The two files spell these fields differently.
    """
    v = store[key]
    pt = v['diff_point'] if 'diff_point' in v else v['diff']
    frac = v['frac_diff_gt0'] if 'frac_diff_gt0' in v else v['frac_gt0']
    return pt, v['diff_ci'][0], v['diff_ci'][1], frac


# Role drives colour, symbol, emphasis, and which row panel B lines up with,
# so the alignment is stated in the data rather than implied by row order.
#   voice      - the voice model
#   ref        - the age comparator the paired difference is taken against
#   comparator - a further demographic model trained without covariate weights
#   overlap    - the same model trained with the voice model's overlap weights;
#                intercept-only in most folds, shown open as a contrast
ROLE_COLOUR = {'voice': S.NAVY, 'ref': S.TEAL, 'comparator': S.TEAL,
               'overlap': S.GREY}
HOLLOW = {'overlap'}
OVERLAP_MARK = '†'
OVERLAP_NOTE = f"{OVERLAP_MARK} trained with the voice model's overlap weights"

GROUPS = [
    ('Charité complete', 'N = 99', [
        ('voice (wav2vec poem split)',      *est(_CHARITE, 'voice_complete'), 'voice'),
        ('age alone',                       *est(_FLAT, 'age_only_complete_flat'), 'ref'),
        ('age + sex',                       *est(_FLAT, 'age_sex_complete_flat'), 'comparator'),
        ('age + sex + comorbidity',         *est(_FLAT, 'age_sexcomorb_complete_flat'), 'comparator'),
        (f'age + sex{OVERLAP_MARK}',        *est(_CHARITE, 'age_sex_complete'), 'overlap'),
        (f'age + sex + comorbidity{OVERLAP_MARK}',
                                            *est(_CHARITE, 'age_sexcomorb_complete'), 'overlap'),
    ], *diff(_FLAT, 'paired_complete_voice_minus_age_only_flat')),
    ('Charité PSM', 'N = 55', [
        ('voice (parselmouth poem spectral)', *est(_CHARITE, 'voice_psm'), 'voice'),
        ('age (raw score)',                   *est(_FLAT, 'age_raw_psm'), 'ref'),
    ], *diff(_FLAT, 'paired_psm_voice_minus_age_raw')),
    ('COVID-19 Sounds', 'N = 902', [
        ('voice (wav2vec read sentence)', *est(_UK, 'voice_uk'), 'voice'),
        ('age (raw score)',               *est(_UK, 'age_only_uk'), 'ref'),
        ('age + sex',                     *est(_FLAT, 'age_sex_uk_flat'), 'comparator'),
        ('age + sex + comorbidity',       *est(_FLAT, 'age_sexcomorb_uk_flat'), 'comparator'),
        (f'age + sex{OVERLAP_MARK}',      *est(_UK, 'age_sex_uk'), 'overlap'),
        (f'age + sex + comorbidity{OVERLAP_MARK}',
                                          *est(_UK, 'age_sex_comorb_uk'), 'overlap'),
    ], *diff(_UK, 'paired_voice_minus_age_only_uk')),
]

S.apply()

# ---------------------------------------------------------------- geometry
# Every column width below is the extent of the text that goes in it, measured
# at the size it will be drawn. The figure is then solved to fill the journal
# text block exactly: the two plot areas take whatever width the labels leave.
# Nothing here is a tuned pixel constant, so re-wording a label re-fits the
# figure instead of silently overflowing it.

TARGET_W = 6.30      # text width of a 1 in-margin A4 page, the figure's slot
GAP_LA = 0.07        # row-label column -> panel A
GAP_AB = 0.07        # panel A -> right edge of its tinted band
GUTTER = 0.26        # band edge -> panel B
PAD_COL = 0.02       # slack so a glyph never touches its column edge
BAND_PAD = 0.06      # left edge of the tinted band -> widest label
ROW_U = 0.175        # inches per y unit; sets the row pitch

# Panel A carries no value column: Table 2 gives the numbers, the figure the
# comparison. Cohort names are bold and flush left; Helvetica Neue has no bold
# face that matplotlib can load, so they are set in Arial Bold.
HEAD_FONT = 'Arial'
HEAD_GAP_PT = 4      # cohort name -> its (N = ...)

FS_ROW, FS_HEAD, FS_VAL, FS_FRAC, FS_TITLE = 8.3, 8.8, 7.8, 7.2, 9.5

PANEL_B_TITLE = 'Voice advantage\nover age'
PANEL_B_XLABEL = 'paired difference:\nvoice − age'
PANEL_A_XLABEL = 'strata-weighted AUROC'

# ---- measurement pass: text extents in inches, independent of any axes ----
_probe_fig, _probe_ax = plt.subplots(figsize=(10, 6))
_probe_r = _probe_fig.canvas.get_renderer()


def text_size(s, size, weight='normal', family=None):
    kw = {'fontfamily': family} if family else {}
    t = _probe_ax.text(0, 0, s, fontsize=size, fontweight=weight, alpha=0, **kw)
    bb = t.get_window_extent(_probe_r)
    t.remove()
    return bb.width / _probe_fig.dpi, bb.height / _probe_fig.dpi


def widest(strings, size, weight='normal'):
    return max(text_size(s, size, weight)[0] for s in strings)


def header_width(gname, gn):
    """Bold cohort name, a small gap, then its (N = ...) in regular weight."""
    return (text_size(gname, FS_HEAD, 'bold', HEAD_FONT)[0] + HEAD_GAP_PT / 72
            + text_size(f'({gn})', FS_HEAD)[0])


def _minus(text):
    """ASCII hyphen -> typographic minus, to match the axis tick labels."""
    return text.replace('-', '−')


def fmt_est(pt, lo, hi):
    return f'{S.fmt(pt)}  [{S.fmt(lo)}, {S.fmt(hi)}]'


def fmt_diff(dp, dlo, dhi):
    return _minus(f'{S.fmt(dp, signed=True)}  [{S.fmt(dlo, signed=True)}, {S.fmt(dhi, signed=True)}]')


def fmt_frac(frac):
    return f'{100 * frac:.0f} % of resamples > 0'


_row_labels = [lab for _g, _n, entries, *_d in GROUPS for lab, *_r in entries]
_voice_labels = [lab for _g, _n, entries, *_d in GROUPS
                 for lab, _p, _l, _h, role in entries if role == 'voice']
_diffs = [fmt_diff(dp, dlo, dhi) for *_g, dp, dlo, dhi, _f in GROUPS]
_fracs = [fmt_frac(f) for *_g, _dp, _dlo, _dhi, f in GROUPS]

# The voice rows are semibold, so they are measured semibold.
LABEL_W = max(widest([l for l in _row_labels if l not in _voice_labels], FS_ROW),
              widest(_voice_labels, FS_ROW, 'semibold'),
              max(header_width(g, n) for g, n, *_ in GROUPS)) + PAD_COL

# Panel B is exactly as wide as the widest thing it has to carry: its own
# title, its (wrapped) axis label, or one of the difference annotations.
AXB_W = max(widest(PANEL_B_TITLE.split('\n')[:1], FS_TITLE, 'semibold')
            + text_size('(B)  ', FS_TITLE, 'semibold')[0],
            widest(PANEL_B_TITLE.split('\n')[1:], FS_TITLE, 'semibold'),
            widest(PANEL_B_XLABEL.split('\n'), 9),
            widest(_diffs, FS_VAL, 'semibold'),
            widest(_fracs, FS_FRAC)) + PAD_COL

FIXED_W = BAND_PAD + LABEL_W + GAP_LA + GAP_AB + GUTTER + AXB_W
AXA_W = TARGET_W - 2 * S.SAVE_PAD - FIXED_W
if AXA_W < 1.2:
    sys.exit(f'panel A would be only {AXA_W:.2f} in wide; shorten a label '
             f'or raise TARGET_W (currently {TARGET_W} in).')

ANN_H = max(text_size(t, FS_VAL, 'semibold')[1] for t in _diffs)

plt.close(_probe_fig)

# ---- vertical layout: walk the rows once to fix every y before drawing ----
HEADER_STEP, ROW_STEP = 1.55, 1.0

y = 0.0
rows, headers, group_spans, ref_y = [], [], [], []
for gname, gn, entries, *_ in GROUPS:
    y -= HEADER_STEP
    headers.append((y, gname, gn))
    span_top = y
    for label, pt, lo, hi, role in entries:
        y -= ROW_STEP
        rows.append((y, label, pt, lo, hi, ROLE_COLOUR[role], role == 'voice',
                     role in HOLLOW))
        if role == 'ref':
            ref_y.append(y)
    group_spans.append((span_top, y))

Y_LO, Y_HI = y - 0.75, 0.05
PLOT_H = (Y_HI - Y_LO) * ROW_U

# The canvas only has to be big enough to hold the laid-out content plus the
# text that hangs off it; savefig's tight bbox crops to what is actually drawn.
CANVAS_W, CANVAS_H = TARGET_W + 2.0, PLOT_H + 2.0
fig = plt.figure(figsize=(CANVAS_W, CANVAS_H))

_x0, _y0 = 1.0, 1.0                       # content origin on the canvas
AXA_X = _x0 + BAND_PAD + LABEL_W + GAP_LA
AXB_X = AXA_X + AXA_W + GAP_AB + GUTTER

axA = fig.add_axes([AXA_X / CANVAS_W, _y0 / CANVAS_H,
                    AXA_W / CANVAS_W, PLOT_H / CANVAS_H])
axB = fig.add_axes([AXB_X / CANVAS_W, _y0 / CANVAS_H,
                    AXB_W / CANVAS_W, PLOT_H / CANVAS_H])

# The label column is placed in axes fractions converted from the measured
# inches, so the tinted band can close flush around it. Cohort names start
# where the widest row label starts.
LAB_X = -GAP_LA / AXA_W
HEAD_X = -(GAP_LA + LABEL_W - PAD_COL) / AXA_W
BAND_L = -(GAP_LA + LABEL_W + BAND_PAD) / AXA_W
BAND_R = 1 + GAP_AB / AXA_W

# ---------------------------------------------------------------- panel A
for y_pos, label, pt, lo, hi, colour, is_voice, hollow in rows:
    S.interval(axA, lo, hi, pt, y_pos, colour, hollow=hollow)
    axA.text(LAB_X, y_pos, label, transform=axA.get_yaxis_transform(),
             ha='right', va='center', fontsize=FS_ROW,
             color=S.INK if is_voice else S.MUTED,
             fontweight='semibold' if is_voice else 'normal')

# One tinted band per cohort, drawn across BOTH panels: it is what shows the
# reader which interval in B belongs to which block of rows in A. Each panel
# has a single x range, so the three bands close on the same two edges.
BAND = '#F4F6F9'
for _span_top, _span_bot in group_spans:
    # panel A's band runs the full width of its columns, so one cohort reads
    # as a single block: row labels, intervals and value column together
    axA.axhspan(_span_bot - 0.55, _span_top + 0.72, xmin=BAND_L, xmax=BAND_R,
                facecolor=BAND, edgecolor='none', zorder=-1, clip_on=False)
    axB.axhspan(_span_bot - 0.55, _span_top + 0.72, facecolor=BAND,
                edgecolor='none', zorder=-1)

for y_pos, gname, gn in headers:
    _name = axA.text(HEAD_X, y_pos, gname, transform=axA.get_yaxis_transform(),
                     ha='left', va='center', fontsize=FS_HEAD, color=S.INK,
                     fontweight='bold', fontfamily=HEAD_FONT)
    axA.annotate(f'({gn})', xy=(1, 0.5), xycoords=_name,
                 xytext=(HEAD_GAP_PT, 0), textcoords='offset points',
                 ha='left', va='center', fontsize=FS_HEAD, color=S.MUTED)

axA.set_xlim(0.18, 1.02)
axA.set_ylim(Y_LO, Y_HI)
axA.set_yticks([])
axA.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axA.set_xlabel(PANEL_A_XLABEL)
S.forest_axes(axA)
S.reference_line(axA, 0.5, 'chance')
S.panel_title(axA, 'A', 'Voice vs. demographic comparator')

# Key for the open symbols, under panel A's axis label and flush with the left
# edge of the tinted bands. Placed after a draw, so it sits below the label's
# measured extent rather than at a guessed offset.
fig.canvas.draw()
_xl = axA.xaxis.label.get_window_extent(fig.canvas.get_renderer())
_xl_bottom = axA.transAxes.inverted().transform((0, _xl.y0))[1]
_key = mlines.Line2D([], [], color=S.GREY, lw=2.2, alpha=0.9, marker='o', ms=5.4,
                     markerfacecolor='white', markeredgecolor=S.GREY,
                     markeredgewidth=1.3)
axA.legend([_key], [OVERLAP_NOTE], loc='upper left', frameon=False,
           bbox_to_anchor=(BAND_L, _xl_bottom - 0.04 / PLOT_H),
           bbox_transform=axA.transAxes, borderaxespad=0, borderpad=0,
           handlelength=1.8, handletextpad=0.6, fontsize=FS_FRAC,
           labelcolor=S.MUTED)

# ---------------------------------------------------------------- panel B
axB.set_xlim(-0.32, 0.66)
axB.set_ylim(Y_LO, Y_HI)

_bx_lo, _bx_hi = axB.get_xlim()
_per_inch = (_bx_hi - _bx_lo) / AXB_W


ANN_PAD = 1.6                      # bbox padding of the annotation backing, pt


def centred_in_panel(x, width_in):
    """Keep an annotation centred on its point, but inside the panel.

    The backing box is padded, so the padding counts towards the width that
    has to fit -- otherwise the tint bleeds past the panel edge.
    """
    half = (width_in + 2 * ANN_PAD / 72) * _per_inch / 2
    return min(max(x, _bx_lo + half), _bx_hi - half)


for y_ref, group in zip(ref_y, GROUPS):
    gname, _gn, _entries, dp, dlo, dhi, frac = group
    S.interval(axB, dlo, dhi, dp, y_ref, S.NAVY)
    # backing keeps the zero reference line from cutting the numerals
    _bg = dict(facecolor=BAND, edgecolor='none', pad=ANN_PAD)
    _val, _frc = fmt_diff(dp, dlo, dhi), fmt_frac(frac)
    # Both lines sit on an opaque backing, so they are separated by the full
    # height of the lower box -- its text plus padding above and below -- or
    # the upper backing would paint over the tops of the numerals below it.
    _off1 = 0.28
    _off2 = _off1 + (ANN_H + 2 * ANN_PAD / 72 + 0.012) / ROW_U
    # The share-of-resamples line is drawn first and sits lower in z, so any
    # residual overlap can never hide the estimate.
    axB.text(centred_in_panel(dp, text_size(_frc, FS_FRAC)[0]),
             y_ref + _off2, _frc, ha='center', va='bottom', fontsize=FS_FRAC,
             color=S.FAINT, bbox=_bg, zorder=5)
    axB.text(centred_in_panel(dp, text_size(_val, FS_VAL, 'semibold')[0]),
             y_ref + _off1, _val, ha='center', va='bottom', fontsize=FS_VAL,
             color=S.INK, fontweight='semibold', bbox=_bg, zorder=6)

axB.set_yticks([])
axB.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6])
axB.set_xlabel(PANEL_B_XLABEL)
S.forest_axes(axB)
S.reference_line(axB, 0.0, 'no difference')
S.panel_title(axB, 'B', PANEL_B_TITLE)

# Echo every interval the figure draws, at the precision it draws it, so the
# manuscript table can be checked against the figure without reading pixels.
# Raw values carry six places: at four, a bound of 0.215034 reads as a tie.
for _gname, _gn, _entries, _dp, _dlo, _dhi, _frac in GROUPS:
    print(f'{_gname}  ({_gn})')
    for _lab, _pt, _lo, _hi, _role in _entries:
        print(f'    {_lab:<34s} {fmt_est(_pt, _lo, _hi)}'
              f'      (raw {_pt:.6f} [{_lo:.6f}, {_hi:.6f}])')
    print(f'    {"panel B: voice - age":<34s} {fmt_diff(_dp, _dlo, _dhi)}'
          f'   {fmt_frac(_frac)}'
          f'      (raw {_dp:.6f} [{_dlo:.6f}, {_dhi:.6f}], {_frac:.4f})')

print(f'columns (in): labels {LABEL_W:.2f}  panel A {AXA_W:.2f}  '
      f'panel B {AXB_W:.2f}')
fig_dir = S.resolve_fig_dir(sys.argv[1] if len(sys.argv) > 1 else None)
S.save(fig, 'fig2_voice_vs_baseline_forest', fig_dir)
