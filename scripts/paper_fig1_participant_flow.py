"""Figure 1: CONSORT-style participant flow for both cohorts.

Ported from the TikZ version in master_thesis/figures/participant_flow.tex so
that all three paper figures share one typeface and palette. Content is
transcribed verbatim from that source; the only intentional differences are
typographic (no hyphenation inside boxes, aligned rows across the two panels,
and explicit connectors from the trunk to each exclusion box).

The figure is drawn at the manuscript's 6.292-inch insertion width, with
8–9 pt box text. Exclusions are inset within each cohort column to avoid
shrinking four parallel box columns into the page. Labels wrap to measured
widths, and rendered text is checked against its containing box.

All counts are taken from the metadata the training runs actually read,
data/2025-08-13/preprocessed__{norm_on,norm_off}/metadata.csv, not from the
raw data/2025-08-13/metadata.csv. Earlier versions of this figure mixed the
two: its endpoints came from the preprocessed file but its middle steps from
the raw one, so the Charite arm did not add up (189 - 88 = 101, yet the last
box read 99). Preprocessing drops recordings, which costs two COPD
participants their fourth elicitation category, so the raw file yields the
cohort the thesis calls "the stale N=101 cohort".

Verified against the canonical MLflow mirror: 1366 charite_complete runs all
scored 99 participants (77/22) and 527 uk_only runs all scored 902 (99/803).

Usage: paper_fig1_participant_flow.py [figures-dir]
"""
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import paper_fig_style as S

CHARITE = [
    ('Recruited participants', 'N = 225'),
    ('Non-longitudinal participants', 'N = 205'),
    ('All 4 recording categories present', 'N = 184 (all disease groups)'),
    ('COPD vs. control analysis', 'N = 99 (77 COPD, 22 controls)'),
]
CHARITE_EXC = [
    ('Excluded: 20 longitudinal-only', 'COPD patients'),
    ('Excluded: 21 with incomplete', 'recordings'),
    ('Excluded: 85 participants from', 'other disease groups'),
]
# The first COVID-19 Sounds box is the analysis metadata, not everyone the
# project enrolled: building that metadata already removed every non-COPD
# participant with any smoking history, so its controls are never-smokers.
UK = [
    ('In the analysis metadata', 'N = 12,794', '(173 COPD, 12,621 never-smoking controls)'),
    ('Voice-sentence recording available', 'N = 6,822 (128 COPD, 6,694 controls)'),
    ('PSM-matched cohort', 'N = 906 (100 COPD, 806 controls)'),
    ('Analysis sample (QC-filtered)', 'N = 902 (99 COPD, 803 controls)'),
]
UK_EXC = [
    ('5,972 without usable', 'voice recording'),
    ('28 COPD unmatched;', '5,888 controls unmatched'),
    ('4 excluded (short recordings', 'or extraction failures)'),
]

# Draw at the manuscript's insertion width so fonts retain their point sizes.
FIG_W = 6.292
MARGIN, CENTER_GAP = 0.08, 0.26
W = (FIG_W - 2 * MARGIN - CENTER_GAP) / 2
EW = W - 0.48
PAD_X, PAD_Y, LINE_STEP = 0.12, 0.12, 0.17
FS_HEAD, FS_BODY, FS_EXC = 9.0, 8.5, 8.0
S.apply()

fig, ax = plt.subplots(figsize=(FIG_W, 7))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
renderer = fig.canvas.get_renderer()


def wrap_line(text, width, size, weight='normal'):
    lines, current = [], ''
    for word in text.split():
        candidate = f'{current} {word}'.strip()
        probe = ax.text(0, 0, candidate, fontsize=size, fontweight=weight)
        measured = probe.get_window_extent(renderer).width / fig.dpi
        probe.remove()
        if current and measured > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + [current]


def prepare(blocks, width, exclusion=False):
    result = []
    for index, block in enumerate(blocks):
        rows = []
        for i, text in enumerate(block):
            size = FS_EXC if exclusion else FS_HEAD if i == 0 else FS_BODY
            weight = 'semibold' if not exclusion and i == 0 and index == 3 else 'normal'
            for line in wrap_line(text, width - 2 * PAD_X, size, weight):
                rows.append((line, size, weight, S.MUTED if exclusion or i else S.INK))
        result.append(rows)
    return result


ch, uk = prepare(CHARITE, W), prepare(UK, W)
ch_exc, uk_exc = prepare(CHARITE_EXC, EW, True), prepare(UK_EXC, EW, True)
# Reserve a full line height plus padding beyond the first/last line centres.
H = max(map(len, ch + uk)) * LINE_STEP + 2 * PAD_Y
EH = max(map(len, ch_exc + uk_exc)) * LINE_STEP + 2 * PAD_Y
ROW_GAP = EH + 0.22
FIG_H = 0.40 + 4 * H + 3 * ROW_GAP + MARGIN
fig.set_size_inches(FIG_W, FIG_H)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
ROW_Y = [FIG_H - 0.40 - H / 2 - i * (H + ROW_GAP) for i in range(4)]
boxes = []


def box(left, y, width, height, rows, terminal=False, exclusion=False):
    patch = FancyBboxPatch(
        (left, y - height / 2), width, height,
        boxstyle='round,pad=0,rounding_size=0.045',
        linewidth=0.8,
        facecolor='#F7F7F7' if exclusion else S.HILITE if terminal else 'white',
        edgecolor='#BFBFBF' if exclusion else S.HILITE_EDGE if terminal else '#9C9C9C',
        linestyle=(0, (2.6, 2.0)) if exclusion else '-', zorder=2)
    ax.add_patch(patch)
    texts = []
    for i, (line, size, weight, color) in enumerate(rows):
        texts.append(ax.text(left + width / 2,
                            y + (len(rows) - 1) * LINE_STEP / 2 - i * LINE_STEP,
                            line, ha='center', va='center', fontsize=size,
                            fontweight=weight, color=color, zorder=3))
    boxes.append((patch, texts))


def arrow(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=8,
                                linewidth=0.8, color='#7A7A7A',
                                shrinkA=0, shrinkB=0, zorder=1))


def column(left, steps, exclusions, title):
    ax.text(left + W / 2, FIG_H - 0.18, title, ha='center', va='center',
            fontsize=10, color=S.INK, fontweight='semibold')
    for i, rows in enumerate(steps):
        box(left, ROW_Y[i], W, H, rows, terminal=i == 3)
    # Route the flow down the left margin, beside the inset exclusion boxes.
    trunk = left + 0.16
    for i, rows in enumerate(exclusions):
        top, bottom = ROW_Y[i] - H / 2, ROW_Y[i + 1] + H / 2
        middle = (top + bottom) / 2
        arrow((trunk, top), (trunk, bottom))
        exc_left = left + W - EW
        arrow((trunk, middle), (exc_left, middle))
        box(exc_left, middle, EW, EH, rows, exclusion=True)


column(MARGIN, ch, ch_exc, 'Charité cohort')
column(MARGIN + W + CENTER_GAP, uk, uk_exc, 'COVID-19 Sounds cohort')
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for patch, texts in boxes:
    bounds = patch.get_window_extent(renderer)
    for text in texts:
        extent = text.get_window_extent(renderer)
        assert bounds.x0 < extent.x0 < extent.x1 < bounds.x1
        assert bounds.y0 < extent.y0 < extent.y1 < bounds.y1

fig_dir = S.resolve_fig_dir(sys.argv[1] if len(sys.argv) > 1 else None)
fig_dir.mkdir(parents=True, exist_ok=True)
# Preserve the exact page width; tight cropping would change effective font sizes.
for suffix in ('pdf', 'png'):
    fig.savefig(fig_dir / f'fig1_participant_flow.{suffix}', dpi=600)
print(f'Saved figure at {FIG_W:.3f} × {FIG_H:.2f} inches; minimum font 8 pt.')
