"""Shared visual style for the three figures of the comparator-audit paper.

One typeface, one palette, one set of drawing primitives, so fig1-fig3 read as
a single family rather than three unrelated plots. Used by the sibling
generators in this directory:

    import paper_fig_style as S
    S.apply()
    fig_dir = S.resolve_fig_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    ...
    S.save(fig, 'fig2_voice_vs_baseline_forest', fig_dir)

The figures are consumed by sensors_comparator_audit.md in the neighbouring
copd-voice-comparator-audit checkout, which embeds the PNGs.
"""
import os
import struct
import sys
from pathlib import Path

import matplotlib as mpl
from decimal import Decimal, ROUND_HALF_UP

# ---------------------------------------------------------------------------
# Output location
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIG_DIR = _REPO_ROOT.parent / 'copd-voice-comparator-audit' / 'figures'
ENV_VAR = 'COPDVD_AUDIT_FIGURES'


def resolve_fig_dir(explicit=None):
    """Figures dir: explicit arg, then $COPDVD_AUDIT_FIGURES, then the
    audit checkout next to this one.

    The default is guarded: creating it blind would silently write figures
    into a phantom tree if the audit checkout moved, and the paper would keep
    embedding the stale copies. An explicitly passed path is trusted as-is.
    """
    if explicit:
        return Path(explicit)
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR])
    if not DEFAULT_FIG_DIR.parent.is_dir():
        sys.exit(
            f'audit repo not found at {DEFAULT_FIG_DIR.parent}\n'
            f'pass a figures dir explicitly, or set {ENV_VAR}.'
        )
    return DEFAULT_FIG_DIR


SAVE_DPI = 600
SAVE_PAD = 0.1          # savefig(bbox_inches='tight') pad, applied on each side


def png_size_inches(path, dpi=SAVE_DPI):
    """Width/height of a PNG in inches, read back from its IHDR header.

    The saved size is what the journal actually gets, and with a tight bbox it
    is not knowable up front -- so it is measured from the file rather than
    assumed from figsize.
    """
    with open(path, 'rb') as fh:
        head = fh.read(33)
    w, h = struct.unpack('>II', head[16:24])
    return w / dpi, h / dpi


def save(fig, stem, fig_dir):
    """Write <stem>.pdf (vector) and <stem>.png (600 dpi, embedded by the md).

    Reports the fitted size of what was written, so a figure meant to fill a
    given text width can be checked without opening it.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f'{stem}.pdf', bbox_inches='tight')
    png = fig_dir / f'{stem}.png'
    fig.savefig(png, dpi=SAVE_DPI, bbox_inches='tight')
    w, h = png_size_inches(png)
    print(f'saved {fig_dir}/{stem}.{{pdf,png}}')
    print(f'  fitted {w:.2f} x {h:.2f} in  ({w * SAVE_DPI:.0f} x {h * SAVE_DPI:.0f} px @ {SAVE_DPI} dpi)')


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Restrained and colourblind-safe: one blue, one teal, one amber, plus greys.
# Hue carries meaning consistently across all three figures --
#   NAVY = voice, TEAL = demographic comparator, AMBER = the published model,
#   GREY = a quantity that is not interpretable on its own.

INK = '#1B1B1B'      # primary text
MUTED = '#5F5F5F'    # secondary text, axis labels
FAINT = '#9A9A9A'    # de-emphasised annotation
RULE = '#E3E3E3'     # gridlines, hairlines
NAVY = '#22537F'
TEAL = '#3D8C87'
AMBER = '#C0762F'
SAND = '#DFA469'     # lighter amber, for a second variant of the same model
GREY = '#ABABAB'
HILITE = '#EDF2F8'   # fill for the terminal box of a flow diagram
HILITE_EDGE = '#9DB4CE'

FONT_STACK = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']


def apply():
    """Install the shared rcParams. Call once, before building a figure."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': FONT_STACK,
        'font.size': 8.5,
        'axes.labelsize': 9,
        'axes.titlesize': 9.5,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'text.color': INK,
        'axes.labelcolor': MUTED,
        'axes.edgecolor': RULE,
        'xtick.color': MUTED,
        'ytick.color': INK,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3.0,
        'ytick.major.size': 0.0,
        'lines.linewidth': 1.2,
        'lines.solid_capstyle': 'round',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'pdf.fonttype': 42,      # embed as TrueType, not Type 3
        'ps.fonttype': 42,
    })


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

# Journal text width. Figures are embedded at this size, so building them at
# it means the rendered type is the size set in `apply()` rather than whatever
# survives the manuscript's downscaling.
TEXT_WIDTH = 6.3        # inches


def fit_to_width(fig, target=None, tol=0.01, max_iter=6):
    """Shrink/grow the canvas until the *saved* figure is `target` inches wide.

    For a figure whose column widths are not solved up front. bbox_inches
    crops to the drawn content, which includes anything deliberately placed
    outside the axes, so the saved file comes out wider than figsize and the
    manuscript then scales it down, shrinking every point size with it. Text
    does not scale with the canvas, so shrinking figsize raises the text's
    share of the image; this converges in two or three passes.

    Call it before `save`, which reports what was actually written.
    """
    # savefig pads the tight bbox on every side, so the file comes out wider
    # than the bbox this loop measures. SAVE_PAD is that pad; a figure fitted
    # against a different value misses the target by twice the difference.
    target = (target or TEXT_WIDTH) - 2 * SAVE_PAD
    for _ in range(max_iter):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        if abs(bb.width - target) <= tol:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * (target / bb.width), h, forward=True)
    fig.canvas.draw()
    return fig.get_tightbbox(fig.canvas.get_renderer()).width


def fmt(x, places=2, signed=False):
    """Round half-up to `places` decimals.

    Feed it unrounded values. An input already rounded to three or four
    decimals can land exactly on .xx5 and then round the wrong way: Figure 2's
    bootstrap bound of 0.41473 was stored as 0.415 and printed as 0.42. With
    unrounded input an exact tie does not arise, and half-up only settles what
    one would do. Decimal(str(...)) takes the shortest decimal that
    round-trips the float, so a literal like 0.215 is treated as 0.215 rather
    than as the binary double just below it.

    `signed` prefixes a plus, for the paired differences of Figure 2. A tie at
    zero is normalised so no value prints as '-0.00'.
    """
    q = Decimal(str(float(x))).quantize(Decimal(1).scaleb(-places),
                                        rounding=ROUND_HALF_UP)
    if q == 0:
        q = abs(q)
    s = f'{q:.{places}f}'
    return f'+{s}' if signed and not s.startswith('-') else s


def forest_axes(ax, xgrid=True):
    """Common chrome for an interval plot: x-rule only, faint vertical grid."""
    ax.spines['bottom'].set_color(RULE)
    if xgrid:
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=RULE, linewidth=0.6, zorder=0)
    ax.yaxis.grid(False)


def interval(ax, x_lo, x_hi, x_mid, y, color, lw=2.2, ms=5.4, zorder=3,
             hollow=False):
    """One forest row: a rounded interval bar with a white-ringed point.

    `hollow` draws the point as an open symbol (white face, coloured ring),
    for a row that is shown as a contrast rather than as a result.
    """
    ax.plot([x_lo, x_hi], [y, y], color=color, lw=lw,
            solid_capstyle='round', alpha=0.9, zorder=zorder)
    if hollow:
        ax.plot([x_mid], [y], 'o', ms=ms, zorder=zorder + 1,
                markerfacecolor='white', markeredgecolor=color,
                markeredgewidth=1.3)
    else:
        ax.plot([x_mid], [y], 'o', color=color, ms=ms, zorder=zorder + 1,
                markeredgecolor='white', markeredgewidth=1.1)


def value_column(ax, y, text, x_axes=1.02, color=None, weight='normal',
                 size=7.8):
    """Numeric label in a fixed column at the right of the axes.

    Anchoring to axes coordinates keeps the numbers aligned in a straight
    column instead of ragged-right after variable-length bars.
    """
    ax.text(x_axes, y, text, transform=ax.get_yaxis_transform(),
            va='center', ha='left', fontsize=size, color=color or INK,
            fontweight=weight, clip_on=False)


def reference_line(ax, x, label=None, color=None, ls=(0, (1.2, 2.0)),
                   lw=0.9, label_size=7.0):
    """Vertical reference line, with its caption riding just under the top."""
    color = color or FAINT
    ax.axvline(x, ls=ls, color=color, lw=lw, zorder=1)
    if label:
        ax.text(x, 1.005, label, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=label_size, color=color)


def panel_title(ax, letter, text, pad=14):
    """Left-aligned '(A) Title', consistent across panels and figures."""
    ax.set_title(f'({letter})  {text}', loc='left', pad=pad,
                 fontsize=9.5, color=INK, fontweight='semibold')
