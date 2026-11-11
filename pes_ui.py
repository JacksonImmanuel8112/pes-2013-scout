"""
pes_ui.py - HTML/CSS building blocks (player cards, pitch, badges).
No Streamlit import in here, so these can be tested on their own.

NOTE: every HTML string is built on a single line with no blank lines or
indentation, because Streamlit's markdown renderer ends an HTML block at the
first blank line.
"""

from __future__ import annotations

import html

BLUE = "#3987E5"      # series 1 / main accent
ORANGE = "#D95926"    # series 2
MUTED = "#6B7683"     # de-emphasised bars
TEXT_MUTED = "#9AA5B1"

TIERS = [(88, "elite"), (82, "gold"), (76, "silver"), (0, "bronze")]


def tier(rating) -> str:
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return "bronze"
    for cutoff, name in TIERS:
        if r >= cutoff:
            return name
    return "bronze"


_CSS = """
<style>
.pes-grid{display:flex;flex-wrap:wrap;gap:10px;margin:4px 0 14px}
.pes-card{position:relative;width:150px;box-sizing:border-box;border-radius:14px;padding:10px 12px 11px;color:#0d141b;box-shadow:0 2px 8px rgba(0,0,0,.28);font-family:inherit}
.pes-card.big{width:230px;padding:14px 16px 16px}
.pes-card .top{display:flex;justify-content:space-between;align-items:baseline}
.pes-card .ovr{font-size:30px;font-weight:800;line-height:1}
.pes-card.big .ovr{font-size:46px}
.pes-card .pos{font-size:12px;font-weight:800;letter-spacing:.06em}
.pes-card .nm{font-weight:800;font-size:14px;margin-top:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pes-card.big .nm{font-size:18px}
.pes-card .meta{font-size:11px;opacity:.8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px}
.pes-card .star{position:absolute;top:-7px;right:-5px;background:#0d141b;color:#ffd35c;border-radius:10px;font-size:11px;padding:1px 6px;font-weight:700}
.tier-elite{background:linear-gradient(160deg,#a6ecd9,#41b9a1)}
.tier-gold{background:linear-gradient(160deg,#f6d983,#d6a42b)}
.tier-silver{background:linear-gradient(160deg,#e6ebf0,#a9b3bf)}
.tier-bronze{background:linear-gradient(160deg,#e0b793,#b27b4f)}
.pitch{position:relative;width:100%;max-width:640px;margin:0 auto;aspect-ratio:68/96;container-type:inline-size;border-radius:14px;border:2px solid rgba(255,255,255,.75);background:repeating-linear-gradient(to top,#2e8b57 0,#2e8b57 10%,#287f4e 10%,#287f4e 20%);overflow:hidden}
.pitch .ln{position:absolute;border:2px solid rgba(255,255,255,.55);box-sizing:border-box}
.pitch .half{left:0;right:0;top:50%;height:0;border-width:2px 0 0 0}
.pitch .circle{left:50%;top:50%;width:26%;aspect-ratio:1;transform:translate(-50%,-50%);border-radius:50%}
.pitch .pb{left:22%;width:56%;height:15%}
.pitch .pb.lo{bottom:0;border-bottom:0}
.pitch .pb.hi{top:0;border-top:0}
.pitch .sb{left:36%;width:28%;height:6%}
.pitch .sb.lo{bottom:0;border-bottom:0}
.pitch .sb.hi{top:0;border-top:0}
.tok{position:absolute;transform:translate(-50%,50%);width:96px;width:18cqw;min-width:58px;text-align:center}
.tok .bd{width:44px;width:9.5cqw;min-width:30px;aspect-ratio:1;margin:0 auto;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:15px;font-size:clamp(10px,3.3cqw,20px);color:#0d141b;border:2px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,.35)}
.tok.oop .bd{border-color:#ff9d5c;border-style:dashed}
.tok .nm{margin-top:3px;background:rgba(8,14,20,.80);color:#fff;border-radius:6px;padding:1px 4px;font-size:11px;font-size:clamp(8px,2cqw,12px);font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tok .sl{font-size:9px;font-size:clamp(7px,1.7cqw,10px);color:#fff;margin-top:1px;text-shadow:0 1px 3px rgba(0,0,0,.8);font-weight:600}
.pes-badges{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px}
.pes-badge{border:1px solid rgba(120,135,150,.55);border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600}
.pes-legend{font-size:12px;opacity:.75;margin-top:8px;text-align:center}
</style>
"""


def css() -> str:
    """The stylesheet, flattened so Streamlit keeps it as one HTML block."""
    return "\n".join(line for line in _CSS.splitlines() if line.strip())


def _e(value) -> str:
    return html.escape("" if value is None else str(value))


def player_card(name, rating, pos, club="", age=None, nationality="",
                big=False, starter=False) -> str:
    """A PES-style player card."""
    meta_bits = []
    if age not in (None, "") and age == age:  # not NaN
        meta_bits.append(f"Age {int(age)}")
    if nationality:
        meta_bits.append(_e(nationality))
    meta = " · ".join(meta_bits)
    t = tier(rating)
    cls = "pes-card big" if big else "pes-card"
    star = '<div class="star">XI</div>' if starter else ""
    return (
        f'<div class="{cls} tier-{t}">{star}'
        f'<div class="top"><div class="ovr">{float(rating):.0f}</div><div class="pos">{_e(pos)}</div></div>'
        f'<div class="nm" title="{_e(name)}">{_e(name)}</div>'
        f'<div class="meta">{_e(club)}</div>'
        f'<div class="meta">{meta}</div>'
        f'</div>'
    )


def cards_grid(cards: list[str]) -> str:
    return '<div class="pes-grid">' + "".join(cards) + "</div>"


def pitch_html(lineup) -> str:
    """Football pitch with a token for each starter. `lineup` is the DataFrame from build_lineup()."""
    parts = [
        '<div class="pitch">',
        '<div class="ln half"></div><div class="ln circle"></div>',
        '<div class="ln pb lo"></div><div class="ln sb lo"></div>',
        '<div class="ln pb hi"></div><div class="ln sb hi"></div>',
    ]
    for _, r in lineup.iterrows():
        rating = r["Rating"]
        if rating != rating or rating is None:          # vacant slot
            badge, t = "–", "bronze"
        else:
            badge, t = f"{float(rating):.0f}", tier(rating)
        oop = "" if r["Natural"] or r["Player"] == "— vacant —" else " oop"
        parts.append(
            f'<div class="tok{oop}" style="left:{r["X"]}%;bottom:{r["Y"]}%">'
            f'<div class="bd tier-{t}">{badge}</div>'
            f'<div class="nm" title="{_e(r["Player"])}">{_e(r["Player"])}</div>'
            f'<div class="sl">{_e(r["Slot"])}</div>'
            f'</div>'
        )
    parts.append("</div>")
    parts.append('<div class="pes-legend">Dashed ring = playing out of position</div>')
    return "".join(parts)


def badges(names: list[str]) -> str:
    if not names:
        return '<div class="pes-legend" style="text-align:left">No special abilities</div>'
    return '<div class="pes-badges">' + "".join(
        f'<span class="pes-badge">{_e(n)}</span>' for n in names) + "</div>"
