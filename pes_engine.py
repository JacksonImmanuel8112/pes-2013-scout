"""
pes_engine.py - all the PES 2013 data logic (no Streamlit in here).

Why this file exists
--------------------
In players.xlsx the 12 position columns (GK, SW, CB ... CF) are NOT ratings.
They are 0/1 flags meaning "this player can play this position".
So we calculate our own position ratings from the 0-99 attributes
(Attack, Defense, Top Speed, Dribble Accuracy ...) using the weights below.
These formulas are OUR design, not something stored in the game - tweak the
WEIGHTS dictionary freely and the whole app updates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------

POSITIONS = ["GK", "SW", "CB", "SB", "DMF", "WB",
             "CMF", "SMF", "AMF", "WF", "SS", "CF"]

POSITION_NAMES = {
    "GK": "Goalkeeper", "SW": "Sweeper", "CB": "Centre Back",
    "SB": "Side Back", "DMF": "Defensive Midfielder", "WB": "Wing Back",
    "CMF": "Central Midfielder", "SMF": "Side Midfielder",
    "AMF": "Attacking Midfielder", "WF": "Wing Forward",
    "SS": "Second Striker", "CF": "Centre Forward",
}

POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}
GK_COL = POS_INDEX["GK"]

# Which tactical line a position belongs to
LINE_OF = {
    "GK": "GK",
    "SW": "DEF", "CB": "DEF", "SB": "DEF", "WB": "DEF",
    "DMF": "MID", "CMF": "MID", "SMF": "MID", "AMF": "MID",
    "WF": "ATT", "SS": "ATT", "CF": "ATT",
}

# 23 special-ability flags (0/1) in the spreadsheet
ABILITY_COLUMNS = [
    "Dribbling", "Tactical Dribble", "Positioning", "Reaction",
    "Playmaking", "Passing", "Scoring", "1-1 Scoring", "Post Player",
    "Line Position", "Middle Shooting", "Side", "Centre", "Penalties",
    "1-Touch Passing", "Outside Curve", "Man Marking", "Sliding",
    "Covering", "D-Line Control", "Penalty GK", "1-on-1 GK", "Long Throw",
]

# The spreadsheet's "Top Speed" column header is stored as "ac"
COLUMN_FIXES = {"ac": "Top Speed"}

# Position rating formulas: weights per attribute (each row sums to 1.0)
WEIGHTS: dict[str, dict[str, float]] = {
    "GK": {"Goalkeeper Skills": .45, "Defense": .20, "Response": .10,
           "Jump": .08, "Agility": .07, "Mentality": .05, "Body Balance": .05},
    "CB": {"Defense": .32, "Heading": .12, "Jump": .08, "Body Balance": .10,
           "Aggression": .08, "Mentality": .06, "Response": .06,
           "Team Work": .04, "Stamina": .04, "Top Speed": .04,
           "Short Pass Accuracy": .03, "Long Pass Accuracy": .03},
    "SW": {"Defense": .30, "Mentality": .10, "Response": .08,
           "Long Pass Accuracy": .08, "Short Pass Accuracy": .06,
           "Heading": .08, "Body Balance": .08, "Jump": .05,
           "Team Work": .05, "Aggression": .05, "Technique": .03,
           "Stamina": .04},
    "SB": {"Defense": .22, "Stamina": .12, "Top Speed": .10,
           "Acceleration": .08, "Long Pass Accuracy": .06,
           "Short Pass Accuracy": .06, "Dribble Speed": .05, "Agility": .05,
           "Response": .05, "Team Work": .05, "Attack": .06,
           "Body Balance": .05, "Aggression": .05},
    "WB": {"Attack": .12, "Defense": .14, "Stamina": .14, "Top Speed": .11,
           "Acceleration": .09, "Dribble Accuracy": .06, "Dribble Speed": .05,
           "Long Pass Accuracy": .07, "Short Pass Accuracy": .05,
           "Agility": .05, "Response": .04, "Team Work": .04,
           "Body Balance": .04},
    "DMF": {"Defense": .24, "Short Pass Accuracy": .09,
            "Long Pass Accuracy": .07, "Stamina": .10, "Body Balance": .08,
            "Aggression": .07, "Mentality": .06, "Team Work": .08,
            "Response": .06, "Heading": .05, "Short Pass Speed": .04,
            "Attack": .06},
    "CMF": {"Attack": .12, "Defense": .12, "Short Pass Accuracy": .12,
            "Long Pass Accuracy": .08, "Short Pass Speed": .05,
            "Technique": .08, "Stamina": .10, "Team Work": .08,
            "Dribble Accuracy": .05, "Mentality": .05, "Response": .05,
            "Body Balance": .05, "Agility": .05},
    "SMF": {"Attack": .15, "Top Speed": .10, "Acceleration": .09,
            "Dribble Accuracy": .10, "Dribble Speed": .07,
            "Short Pass Accuracy": .07, "Long Pass Accuracy": .08,
            "Stamina": .10, "Agility": .06, "Technique": .06,
            "Defense": .05, "Team Work": .04, "Response": .03},
    "AMF": {"Attack": .22, "Technique": .10, "Dribble Accuracy": .09,
            "Short Pass Accuracy": .10, "Short Pass Speed": .05,
            "Long Pass Accuracy": .05, "Shot Accuracy": .06,
            "Shot Technique": .04, "Agility": .06, "Acceleration": .05,
            "Response": .06, "Mentality": .04, "Team Work": .04,
            "Dribble Speed": .04},
    "WF": {"Attack": .22, "Top Speed": .11, "Acceleration": .11,
           "Dribble Accuracy": .11, "Dribble Speed": .08, "Agility": .06,
           "Technique": .06, "Shot Accuracy": .05,
           "Short Pass Accuracy": .04, "Long Pass Accuracy": .04,
           "Response": .05, "Stamina": .04, "Shot Power": .03},
    "SS": {"Attack": .25, "Shot Accuracy": .09, "Shot Technique": .06,
           "Dribble Accuracy": .09, "Technique": .08, "Acceleration": .08,
           "Agility": .06, "Response": .07, "Short Pass Accuracy": .05,
           "Top Speed": .06, "Shot Power": .04, "Dribble Speed": .04,
           "Mentality": .03},
    "CF": {"Attack": .25, "Shot Accuracy": .12, "Shot Power": .08,
           "Shot Technique": .07, "Heading": .07, "Body Balance": .07,
           "Response": .07, "Acceleration": .06, "Top Speed": .05,
           "Jump": .04, "Dribble Accuracy": .04, "Technique": .04,
           "Mentality": .04},
}

for _p, _w in WEIGHTS.items():
    assert abs(sum(_w.values()) - 1.0) < 1e-9, f"Weights for {_p} must sum to 1"

# Grouped attributes for the radar charts (8 axes)
RADAR_GROUPS = {
    "Attack": ["Attack"],
    "Defence": ["Defense"],
    "Passing": ["Short Pass Accuracy", "Short Pass Speed",
                "Long Pass Accuracy", "Long Pass Speed"],
    "Pace": ["Top Speed", "Acceleration", "Agility"],
    "Technique": ["Dribble Accuracy", "Dribble Speed", "Technique"],
    "Shooting": ["Shot Accuracy", "Shot Power", "Shot Technique"],
    "Physical": ["Stamina", "Body Balance", "Jump", "Aggression"],
    "Mind": ["Mentality", "Team Work", "Response"],
}

# Formation templates: (slot label, PES position, x 0-100, y 0-100)
# y = 0 is our own goal line, y = 100 is the opponent's goal line.
FORMATIONS: dict[str, list[tuple[str, str, int, int]]] = {
    "4-4-2": [
        ("GK", "GK", 50, 6),
        ("LB", "SB", 14, 24), ("LCB", "CB", 38, 22),
        ("RCB", "CB", 62, 22), ("RB", "SB", 86, 24),
        ("LM", "SMF", 14, 52), ("LCM", "CMF", 38, 48),
        ("RCM", "CMF", 62, 48), ("RM", "SMF", 86, 52),
        ("LCF", "CF", 38, 82), ("RCF", "CF", 62, 82),
    ],
    "4-3-3": [
        ("GK", "GK", 50, 6),
        ("LB", "SB", 14, 24), ("LCB", "CB", 38, 22),
        ("RCB", "CB", 62, 22), ("RB", "SB", 86, 24),
        ("DMF", "DMF", 50, 40),
        ("LCM", "CMF", 30, 54), ("RCM", "CMF", 70, 54),
        ("LW", "WF", 16, 80), ("CF", "CF", 50, 86), ("RW", "WF", 84, 80),
    ],
    "4-2-3-1": [
        ("GK", "GK", 50, 6),
        ("LB", "SB", 14, 24), ("LCB", "CB", 38, 22),
        ("RCB", "CB", 62, 22), ("RB", "SB", 86, 24),
        ("LDM", "DMF", 36, 40), ("RDM", "DMF", 64, 40),
        ("LW", "WF", 16, 66), ("AMF", "AMF", 50, 64), ("RW", "WF", 84, 66),
        ("CF", "CF", 50, 86),
    ],
    "4-1-4-1": [
        ("GK", "GK", 50, 6),
        ("LB", "SB", 14, 24), ("LCB", "CB", 38, 22),
        ("RCB", "CB", 62, 22), ("RB", "SB", 86, 24),
        ("DMF", "DMF", 50, 38),
        ("LM", "SMF", 14, 58), ("LCM", "CMF", 38, 54),
        ("RCM", "CMF", 62, 54), ("RM", "SMF", 86, 58),
        ("CF", "CF", 50, 86),
    ],
    "4-3-1-2": [
        ("GK", "GK", 50, 6),
        ("LB", "SB", 14, 24), ("LCB", "CB", 38, 22),
        ("RCB", "CB", 62, 22), ("RB", "SB", 86, 24),
        ("DMF", "DMF", 50, 40),
        ("LCM", "CMF", 28, 52), ("RCM", "CMF", 72, 52),
        ("AMF", "AMF", 50, 68),
        ("LCF", "CF", 38, 86), ("RCF", "CF", 62, 86),
    ],
    "3-5-2": [
        ("GK", "GK", 50, 6),
        ("LCB", "CB", 26, 22), ("CB", "CB", 50, 20), ("RCB", "CB", 74, 22),
        ("LWB", "WB", 8, 50), ("RWB", "WB", 92, 50),
        ("DMF", "DMF", 50, 38),
        ("LCM", "CMF", 32, 54), ("RCM", "CMF", 68, 54),
        ("LCF", "CF", 38, 84), ("RCF", "CF", 62, 84),
    ],
    "3-4-3": [
        ("GK", "GK", 50, 6),
        ("LCB", "CB", 26, 22), ("CB", "CB", 50, 20), ("RCB", "CB", 74, 22),
        ("LM", "SMF", 12, 48), ("LCM", "CMF", 38, 46),
        ("RCM", "CMF", 62, 46), ("RM", "SMF", 88, 48),
        ("LW", "WF", 18, 80), ("CF", "CF", 50, 86), ("RW", "WF", 82, 80),
    ],
    "5-3-2": [
        ("GK", "GK", 50, 6),
        ("LWB", "WB", 8, 34), ("LCB", "CB", 28, 22), ("CB", "CB", 50, 20),
        ("RCB", "CB", 72, 22), ("RWB", "WB", 92, 34),
        ("LCM", "CMF", 30, 52), ("DMF", "DMF", 50, 46),
        ("RCM", "CMF", 70, 52),
        ("LCF", "CF", 38, 84), ("RCF", "CF", 62, 84),
    ],
}

NO_CLUB = "No Club / Free Agent"
NEG = -1e6          # "not allowed" marker inside rating matrices
OOP_PENALTY = 10    # default points lost when playing out of position


# ----------------------------------------------------------------------
# LOADING & RATINGS
# ----------------------------------------------------------------------

def load_players(source) -> pd.DataFrame:
    """Read the spreadsheet and add: Player ID, Label, R_<pos> ratings, Overall."""
    df = pd.read_excel(source)
    df.columns = df.columns.astype(str).str.strip()
    df = df.rename(columns=COLUMN_FIXES)

    if "Player Name" not in df.columns:
        raise ValueError("The spreadsheet has no 'Player Name' column.")

    text_cols = ["Player Name", "Nationality", "European Club",
                 "Registered Position", "Dominant Leg", "Dominant Side",
                 "Injury Tolerance"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype("object").where(df[c].notna(), None)

    numeric = [c for c in df.columns if c not in text_cols]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Player Name"] = df["Player Name"].fillna("Unknown").astype(str).str.strip()
    df["Club"] = df["European Club"].fillna(NO_CLUB).astype(str)
    df["Nationality"] = df["Nationality"].fillna("Unknown").astype(str)
    df["Registered Position"] = df["Registered Position"].fillna("CMF").astype(str)

    df["Player ID"] = np.arange(len(df))

    # Position flags must be 0/1 integers
    for p in POSITIONS:
        df[p] = df[p].fillna(0).gt(0).astype(int)

    # Make sure the registered position is always playable
    for p in POSITIONS:
        df.loc[df["Registered Position"] == p, p] = 1

    df = add_ratings(df)

    # e.g. "CB/SB/DMF" - every position the player can play
    flags = df[POSITIONS].to_numpy() > 0
    df["Plays"] = ["/".join(p for p, ok in zip(POSITIONS, row) if ok) for row in flags]

    # Unique, human readable label for dropdowns
    label = (df["Player Name"] + "  ·  " + df["Club"] + "  ·  "
             + df["Age"].fillna(0).astype(int).astype(str) + "  ·  "
             + df["Registered Position"] + " " + df["Overall"].round(0).astype(int).astype(str))
    dup = label.duplicated(keep=False)
    label[dup] = label[dup] + "  #" + df.loc[dup, "Player ID"].astype(str)
    df["Label"] = label
    return df


def add_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute R_<pos> (0-99) for every player and 'Overall' (rating at registered position)."""
    df = df.copy()
    for pos, weights in WEIGHTS.items():
        total = np.zeros(len(df))
        for attr, w in weights.items():
            total += w * df[attr].fillna(df[attr].median()).to_numpy(float)
        df[f"R_{pos}"] = total

    r = df[[f"R_{p}" for p in POSITIONS]].to_numpy()
    idx = df["Registered Position"].map(POS_INDEX).fillna(POS_INDEX["CMF"]).astype(int).to_numpy()
    df["Overall"] = r[np.arange(len(df)), idx]
    return df


def effective_matrix(df: pd.DataFrame, oop_penalty: float = OOP_PENALTY,
                     allow_oop: bool = True) -> np.ndarray:
    """
    n_players x 12 matrix: how good would each player be in each position?
      - flag = 1  -> full rating
      - flag = 0  -> rating minus penalty (or not allowed if allow_oop is False)
      - goalkeepers can never play outfield, and outfielders never in goal
    """
    R = df[[f"R_{p}" for p in POSITIONS]].to_numpy(float)
    F = df[POSITIONS].to_numpy() > 0
    E = np.where(F, R, R - oop_penalty if allow_oop else NEG)
    is_gk = F[:, GK_COL]
    E[:, GK_COL] = np.where(is_gk, R[:, GK_COL], NEG)
    E[is_gk, :GK_COL] = NEG
    E[is_gk, GK_COL + 1:] = NEG
    return E


# ----------------------------------------------------------------------
# SQUAD SELECTION
# ----------------------------------------------------------------------

def solve_xi(E: np.ndarray, formation: str):
    """
    Best XI for a formation from a rating matrix E (rows = candidate players).
    Uses the Hungarian algorithm so nobody is picked twice.
    Returns a list (one entry per slot) of (row_index or None, rating or None).
    """
    slots = FORMATIONS[formation]
    cols = [POS_INDEX[s[1]] for s in slots]
    sub = E[:, cols]                      # players x slots
    n, m = sub.shape
    if n < m:                             # pad with impossible dummy players
        sub = np.vstack([sub, np.full((m - n, m), NEG)])
    rows, slot_ids = linear_sum_assignment(sub, maximize=True)
    out = [(None, None)] * m
    for r, s in zip(rows, slot_ids):
        if r < n and sub[r, s] > NEG / 2:
            out[s] = (int(r), float(sub[r, s]))
    return out


def build_lineup(pool: pd.DataFrame, formation: str, oop_penalty: float = OOP_PENALTY,
                 allow_oop: bool = True):
    """Returns (lineup DataFrame, E matrix). Lineup rows follow the formation's slot order."""
    E = effective_matrix(pool, oop_penalty, allow_oop)
    result = solve_xi(E, formation)
    rows = []
    for (label, pos, x, y), (ri, rating) in zip(FORMATIONS[formation], result):
        row = {"Slot": label, "Position": pos, "X": x, "Y": y,
               "_row": ri, "Rating": rating}
        if ri is not None:
            p = pool.iloc[ri]
            row.update({
                "Player ID": int(p["Player ID"]), "Player": p["Player Name"],
                "Age": int(p["Age"]), "Club": p["Club"],
                "Nationality": p["Nationality"],
                "Registered": p["Registered Position"],
                "Natural": bool(p[pos] == 1),
            })
        else:
            row.update({"Player ID": None, "Player": "— vacant —", "Age": None,
                        "Club": "", "Nationality": "", "Registered": "",
                        "Natural": False})
        rows.append(row)
    return pd.DataFrame(rows), E


BENCH_ORDER = ["GK", "CB", "CMF", "CF", "SB", "AMF", "WF", "DMF", "SS", "SMF", "WB", "SW"]


def pick_bench(pool: pd.DataFrame, E: np.ndarray, used_rows: set[int], size: int) -> pd.DataFrame:
    """Greedy bench: a keeper first, then cover every area of the pitch."""
    taken = set(used_rows)
    picks = []
    for i in range(size):
        pos = BENCH_ORDER[i % len(BENCH_ORDER)]
        col = POS_INDEX[pos]
        order = np.argsort(-E[:, col])
        chosen = next((r for r in order if r not in taken and E[r, col] > NEG / 2), None)
        if chosen is None:
            # fall back: best remaining outfield player anywhere
            if pos == "GK":
                continue
            masked = E.max(axis=1).copy()
            masked[list(taken)] = NEG
            chosen = int(np.argmax(masked))
            if masked[chosen] <= NEG / 2:
                continue
            col = int(np.argmax(E[chosen]))
            pos = POSITIONS[col]
        taken.add(int(chosen))
        p = pool.iloc[int(chosen)]
        picks.append({"Cover": pos, "Player": p["Player Name"], "Age": int(p["Age"]),
                      "Registered": p["Registered Position"],
                      "Rating": float(E[chosen, col]), "Club": p["Club"],
                      "Player ID": int(p["Player ID"]), "_row": int(chosen)})
    return pd.DataFrame(picks)


# ----------------------------------------------------------------------
# TACTICAL ANALYSIS
# ----------------------------------------------------------------------

def line_ratings(lineup: pd.DataFrame) -> dict[str, float]:
    """Average rating per line (GK / DEF / MID / ATT) plus overall team rating."""
    lineup = lineup.dropna(subset=["Rating"])
    out = {}
    for line in ["GK", "DEF", "MID", "ATT"]:
        mask = lineup["Position"].map(LINE_OF) == line
        out[line] = float(lineup.loc[mask, "Rating"].mean()) if mask.any() else float("nan")
    out["TEAM"] = float(lineup["Rating"].mean()) if len(lineup) else float("nan")
    return out


def team_profile(pool: pd.DataFrame, lineup: pd.DataFrame) -> dict[str, float]:
    """Radar-chart profile of the XI (outfield players; Defence also counts the keeper)."""
    lu = lineup.dropna(subset=["Player ID"]).copy()
    lu["Line"] = lu["Position"].map(LINE_OF)
    by_id = pool.set_index("Player ID")
    profile = {}
    for axis, cols in RADAR_GROUPS.items():
        if axis == "Defence":          # keeper + defenders + midfielders
            lines = ["GK", "DEF", "MID"]
        elif axis == "Attack":         # midfielders + forwards
            lines = ["MID", "ATT"]
        else:                          # everybody except the keeper
            lines = ["DEF", "MID", "ATT"]
        ids = lu.loc[lu["Line"].isin(lines), "Player ID"].astype(int)
        if ids.empty:
            profile[axis] = float("nan")
        else:
            profile[axis] = float(by_id.loc[ids, cols].mean(axis=1).mean())
    return profile


def formation_fit(pool: pd.DataFrame, oop_penalty: float = OOP_PENALTY,
                  allow_oop: bool = True) -> pd.DataFrame:
    """Team rating of this squad in every formation, best first."""
    E = effective_matrix(pool, oop_penalty, allow_oop)
    rows = []
    for name in FORMATIONS:
        res = solve_xi(E, name)
        ratings = [r for _, r in res if r is not None]
        rows.append({"Formation": name,
                     "Team Rating": float(np.mean(ratings)) if len(ratings) == 11 else float("nan"),
                     "Vacant": 11 - len(ratings)})
    return pd.DataFrame(rows).sort_values("Team Rating", ascending=False).reset_index(drop=True)


def club_power_table(df: pd.DataFrame, formations=("4-4-2", "4-3-3", "4-2-3-1"),
                     oop_penalty: float = OOP_PENALTY, allow_oop: bool = True) -> pd.DataFrame:
    """Best-XI strength of every real club (best of a few formations)."""
    clubbed = df[df["Club"] != NO_CLUB].reset_index(drop=True)
    E_all = effective_matrix(clubbed, oop_penalty, allow_oop)
    rows = []
    for club, idx in clubbed.groupby("Club").indices.items():
        E = E_all[idx]
        best = (None, -1.0, None)
        for f in formations:
            res = solve_xi(E, f)
            vals = [r for _, r in res if r is not None]
            if len(vals) < 11:
                continue
            score = float(np.mean(vals))
            if score > best[1]:
                best = (f, score, res)
        if best[0] is None:
            continue
        rows.append({"Club": club, "Squad Size": len(idx),
                     "Best XI Rating": best[1], "Formation": best[0]})
    out = pd.DataFrame(rows).sort_values("Best XI Rating", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", out.index + 1)
    return out


# ----------------------------------------------------------------------
# TRANSFER TARGETS
# ----------------------------------------------------------------------

def transfer_shortlist(squad: pd.DataFrame, market: pd.DataFrame, formation: str,
                       oop_penalty: float = OOP_PENALTY, allow_oop: bool = True,
                       max_candidates: int = 400) -> pd.DataFrame:
    """
    For each player on the market: how much would signing him improve our best XI?
    gain = (new total of the 11 slot ratings) - (old total) ; reported per-player average too.
    """
    if market.empty:
        return pd.DataFrame()

    slots = FORMATIONS[formation]
    slot_cols = [POS_INDEX[s[1]] for s in slots]

    E_squad = effective_matrix(squad, oop_penalty, allow_oop)
    base = solve_xi(E_squad, formation)
    base_total = sum(r for _, r in base if r is not None)
    base_by_slot = [r if r is not None else 0.0 for _, r in base]
    base_names = [squad.iloc[i]["Player Name"] if i is not None else "— vacant —"
                  for i, _ in base]

    E_mkt = effective_matrix(market, oop_penalty, allow_oop)

    # cheap pre-filter: must beat at least one starter in a formation slot
    best_gap = np.full(len(market), -np.inf)
    for s, col in enumerate(slot_cols):
        best_gap = np.maximum(best_gap, E_mkt[:, col] - base_by_slot[s])
    keep = np.where(best_gap > 0)[0]
    if len(keep) > max_candidates:
        keep = keep[np.argsort(-best_gap[keep])[:max_candidates]]

    rows = []
    for k in keep:
        E_new = np.vstack([E_squad, E_mkt[k:k + 1]])
        res = solve_xi(E_new, formation)
        new_total = sum(r for _, r in res if r is not None)
        gain = new_total - base_total
        if gain <= 0:
            continue
        cand_row = len(E_squad)
        slot_idx = next((s for s, (i, _) in enumerate(res) if i == cand_row), None)
        if slot_idx is None:
            continue
        p = market.iloc[k]
        rows.append({
            "Player ID": int(p["Player ID"]),
            "Player": p["Player Name"], "Age": int(p["Age"]),
            "Club": p["Club"], "Nationality": p["Nationality"],
            "Registered": p["Registered Position"],
            "Plays As": slots[slot_idx][0], "Position": slots[slot_idx][1],
            "Rating": res[slot_idx][1],
            "Replaces": base_names[slot_idx],
            "Current Rating": base_by_slot[slot_idx],
            "XI Gain": gain,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["XI Gain", "Rating"], ascending=False).reset_index(drop=True)


def shopping_list(squad: pd.DataFrame, market: pd.DataFrame, formation: str, signings: int = 3,
                  oop_penalty: float = OOP_PENALTY, allow_oop: bool = True) -> pd.DataFrame:
    """Greedy plan: sign the best upgrade, update the squad, repeat."""
    plan = []
    current = squad
    pool_market = market
    for step in range(1, signings + 1):
        short = transfer_shortlist(current, pool_market, formation, oop_penalty, allow_oop)
        if short.empty:
            break
        top = short.iloc[0].to_dict()
        top["Step"] = step
        plan.append(top)
        pid = top["Player ID"]
        signed = pool_market[pool_market["Player ID"] == pid]
        current = pd.concat([current, signed], ignore_index=True)
        pool_market = pool_market[pool_market["Player ID"] != pid]
    return pd.DataFrame(plan)


# ----------------------------------------------------------------------
# SIMILAR PLAYERS
# ----------------------------------------------------------------------

SIMILARITY_ATTRS = ["Attack", "Defense", "Body Balance", "Stamina", "Top Speed",
                    "Acceleration", "Response", "Agility", "Dribble Accuracy",
                    "Dribble Speed", "Short Pass Accuracy", "Short Pass Speed",
                    "Long Pass Accuracy", "Long Pass Speed", "Shot Accuracy",
                    "Shot Power", "Shot Technique", "Heading", "Jump",
                    "Technique", "Aggression", "Mentality", "Team Work"]


def similar_players(df: pd.DataFrame, player_id: int, n: int = 8) -> pd.DataFrame:
    """Players with the closest attribute profile (Euclidean distance, same keeper/outfield group)."""
    me = df[df["Player ID"] == player_id].iloc[0]
    pool = df[(df["GK"] == me["GK"]) & (df["Player ID"] != player_id)]
    X = pool[SIMILARITY_ATTRS].to_numpy(float)
    v = me[SIMILARITY_ATTRS].to_numpy(float)
    dist = np.sqrt(((X - v) ** 2).sum(axis=1))
    out = pool.assign(Distance=dist).sort_values("Distance").head(n)
    out["Similarity"] = (100 - out["Distance"] / np.sqrt(len(SIMILARITY_ATTRS)) * 2).clip(0, 100)
    return out
