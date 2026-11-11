"""
PES 2013 Master League Scout  -  app.py

Run with:   python -m streamlit run app.py

Files that must sit in the same folder:
    app.py          <- this file (the screens)
    pes_engine.py   <- ratings, squad selection, transfer logic
    pes_ui.py       <- player cards and the pitch
    data/players.xlsx
"""

import inspect
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pes_engine as E
import pes_ui as U

st.set_page_config(
    page_title="PES 2013 Master League Scout",
    page_icon="⚽",
    layout="wide",
)
st.markdown(U.css(), unsafe_allow_html=True)

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "data" / "players.xlsx"

DREAM = "★ All players (Dream Team)"
GRID = "rgba(150,160,175,0.25)"

ATTRIBUTES = [
    "Attack", "Defense", "Body Balance", "Stamina", "Top Speed",
    "Acceleration", "Response", "Agility", "Dribble Accuracy",
    "Dribble Speed", "Short Pass Accuracy", "Short Pass Speed",
    "Long Pass Accuracy", "Long Pass Speed", "Shot Accuracy", "Shot Power",
    "Shot Technique", "Free Kick Accuracy", "Swerve", "Heading", "Jump",
    "Technique", "Aggression", "Mentality", "Goalkeeper Skills", "Team Work",
]


# ======================================================================
# SMALL HELPERS
# ======================================================================

_PLOT_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show_plot(fig):
    """Draw a Plotly chart (works on old and new Streamlit versions)."""
    if _PLOT_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def show_table(data, **kwargs):
    """Draw a DataFrame full-width without the index (old + new Streamlit)."""
    try:
        st.dataframe(data, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(data, use_container_width=True, hide_index=True, **kwargs)


def html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def base_layout(fig, height=None):
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=U.TEXT_MUTED),
        showlegend=False,
        **({"height": height} if height else {}),
    )
    return fig


def lollipop(labels, values, xmin, colors=None, height=None, fmt="{:.1f}"):
    """
    Dot-and-stem chart. Used instead of bars because ratings live between
    ~60 and ~95: bars that start at 60 would exaggerate small differences.
    """
    values = [float(v) for v in values]
    n = len(values)
    colors = colors or [U.BLUE] * n
    top = max(values) if values else xmin + 1
    fig = go.Figure(go.Scatter(
        x=values, y=list(labels), mode="markers+text",
        text=[fmt.format(v) for v in values],
        textposition="middle right",
        textfont=dict(color=U.TEXT_MUTED, size=12),
        marker=dict(size=11, color=colors),
        error_x=dict(type="data", symmetric=False, array=[0] * n,
                     arrayminus=[v - xmin for v in values],
                     width=0, thickness=2, color="rgba(150,160,175,0.45)"),
        hovertemplate="%{y}: %{x:.1f}<extra></extra>",
    ))
    fig.update_xaxes(range=[xmin, top + (top - xmin) * 0.18 + 1],
                     gridcolor=GRID, zeroline=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return base_layout(fig, height or max(260, 28 * n + 60))


def radar(series, rmin=40):
    """series = [(name, {axis: value}, colour), ...]"""
    fig = go.Figure()
    for name, vals, color in series:
        labels, r = list(vals.keys()), list(vals.values())
        fig.add_trace(go.Scatterpolar(
            r=r + r[:1], theta=labels + labels[:1], name=name, fill="toself",
            line=dict(color=color, width=2), fillcolor=rgba(color, 0.25),
            marker=dict(size=5, color=color),
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[rmin, 100], gridcolor=GRID,
                            linecolor=GRID, tickfont=dict(color=U.TEXT_MUTED, size=10)),
            angularaxis=dict(gridcolor=GRID, linecolor=GRID,
                             tickfont=dict(color=U.TEXT_MUTED, size=12)),
        ),
        margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=U.TEXT_MUTED),
        showlegend=len(series) > 1,
        legend=dict(orientation="h", y=-0.1),
        height=430,
    )
    return fig


def idx_of(options, preferred):
    return options.index(preferred) if preferred in options else 0


def card_for(row, rating=None, pos=None, starter=False, big=False):
    return U.player_card(
        row["Player Name"],
        row["Overall"] if rating is None else rating,
        row["Registered Position"] if pos is None else pos,
        row["Club"], row["Age"], row["Nationality"],
        big=big, starter=starter,
    )


def group_profile(row_df: pd.DataFrame) -> dict:
    """8-axis radar profile for one player (row_df is a one-row DataFrame)."""
    return {axis: float(row_df[cols].mean(axis=1).iloc[0])
            for axis, cols in E.RADAR_GROUPS.items()}


# ======================================================================
# DATA LOADING
# ======================================================================

@st.cache_data(show_spinner="Loading the PES database…")
def load_from_path(path: str, modified: float):
    return E.load_players(path)


@st.cache_data(show_spinner="Loading the PES database…")
def load_from_bytes(raw: bytes):
    return E.load_players(io.BytesIO(raw))


@st.cache_data(show_spinner="Ranking every club…")
def cached_power(_df, version, oop_penalty, allow_oop):
    return E.club_power_table(_df, oop_penalty=oop_penalty, allow_oop=allow_oop)


try:
    if DATA_FILE.exists():
        df = load_from_path(str(DATA_FILE), DATA_FILE.stat().st_mtime)
    else:
        st.warning(f"Could not find {DATA_FILE}. Upload your players.xlsx to continue "
                   "(or place it in the data folder).")
        upload = st.file_uploader("Upload players.xlsx", type=["xlsx"])
        if upload is None:
            st.stop()
        df = load_from_bytes(upload.getvalue())
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read the player database: {exc}")
    st.stop()

DATA_VERSION = f"{len(df)}-{df['Overall'].sum():.3f}"
ALL_CLUBS = sorted(c for c in df["Club"].unique() if c != E.NO_CLUB)


# ======================================================================
# SIDEBAR
# ======================================================================

st.sidebar.title("⚽ PES 2013")
st.sidebar.caption("MASTER LEAGUE SCOUT")

PAGES = [
    "🏠 Dashboard",
    "🏟️ Squad Manager",
    "💸 Transfer Targets",
    "👤 Player Profile",
    "⚔️ Player Comparison",
    "🏆 Position Rankings",
    "📋 Player Database",
    "🧮 Rating Method",
]
page = st.sidebar.radio("Menu", PAGES, label_visibility="collapsed")

st.sidebar.divider()
with st.sidebar.expander("⚙️ Rating rules"):
    ALLOW_OOP = st.toggle(
        "Allow out-of-position players", value=True,
        help="If off, a player can only be picked for positions marked as playable in your spreadsheet.")
    OOP_PENALTY = st.slider(
        "Out-of-position penalty", 0, 25, E.OOP_PENALTY,
        disabled=not ALLOW_OOP,
        help="Rating points lost when a player plays a position he is not listed for.")

st.sidebar.metric("Players in database", f"{len(df):,}")
st.sidebar.caption("Built with Python, Streamlit & Plotly")


def power_table():
    return cached_power(df, DATA_VERSION, OOP_PENALTY, ALLOW_OOP)


# ======================================================================
# 1. DASHBOARD
# ======================================================================

def page_dashboard():
    st.title("⚽ PES 2013 Master League Scout")
    st.caption("Overall = a player's rating at his registered position, "
               "calculated from his attributes (see Rating Method).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players", f"{len(df):,}")
    c2.metric("Clubs", len(ALL_CLUBS))
    c3.metric("Nationalities", df["Nationality"].nunique())
    c4.metric("Average Overall", f"{df['Overall'].mean():.1f}")

    st.subheader("Top 10 players")
    top = df.sort_values("Overall", ascending=False).head(10)
    html(U.cards_grid([card_for(r) for _, r in top.iterrows()]))

    power = power_table()
    left, right = st.columns(2)

    with left:
        st.subheader("Club power rankings")
        st.caption("Rating of each club's best XI (best of 4-4-2, 4-3-3, 4-2-3-1).")
        t15 = power.head(15)
        show_plot(lollipop(t15["Club"], t15["Best XI Rating"], xmin=70))

    with right:
        st.subheader("Overall rating distribution")
        fig = go.Figure(go.Histogram(x=df["Overall"], nbinsx=32, marker_color=U.BLUE))
        fig.update_xaxes(title="Overall", gridcolor=GRID)
        fig.update_yaxes(title="Players", gridcolor=GRID)
        show_plot(base_layout(fig, 420))

    left, right = st.columns(2)
    with left:
        st.subheader("Players by registered position")
        counts = (df["Registered Position"].value_counts()
                  .reindex(E.POSITIONS).dropna())
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=U.BLUE))
        fig.update_yaxes(gridcolor=GRID)
        show_plot(base_layout(fig, 340))
    with right:
        st.subheader("Age distribution")
        fig = go.Figure(go.Histogram(x=df["Age"], xbins=dict(size=1), marker_color=U.BLUE))
        fig.update_xaxes(title="Age", gridcolor=GRID)
        fig.update_yaxes(gridcolor=GRID)
        show_plot(base_layout(fig, 340))

    with st.expander("Full club power table"):
        show_table(power.round({"Best XI Rating": 1}))


# ======================================================================
# 2. SQUAD MANAGER
# ======================================================================

def render_squad_cards(squad_all: pd.DataFrame, starter_ids: set):
    groups = [("GK", "Goalkeepers"), ("DEF", "Defenders"),
              ("MID", "Midfielders"), ("ATT", "Forwards")]
    cap = 24
    for line, title in groups:
        g = squad_all[squad_all["Registered Position"].map(E.LINE_OF) == line]
        g = g.sort_values("Overall", ascending=False)
        if g.empty:
            continue
        note = f" — showing top {cap}" if len(g) > cap else ""
        st.markdown(f"**{title}** ({len(g)}){note}")
        html(U.cards_grid([
            card_for(r, starter=int(r["Player ID"]) in starter_ids)
            for _, r in g.head(cap).iterrows()
        ]))


def page_squad():
    st.title("🏟️ Squad Manager")

    c1, c2, c3 = st.columns([3, 1.4, 1])
    options = [DREAM] + sorted(df["Club"].unique())
    club = c1.selectbox("Club", options, index=idx_of(options, "FC BARCELONA"))
    formation = c2.selectbox("Formation", list(E.FORMATIONS), index=1)
    bench_n = c3.selectbox("Bench", [5, 7, 9], index=1)

    squad_all = df if club == DREAM else df[df["Club"] == club]
    ordered = squad_all.sort_values("Overall", ascending=False)["Label"].tolist()
    out = st.multiselect("Unavailable players (injured / rested)", ordered)
    pool = squad_all[~squad_all["Label"].isin(out)].reset_index(drop=True)

    if len(pool) == 0:
        st.error("No players available.")
        return

    lineup, E_mat = E.build_lineup(pool, formation, OOP_PENALTY, ALLOW_OOP)
    lines = E.line_ratings(lineup)
    vacant = int(lineup["Rating"].isna().sum())
    if vacant:
        st.warning(f"{vacant} slot(s) could not be filled — the squad lacks players for "
                   "those positions. Try allowing out-of-position players in the sidebar.")

    power = power_table()
    avg_club = power["Best XI Rating"].mean()
    m = st.columns(5)
    m[0].metric("Team rating", f"{lines['TEAM']:.1f}",
                delta=f"{lines['TEAM'] - avg_club:+.1f} vs average club")
    m[1].metric("Goalkeeper", f"{lines['GK']:.1f}")
    m[2].metric("Defence", f"{lines['DEF']:.1f}")
    m[3].metric("Midfield", f"{lines['MID']:.1f}")
    m[4].metric("Attack", f"{lines['ATT']:.1f}")
    if club in set(power["Club"]):
        rank = int(power.loc[power["Club"] == club, "Rank"].iloc[0])
        st.caption(f"Power rank at full strength: **#{rank}** of {len(power)} clubs")

    tab_xi, tab_tac, tab_squad, tab_bench = st.tabs(
        ["🏟️ Starting XI", "📊 Tactics", "🧑‍🤝‍🧑 Squad", "🪑 Bench"])

    starter_rows = set(lineup["_row"].dropna().astype(int))
    starter_ids = set(lineup["Player ID"].dropna().astype(int))

    # ---- Starting XI -------------------------------------------------
    with tab_xi:
        left, right = st.columns([5, 4])
        with left:
            html(U.pitch_html(lineup))
        with right:
            st.subheader(f"{formation} · {club}")
            tbl = lineup[["Slot", "Position", "Player", "Rating", "Age", "Registered"]].copy()
            tbl["Fit"] = np.where(lineup["Natural"], "Natural", "Out of position")
            tbl.loc[lineup["Player"] == "— vacant —", "Fit"] = "Vacant"
            show_table(tbl, column_config={
                "Rating": st.column_config.NumberColumn(format="%.1f")})
            st.download_button(
                "⬇️ Download starting XI",
                lineup.drop(columns=["_row", "X", "Y"]).to_csv(index=False),
                file_name="PES_Starting_XI.csv", mime="text/csv")

    # ---- Tactics ------------------------------------------------------
    with tab_tac:
        a, b = st.columns(2)
        with a:
            st.subheader("Team profile")
            st.caption("Average attributes of the starting XI "
                       "(Attack: midfield + forwards, Defence: keeper + defence + midfield).")
            profile = E.team_profile(pool, lineup)
            show_plot(radar([("Starting XI", profile, U.BLUE)], rmin=50))
        with b:
            st.subheader("Strength by line")
            names = {"GK": "Goalkeeper", "DEF": "Defence", "MID": "Midfield", "ATT": "Attack"}
            vals = [lines[k] for k in names]
            show_plot(lollipop(list(names.values()), vals, xmin=50, height=230))

            st.subheader("Best formation for this squad")
            fit = E.formation_fit(pool, OOP_PENALTY, ALLOW_OOP).dropna(subset=["Team Rating"])
            cols = [U.BLUE if f == formation else U.MUTED for f in fit["Formation"]]
            show_plot(lollipop(fit["Formation"], fit["Team Rating"],
                               xmin=float(fit["Team Rating"].min()) - 5,
                               colors=cols, height=330))
            if len(fit):
                best = fit.iloc[0]
                if best["Formation"] != formation:
                    st.info(f"**{best['Formation']}** suits this squad best "
                            f"({best['Team Rating']:.1f} vs {lines['TEAM']:.1f}).")
                else:
                    st.success(f"{formation} is already the best fit for this squad.")

        st.subheader("Weak spots")
        weak = lineup.dropna(subset=["Rating"]).sort_values("Rating").head(3)
        for _, r in weak.iterrows():
            st.write(f"• **{r['Slot']}** — {r['Player']} ({r['Rating']:.1f})")
        st.caption("Open **Transfer Targets** to see who could fix these.")

    # ---- Squad ---------------------------------------------------------
    with tab_squad:
        st.caption("XI badge = in your starting lineup. Cards show each player's Overall at his registered position.")
        render_squad_cards(squad_all, starter_ids)

    # ---- Bench ---------------------------------------------------------
    with tab_bench:
        bench = E.pick_bench(pool, E_mat, starter_rows, bench_n)
        if bench.empty:
            st.info("No substitutes available.")
        else:
            cards = []
            for _, r in bench.iterrows():
                p = pool.iloc[int(r["_row"])]
                cards.append(U.player_card(r["Player"], r["Rating"], r["Cover"],
                                           r["Club"], r["Age"], p["Nationality"]))
            st.caption("Position tag = the role each substitute is picked to cover.")
            html(U.cards_grid(cards))
            show_table(bench.drop(columns=["_row", "Player ID"]),
                       column_config={"Rating": st.column_config.NumberColumn(format="%.1f")})


# ======================================================================
# 3. TRANSFER TARGETS
# ======================================================================

def page_transfers():
    st.title("💸 Transfer Targets")
    st.caption("Finds the signings that improve your best XI the most. "
               "The spreadsheet has no prices or wages, so players are ranked purely by "
               "how much they would strengthen the team.")

    c1, c2, c3 = st.columns([3, 1.4, 1.4])
    club = c1.selectbox("Your club", ALL_CLUBS, index=idx_of(ALL_CLUBS, "C. ATLÉTICO DE MADRID"))
    formation = c2.selectbox("Formation", list(E.FORMATIONS), index=1)
    signings = c3.slider("Signings in the plan", 1, 6, 3)

    f1, f2, f3 = st.columns(3)
    sources = f1.multiselect("Look in", ["Other clubs", "Free agents & legends"],
                             default=["Other clubs", "Free agents & legends"])
    max_age = f2.slider("Maximum age", 17, 45, 30)
    focus = f3.multiselect("Only players who can play…", E.POSITIONS,
                           help="Leave empty to consider every position.")

    squad = df[df["Club"] == club].reset_index(drop=True)
    market = df[df["Club"] != club]
    is_free = market["Club"] == E.NO_CLUB
    keep = pd.Series(False, index=market.index)
    if "Other clubs" in sources:
        keep |= ~is_free
    if "Free agents & legends" in sources:
        keep |= is_free
    market = market[keep & (market["Age"] <= max_age)]
    if focus:
        market = market[market[focus].sum(axis=1) > 0]
    market = market.reset_index(drop=True)

    if market.empty:
        st.info("No players match these filters.")
        return

    now_lineup, _ = E.build_lineup(squad, formation, OOP_PENALTY, ALLOW_OOP)
    plan = E.shopping_list(squad, market, formation, signings, OOP_PENALTY, ALLOW_OOP)

    if plan.empty:
        st.success("Nobody on the market would improve your starting XI with these filters.")
        return

    signed = market[market["Player ID"].isin(plan["Player ID"])]
    after_pool = pd.concat([squad, signed], ignore_index=True)
    after_lineup, _ = E.build_lineup(after_pool, formation, OOP_PENALTY, ALLOW_OOP)

    before_rating = now_lineup["Rating"].mean()
    after_rating = after_lineup["Rating"].mean()
    m1, m2, m3 = st.columns(3)
    m1.metric("Team rating now", f"{before_rating:.1f}")
    m2.metric("After the plan", f"{after_rating:.1f}", delta=f"{after_rating - before_rating:+.1f}")
    m3.metric("Signings", len(plan))

    st.subheader("Recommended shopping list")
    cards = []
    for _, r in plan.iterrows():
        p = market[market["Player ID"] == r["Player ID"]].iloc[0]
        cards.append(U.player_card(p["Player Name"], r["Rating"], r["Plays As"],
                                   p["Club"], p["Age"], p["Nationality"]))
    html(U.cards_grid(cards))
    plan_view = plan[["Step", "Player", "Age", "Club", "Plays As", "Replaces",
                      "Current Rating", "Rating", "XI Gain"]].rename(
        columns={"Rating": "New Rating"})
    show_table(plan_view, column_config={
        "Current Rating": st.column_config.NumberColumn(format="%.1f"),
        "New Rating": st.column_config.NumberColumn(format="%.1f"),
        "XI Gain": st.column_config.NumberColumn(format="%.1f"),
    })
    st.caption("Each step assumes the previous signings have already joined. "
               "XI Gain = rating points added to the starting XI's total.")

    left, right = st.columns(2)
    with left:
        st.subheader("Now")
        html(U.pitch_html(now_lineup))
    with right:
        st.subheader("After the plan")
        html(U.pitch_html(after_lineup))

    st.subheader("Full shortlist")
    short = E.transfer_shortlist(squad, market, formation, OOP_PENALTY, ALLOW_OOP)
    if short.empty:
        st.info("No single player would improve the XI.")
    else:
        view = short.head(60).drop(columns=["Player ID", "Position"])
        view = view.rename(columns={"Rating": "New Rating"})
        show_table(view, column_config={
            "Current Rating": st.column_config.NumberColumn(format="%.1f"),
            "New Rating": st.column_config.NumberColumn(format="%.1f"),
            "XI Gain": st.column_config.NumberColumn(format="%.1f"),
        })
        st.download_button("⬇️ Download shortlist", short.drop(columns=["Player ID"]).to_csv(index=False),
                           file_name="PES_Transfer_Shortlist.csv", mime="text/csv")


# ======================================================================
# 4. PLAYER PROFILE
# ======================================================================

def page_profile():
    st.title("👤 Player Profile")

    labels = df.sort_values("Overall", ascending=False)["Label"].tolist()
    default = next((l for l in labels if l.startswith("MESSI  ·")), labels[0])
    chosen = st.selectbox("Search player (type a name)", labels, index=labels.index(default))
    one = df[df["Label"] == chosen].head(1)
    p = one.iloc[0]

    left, right = st.columns([1, 2.4])
    with left:
        html(card_for(p, big=True))
        st.markdown("**Special abilities**")
        abilities = [a for a in E.ABILITY_COLUMNS if p[a] == 1]
        html(U.badges(abilities))
    with right:
        r1 = st.columns(4)
        r1[0].metric("Age", int(p["Age"]))
        r1[1].metric("Height", f"{int(p['Height'])} cm")
        r1[2].metric("Foot", p["Dominant Leg"])
        r1[3].metric("Sides", p["Dominant Side"])
        r2 = st.columns(4)
        r2[0].metric("Club", p["Club"])
        r2[1].metric("Nationality", p["Nationality"])
        r2[2].metric("Plays", p["Plays"])
        r2[3].metric("Versatility", int(p["Versatility Index"]))
        r3 = st.columns(4)
        r3[0].metric("Injury tolerance", p["Injury Tolerance"])
        r3[1].metric("Condition", int(p["Condition"]))
        r3[2].metric("Weak foot (acc / freq)",
                     f"{int(p['Weak Foot Accuracy'])} / {int(p['Weak Foot Frequency'])}")
        r3[3].metric("Consistency", int(p["Consistency"]))

    a, b = st.columns(2)
    with a:
        st.subheader("Position ratings")
        st.caption("Blue = listed position. Grey = out of position (penalty applied).")
        eff = E.effective_matrix(one, OOP_PENALTY, ALLOW_OOP)[0]
        natural = one[E.POSITIONS].iloc[0].to_numpy() > 0
        rows = [(pos, eff[i], natural[i]) for i, pos in enumerate(E.POSITIONS)
                if eff[i] > E.NEG / 2]
        rows.sort(key=lambda t: -t[1])
        show_plot(lollipop([f"{pos}" for pos, _, _ in rows], [v for _, v, _ in rows],
                           xmin=max(0, min(v for _, v, _ in rows) - 10),
                           colors=[U.BLUE if nat else U.MUTED for _, _, nat in rows],
                           height=max(260, 30 * len(rows) + 60)))
    with b:
        st.subheader("Attribute profile")
        show_plot(radar([(p["Player Name"], group_profile(one), U.BLUE)], rmin=30))

    with st.expander("All attributes"):
        attr_df = pd.DataFrame({"Attribute": ATTRIBUTES,
                                "Value": [int(p[a]) for a in ATTRIBUTES]})
        show_table(attr_df, column_config={
            "Value": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d")})

    st.subheader("Similar players")
    sim = E.similar_players(df, int(p["Player ID"]), 8)
    show_table(sim[["Player Name", "Age", "Club", "Registered Position", "Overall", "Similarity"]],
               column_config={
                   "Overall": st.column_config.NumberColumn(format="%.1f"),
                   "Similarity": st.column_config.NumberColumn(format="%.0f%%")})


# ======================================================================
# 5. PLAYER COMPARISON
# ======================================================================

def page_compare():
    st.title("⚔️ Player Comparison")

    labels = df.sort_values("Overall", ascending=False)["Label"].tolist()
    d1 = next((l for l in labels if l.startswith("MESSI  ·")), labels[0])
    d2 = next((l for l in labels if l.startswith("C. RONALDO  ·")), labels[min(1, len(labels) - 1)])
    c1, c2 = st.columns(2)
    l1 = c1.selectbox("First player", labels, index=labels.index(d1))
    l2 = c2.selectbox("Second player", labels, index=labels.index(d2))
    one = df[df["Label"] == l1].head(1)
    two = df[df["Label"] == l2].head(1)
    p1, p2 = one.iloc[0], two.iloc[0]
    n1 = p1["Player Name"]
    n2 = p2["Player Name"] if p2["Player Name"] != n1 else p2["Player Name"] + " (2)"

    k1, k2 = st.columns(2)
    with k1:
        html(card_for(p1, big=True))
    with k2:
        html(card_for(p2, big=True))

    show_plot(radar([(n1, group_profile(one), U.BLUE),
                     (n2, group_profile(two), U.ORANGE)], rmin=30))

    st.subheader("Attribute by attribute")
    comp = pd.DataFrame({
        "Attribute": ATTRIBUTES,
        n1: [int(p1[a]) for a in ATTRIBUTES],
        n2: [int(p2[a]) for a in ATTRIBUTES],
    })
    comp["Difference"] = comp[n1] - comp[n2]
    comp["Better"] = np.where(comp["Difference"] > 0, n1,
                              np.where(comp["Difference"] < 0, n2, "Equal"))
    show_table(comp)

    st.subheader("Ratings by position")
    e1 = E.effective_matrix(one, OOP_PENALTY, ALLOW_OOP)[0]
    e2 = E.effective_matrix(two, OOP_PENALTY, ALLOW_OOP)[0]
    pos_df = pd.DataFrame({
        "Position": E.POSITIONS,
        n1: [round(v, 1) if v > E.NEG / 2 else None for v in e1],
        n2: [round(v, 1) if v > E.NEG / 2 else None for v in e2],
    })
    show_table(pos_df)

    s1 = (comp["Difference"] > 0).sum()
    s2 = (comp["Difference"] < 0).sum()
    st.caption(f"{n1} is better in {s1} attributes, {n2} in {s2}.")


# ======================================================================
# 6. POSITION RANKINGS
# ======================================================================

def page_rankings():
    st.title("🏆 Position Rankings")

    c1, c2, c3, c4 = st.columns([2, 1.2, 1.4, 1.4])
    position = c1.selectbox("Position", E.POSITIONS,
                            format_func=lambda p: f"{p} — {E.POSITION_NAMES[p]}", index=2)
    natural = c2.toggle("Natural only", value=True,
                        help="Only players listed for this position in the spreadsheet.")
    max_age = c3.slider("Maximum age", 17, 45, 45)
    top_n = c4.slider("Players", 5, 50, 20)

    col = E.POS_INDEX[position]
    eff = E.effective_matrix(df, OOP_PENALTY, ALLOW_OOP)[:, col]
    mask = (eff > E.NEG / 2) & (df["Age"] <= max_age).to_numpy()
    if natural:
        mask &= (df[position] == 1).to_numpy()
    ranked = df[mask].assign(Rating=eff[mask]).sort_values("Rating", ascending=False).head(top_n)

    if ranked.empty:
        st.info("No players match.")
        return

    labels = [f"{i + 1}. {n}" for i, n in enumerate(ranked["Player Name"])]
    show_plot(lollipop(labels, ranked["Rating"], xmin=float(ranked["Rating"].min()) - 5,
                       height=max(300, 26 * len(ranked) + 60)))
    show_table(ranked[["Player Name", "Age", "Club", "Nationality", "Registered Position",
                       "Plays", "Rating"]],
               column_config={"Rating": st.column_config.NumberColumn(
                   f"Rating at {position}", format="%.1f")})


# ======================================================================
# 7. PLAYER DATABASE
# ======================================================================

def page_database():
    st.title("📋 Player Database")

    q = st.text_input("🔍 Search player name", placeholder="e.g. XAVI")
    a, b, c = st.columns(3)
    club = a.selectbox("Club", ["All"] + sorted(df["Club"].unique()))
    nation = b.selectbox("Nationality", ["All"] + sorted(df["Nationality"].unique()))
    can_play = c.multiselect("Can play (any of)", E.POSITIONS)
    d, e, f = st.columns(3)
    age_lo, age_hi = d.slider("Age", int(df["Age"].min()), int(df["Age"].max()),
                              (int(df["Age"].min()), int(df["Age"].max())))
    min_ovr = e.slider("Minimum Overall", 60, 95, 60)
    abilities = f.multiselect("Has abilities (all of)", E.ABILITY_COLUMNS)

    res = df
    if q:
        res = res[res["Player Name"].str.contains(q, case=False, na=False, regex=False)]
    if club != "All":
        res = res[res["Club"] == club]
    if nation != "All":
        res = res[res["Nationality"] == nation]
    if can_play:
        res = res[res[can_play].sum(axis=1) > 0]
    res = res[res["Age"].between(age_lo, age_hi) & (res["Overall"] >= min_ovr)]
    for ab in abilities:
        res = res[res[ab] == 1]
    res = res.sort_values("Overall", ascending=False)

    st.metric("Players found", f"{len(res):,}")
    cols = ["Player Name", "Age", "Height", "Club", "Nationality", "Registered Position",
            "Plays", "Overall", "Versatility Index", "Dominant Leg"]
    show_table(res[cols], height=600, column_config={
        "Overall": st.column_config.ProgressColumn(min_value=60, max_value=95, format="%.1f")})
    export_cols = [c for c in res.columns if c not in ("Label",)]
    st.download_button("⬇️ Export filtered players (CSV)",
                       res[export_cols].to_csv(index=False),
                       file_name="PES_Filtered_Players.csv", mime="text/csv")


# ======================================================================
# 8. RATING METHOD
# ======================================================================

def page_method():
    st.title("🧮 Rating Method")
    st.markdown(
        """
**What the spreadsheet contains**

* The 12 position columns (GK, SW, CB … CF) are **0/1 flags** – "can this player play here?" – not ratings.
* The 23 columns from *Dribbling* to *Long Throw* are **0/1 special abilities**.
* There is no single overall rating, so this app calculates one.

**How ratings are calculated**

For each position, a weighted average of the 0–99 attributes is taken (table below).
A player's **Overall** is his rating at his registered position.
In the Squad Manager, a player playing a position he is *not* listed for loses the
out-of-position penalty (set in the sidebar). Goalkeepers never play outfield and vice-versa.

**These formulas are this app's own design** – not values stored in the game. To change them,
edit the `WEIGHTS` table at the top of `pes_engine.py`; every screen updates automatically.

**Squad selection** uses the Hungarian algorithm, which finds the best combination of
11 different players for the formation (nobody is picked twice, and a star is moved to
whichever slot helps the team most).
        """
    )
    weights = (pd.DataFrame(E.WEIGHTS).fillna(0) * 100).round(0).astype(int)
    weights = weights.reindex(columns=E.POSITIONS).reset_index().rename(columns={"index": "Attribute"})
    st.subheader("Attribute weights (% of the position rating)")
    show_table(weights)


# ======================================================================
# ROUTER
# ======================================================================

if page == PAGES[0]:
    page_dashboard()
elif page == PAGES[1]:
    page_squad()
elif page == PAGES[2]:
    page_transfers()
elif page == PAGES[3]:
    page_profile()
elif page == PAGES[4]:
    page_compare()
elif page == PAGES[5]:
    page_rankings()
elif page == PAGES[6]:
    page_database()
else:
    page_method()
