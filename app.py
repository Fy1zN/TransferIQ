import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
#  PAGE CONFIG  (must be first st call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="TransferIQ · Market Value Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Barlow', sans-serif; background-color: #0a0f1e; color: #e8edf5; }
.stApp { background-color: #0a0f1e; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1628 0%, #0a1020 100%); border-right: 1px solid #1e3a5f; }
[data-testid="stSidebar"] label { color: #7eb8f7 !important; font-size: 0.78rem !important; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; }

h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: 0.03em; }

.metric-card { background: linear-gradient(135deg, #0f1e38 0%, #0d1628 100%); border: 1px solid #1e3a5f; border-radius: 12px; padding: 20px 24px; text-align: center; }
.metric-card:hover { border-color: #3a9fff; }
.metric-label { font-size: 0.72rem; color: #5b8ab5; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
.metric-value { font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; font-weight: 700; color: #e8edf5; line-height: 1; }
.metric-sub { font-size: 0.75rem; color: #3a9fff; margin-top: 4px; }

.pred-box { background: linear-gradient(135deg, #0b3d1f 0%, #062d16 100%); border: 2px solid #1db954; border-radius: 16px; padding: 32px; text-align: center; }
.pred-label { font-size: 0.8rem; color: #1db954; text-transform: uppercase; letter-spacing: 0.15em; }
.pred-value { font-family: 'Barlow Condensed', sans-serif; font-size: 3.5rem; font-weight: 800; color: #ffffff; line-height: 1.1; }
.pred-range { font-size: 0.85rem; color: #a0c8a0; margin-top: 6px; }

.section-header { font-family: 'Barlow Condensed', sans-serif; font-size: 1.4rem; font-weight: 700; color: #e8edf5; border-left: 4px solid #3a9fff; padding-left: 12px; margin: 28px 0 16px 0; text-transform: uppercase; letter-spacing: 0.05em; }

.stTabs [data-baseweb="tab-list"] { background: #0d1628; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { font-family: 'Barlow Condensed', sans-serif; font-size: 0.95rem; font-weight: 600; letter-spacing: 0.05em; color: #5b8ab5; background: transparent; border-radius: 8px; padding: 8px 20px; border: none; }
.stTabs [aria-selected="true"] { background: #1e3a5f !important; color: #3a9fff !important; }

.stButton > button { font-family: 'Barlow Condensed', sans-serif; font-size: 1rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; background: linear-gradient(135deg, #1db954, #15833c); color: white; border: none; border-radius: 10px; padding: 12px 32px; width: 100%; }
.stButton > button:hover { background: linear-gradient(135deg, #22d45f, #1db954); box-shadow: 0 4px 20px rgba(29,185,84,0.4); }

.info-box { background: #0d1628; border: 1px solid #1e3a5f; border-radius: 10px; padding: 16px 20px; font-size: 0.88rem; color: #7eb8f7; margin: 8px 0; }
.sidebar-brand { font-family: 'Barlow Condensed', sans-serif; font-size: 1.6rem; font-weight: 800; color: #3a9fff; letter-spacing: 0.05em; text-align: center; padding: 0 0 16px 0; }
.sidebar-section { font-family: 'Barlow Condensed', sans-serif; font-size: 0.85rem; font-weight: 700; color: #3a9fff; text-transform: uppercase; letter-spacing: 0.12em; margin: 16px 0 8px 0; padding-bottom: 4px; border-bottom: 1px solid #1e3a5f; }

hr { border-color: #1e3a5f; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor="#0a0f1e",
    plot_bgcolor="#0d1628",
    font=dict(color="#e8edf5", family="Barlow, sans-serif"),
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    colorway=["#3a9fff","#1db954","#ffd700","#ff6b6b","#a78bfa","#f97316"],
)

# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────

def fmt(v):
    try:
        v = float(v)
        if v >= 1_000_000: return f"€{v/1e6:.1f}m"
        if v >= 1_000:     return f"€{v/1000:.0f}k"
        return f"€{v:,.0f}"
    except Exception:
        return "N/A"

def build_input_vector(inputs: dict, model_features: list) -> pd.DataFrame:
    row = {col: 0.0 for col in model_features}
    for key, val in inputs.items():
        if key in row:
            row[key] = float(val)
        if key == "Position":
            ohe = f"Pos_{val}"
            if ohe in row: row[ohe] = 1.0
        if key == "League":
            ohe = f"Comp_{val}"
            if ohe in row: row[ohe] = 1.0
        if key == "Nation":
            ohe = f"Nation_{val}"
            if ohe in row: row[ohe] = 1.0
        if key == "Squad":
            ohe = f"Squad_{val}"
            if ohe in row: row[ohe] = 1.0
    return pd.DataFrame([row])

# ─────────────────────────────────────────
#  LOAD ASSETS
# ─────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    with open("xgb_final_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("preprocessed_dataset.csv")

@st.cache_data(show_spinner=False)
def load_feature_importance():
    return pd.read_excel("feature_importance.xlsx")

@st.cache_data(show_spinner=False)
def load_predictions():
    path = "player_data_with_predictions.xlsx"
    if not os.path.exists(path):
        return None
    try:
        return pd.read_excel(path)
    except Exception:
        return None

# ── Fail fast with a clear message ──
missing = [f for f in ["xgb_final_model.pkl", "preprocessed_dataset.csv", "feature_importance.xlsx"]
           if not os.path.exists(f)]
if missing:
    st.error(f"❌ Missing files: `{'`, `'.join(missing)}`  \nMake sure they are in the same folder as `app.py`.")
    st.stop()

try:
    model     = load_model()
    player_df = load_data()
    fi_df     = load_feature_importance()
    pred_df   = load_predictions()
except Exception as e:
    st.error(f"❌ Failed to load assets: {e}")
    st.stop()

# Get model feature list
try:
    model_features = model.get_booster().feature_names
    if model_features is None:
        raise ValueError("no feature names")
except Exception:
    model_features = fi_df['Variable'].tolist()

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚽ TransferIQ</div>', unsafe_allow_html=True)
    st.markdown("*Football Market Value Predictor*")
    st.markdown("---")

    st.markdown('<div class="sidebar-section">🧍 Player Profile</div>', unsafe_allow_html=True)
    age      = st.slider("Age", 16, 40, 24)
    position = st.selectbox("Position", ["FW","MF","DF","GK","MFFW","DFMF","MFDF"])
    leagues  = sorted(player_df['Comp'].dropna().unique().tolist())
    league   = st.selectbox("League", leagues)
    nations  = sorted(player_df['Nation'].dropna().unique().tolist())
    nation   = st.selectbox("Nationality", nations)
    squads   = sorted(player_df['Squad'].dropna().unique().tolist())
    squad    = st.selectbox("Club", squads)

    st.markdown('<div class="sidebar-section">⏱ Playing Time</div>', unsafe_allow_html=True)
    minutes  = st.number_input("Minutes Played", 0, 3800, 2000, step=50)
    starts   = st.number_input("Starts", 0, 38, 20)
    nineties = round(minutes / 90, 1)

    st.markdown('<div class="sidebar-section">⚽ Attacking</div>', unsafe_allow_html=True)
    goals   = st.number_input("Goals", 0, 50, 5)
    assists = st.number_input("Assists", 0, 30, 4)
    sot     = st.number_input("Shots on Target", 0, 100, 15)
    gca     = st.number_input("Goal Creating Actions", 0, 50, 8)
    sca     = st.number_input("Shot Creating Actions", 0, 150, 30)

    st.markdown('<div class="sidebar-section">🎯 Passing</div>', unsafe_allow_html=True)
    pas_cmp     = st.number_input("Passes Completed", 0, 3000, 800)
    pas_att     = st.number_input("Passes Attempted", 1, 3500, 950)
    pas_cmp_pct = round(pas_cmp / max(pas_att, 1) * 100, 1)
    pas_prog    = st.number_input("Progressive Passes", 0, 300, 60)
    pas_3rd     = st.number_input("Passes into Final 3rd", 0, 200, 30)

    st.markdown('<div class="sidebar-section">🛡 Defending</div>', unsafe_allow_html=True)
    tkl     = st.number_input("Tackles", 0, 200, 30)
    tkl_won = st.number_input("Tackles Won", 0, 150, 20)
    inter   = st.number_input("Interceptions", 0, 100, 15)
    clr     = st.number_input("Clearances", 0, 200, 20)
    blocks  = st.number_input("Blocks", 0, 100, 10)

    st.markdown('<div class="sidebar-section">🏃 Possession</div>', unsafe_allow_html=True)
    touches  = st.number_input("Touches", 0, 4000, 1200)
    carries  = st.number_input("Carries", 0, 2000, 600)
    car_prog = st.number_input("Progressive Carries", 0, 300, 60)
    recov    = st.number_input("Ball Recoveries", 0, 300, 60)

    st.markdown('<div class="sidebar-section">📊 Aerial & Discipline</div>', unsafe_allow_html=True)
    aer_won  = st.number_input("Aerials Won", 0, 200, 20)
    aer_lost = st.number_input("Aerials Lost", 0, 200, 15)
    crd_y    = st.number_input("Yellow Cards", 0, 15, 2)
    crd_r    = st.number_input("Red Cards", 0, 5, 0)

# ─────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────

user_inputs = {
    "Age": age, "Min": minutes, "Starts": starts, "90s": nineties,
    "Goals": goals, "Assists": assists, "SoT": sot, "GCA": gca, "SCA": sca,
    "PasTotCmp": pas_cmp, "PasTotAtt": pas_att, "PasTotCmp%": pas_cmp_pct,
    "PasProg": pas_prog, "Pas3rd": pas_3rd,
    "Tkl": tkl, "TklWon": tkl_won, "Int": inter, "Clr": clr, "Blocks": blocks,
    "Touches": touches, "Carries": carries, "CarProg": car_prog, "Recov": recov,
    "AerWon": aer_won, "AerLost": aer_lost, "CrdY": crd_y, "CrdR": crd_r,
    "Position": position, "League": league, "Nation": nation, "Squad": squad,
}

try:
    input_vector = build_input_vector(user_inputs, model_features)
    log_pred     = float(model.predict(input_vector)[0])
    pred_euros   = float(np.expm1(log_pred))
    pred_low     = float(np.expm1(log_pred - 0.3))
    pred_high    = float(np.expm1(log_pred + 0.3))
    pred_ok      = True
except Exception as e:
    pred_ok    = False
    pred_error = str(e)

# ─────────────────────────────────────────
#  PAGE HEADER
# ─────────────────────────────────────────

st.markdown("""
<div style="padding:28px 0 8px 0;">
  <div style="font-family:'Barlow Condensed',sans-serif;font-size:3rem;font-weight:800;
              letter-spacing:0.04em;color:#ffffff;line-height:1;">
    TRANSFER<span style="color:#3a9fff;">IQ</span>
  </div>
  <div style="font-size:1rem;color:#5b8ab5;letter-spacing:0.15em;text-transform:uppercase;margin-top:4px;">
    Football Market Value Intelligence Platform
  </div>
</div><hr>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  KPI STRIP
# ─────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)
for col, label, value, sub in [
    (k1, "Players Analysed",  f"{len(player_df):,}",                          "2022–23 Season"),
    (k2, "Avg Market Value",  fmt(player_df['Market Value Euros'].mean()),     "Mean"),
    (k3, "Median Value",      fmt(player_df['Market Value Euros'].median()),   "50th Percentile"),
    (k4, "Leagues Covered",   str(player_df['Comp'].nunique()),                "Top 5 European"),
    (k5, "Features Used",     str(len(model_features)),                        "After Selection"),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  PREDICTION PANEL
# ─────────────────────────────────────────

pred_col, info_col = st.columns([1, 2])

with pred_col:
    if pred_ok:
        st.markdown(f"""
        <div class="pred-box">
            <div class="pred-label">Estimated Market Value</div>
            <div class="pred-value">{fmt(pred_euros)}</div>
            <div class="pred-range">Range: {fmt(pred_low)} – {fmt(pred_high)}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.error(f"Prediction failed: {pred_error}")

with info_col:
    if pred_ok:
        pct_rank   = float((player_df['Market Value Euros'] < pred_euros).mean() * 100)
        tier_color = ("#ffd700" if pred_euros >= 80e6 else
                      "#3a9fff" if pred_euros >= 50e6 else
                      "#1db954" if pred_euros >= 20e6 else
                      "#ff9f40" if pred_euros >= 5e6  else "#a0a0a0")
        tier_label = ("World Class ⭐⭐⭐" if pred_euros >= 80e6 else
                      "Elite ⭐⭐"         if pred_euros >= 50e6 else
                      "Top Tier ⭐"        if pred_euros >= 20e6 else
                      "Solid Pro"          if pred_euros >= 5e6  else "Fringe / Reserve")

        st.markdown(f"""
        <div class="info-box" style="border-color:{tier_color};margin-bottom:10px;">
            <span style="color:{tier_color};font-family:'Barlow Condensed',sans-serif;
                         font-size:1.1rem;font-weight:700;">{tier_label}</span><br>
            <span style="color:#a0b8d0;">Valued higher than
            <strong style="color:#e8edf5;">{pct_rank:.0f}%</strong> of dataset players.</span>
        </div>""", unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_euros / 1e6,
            number={"prefix": "€", "suffix": "m", "font": {"size": 22, "color": "#e8edf5"}},
            gauge={
                "axis":        {"range": [0, 200], "tickcolor": "#5b8ab5", "tickfont": {"color": "#5b8ab5"}},
                "bar":         {"color": tier_color, "thickness": 0.25},
                "bgcolor":     "#0d1628",
                "bordercolor": "#1e3a5f",
                "steps": [
                    {"range": [0,   5], "color": "#0f1e38"},
                    {"range": [5,  20], "color": "#102030"},
                    {"range": [20, 50], "color": "#0f2840"},
                    {"range": [50,200], "color": "#0d2035"},
                ],
                "threshold": {"line": {"color": "#ffffff", "width": 2},
                              "thickness": 0.8, "value": pred_euros / 1e6},
            }
        ))
        gauge.update_layout(height=180, margin=dict(t=20,b=10,l=20,r=20),
                            paper_bgcolor="#0a0f1e", font_color="#e8edf5")
        st.plotly_chart(gauge, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Dataset Insights",
    "🔍  Feature Importance",
    "🏆  Player Explorer",
    "📈  Model Performance",
])

# ── TAB 1: DATASET INSIGHTS ──────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Market Value Distribution</div>', unsafe_allow_html=True)
        fig_dist = make_subplots(rows=1, cols=2, subplot_titles=["Raw (€m)", "Log-Transformed"])
        fig_dist.add_trace(go.Histogram(x=player_df['Market Value Euros']/1e6,
                                         nbinsx=60, name="Raw",
                                         marker_color="#3a9fff", opacity=0.8), row=1, col=1)
        fig_dist.add_trace(go.Histogram(x=np.log1p(player_df['Market Value Euros']),
                                         nbinsx=60, name="Log",
                                         marker_color="#1db954", opacity=0.8), row=1, col=2)
        fig_dist.update_layout(**PLOT_THEME, height=320,
                                margin=dict(t=40,b=20,l=20,r=20), showlegend=False)
        fig_dist.update_annotations(font_color="#7eb8f7")
        st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Value by Position</div>', unsafe_allow_html=True)
        pos_stats = (player_df.groupby('Pos')['Market Value Euros']
                     .agg(Median='median', Mean='mean', Count='count')
                     .reset_index().sort_values('Median', ascending=False))
        fig_pos = px.bar(pos_stats, x='Pos', y='Median',
                         color='Median', color_continuous_scale='Blues',
                         hover_data=['Mean','Count'],
                         labels={'Median':'Median Value (€)','Pos':'Position'})
        fig_pos.update_layout(**PLOT_THEME, height=320,
                               margin=dict(t=20,b=20,l=20,r=20),
                               showlegend=False, coloraxis_showscale=False)
        fig_pos.update_traces(marker_line_width=0)
        st.plotly_chart(fig_pos, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-header">Value by League</div>', unsafe_allow_html=True)
        league_stats = (player_df.groupby('Comp')['Market Value Euros']
                        .median().reset_index()
                        .rename(columns={'Market Value Euros':'Median Value','Comp':'League'})
                        .sort_values('Median Value'))
        fig_lg = px.bar(league_stats, x='Median Value', y='League', orientation='h',
                        color='Median Value', color_continuous_scale='Teal',
                        labels={'Median Value':'Median Value (€)'})
        fig_lg.update_layout(**PLOT_THEME, height=300,
                              margin=dict(t=20,b=20,l=20,r=20), coloraxis_showscale=False)
        fig_lg.update_traces(marker_line_width=0)
        st.plotly_chart(fig_lg, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header">Age vs Market Value</div>', unsafe_allow_html=True)
        sample = player_df.sample(min(600, len(player_df)), random_state=42)
        fig_age = px.scatter(sample, x='Age', y='Market Value Euros',
                             color='Pos', opacity=0.6,
                             labels={'Market Value Euros':'Market Value (€)'},
                             color_discrete_sequence=PLOT_THEME['colorway'])
        fig_age.update_layout(**PLOT_THEME, height=300,
                               margin=dict(t=20,b=20,l=20,r=20),
                               legend=dict(bgcolor="#0d1628", bordercolor="#1e3a5f"))
        fig_age.update_traces(marker_size=5)
        st.plotly_chart(fig_age, use_container_width=True)

# ── TAB 2: FEATURE IMPORTANCE ────────────
with tab2:
    st.markdown('<div class="section-header">What Drives Market Value?</div>', unsafe_allow_html=True)

    max_fi = min(50, len(fi_df))
    n_show = st.slider("Show top N features", 10, max_fi, min(25, max_fi))
    top_fi = fi_df.head(n_show).sort_values('Feature Importance Score', ascending=True)

    fig_fi = px.bar(top_fi, x='Feature Importance Score', y='Variable', orientation='h',
                    color='Feature Importance Score', color_continuous_scale='Blues',
                    labels={'Variable':'Feature','Feature Importance Score':'Importance'})
    fig_fi.update_layout(**PLOT_THEME, height=max(400, n_show * 22),
                          margin=dict(t=20,b=20,l=20,r=20), coloraxis_showscale=False)
    fig_fi.update_traces(marker_line_width=0)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="section-header">Feature Correlations</div>', unsafe_allow_html=True)
    numeric_features = [
        f for f in fi_df['Variable'].head(12).tolist()
        if f in player_df.columns and pd.api.types.is_numeric_dtype(player_df[f])
    ]
    if 'Market Value Euros' not in numeric_features:
        numeric_features.append('Market Value Euros')

    if len(numeric_features) > 2:
        corr = player_df[numeric_features].corr().round(2)
        fig_heat = px.imshow(corr, text_auto=True, aspect="auto",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig_heat.update_layout(**PLOT_THEME, height=500, margin=dict(t=20,b=20,l=20,r=20))
        fig_heat.update_traces(textfont_size=10)
        st.plotly_chart(fig_heat, use_container_width=True)

# ── TAB 3: PLAYER EXPLORER ───────────────
with tab3:
    st.markdown('<div class="section-header">Top Players by Market Value</div>', unsafe_allow_html=True)

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        filt_pos = st.multiselect("Filter by Position",
                                   sorted(player_df['Pos'].dropna().unique()), default=[])
    with ec2:
        filt_league = st.multiselect("Filter by League",
                                      sorted(player_df['Comp'].dropna().unique()), default=[])
    with ec3:
        top_n = st.slider("Number of Players", 10, 50, 20)

    filtered = player_df.copy()
    if filt_pos:    filtered = filtered[filtered['Pos'].isin(filt_pos)]
    if filt_league: filtered = filtered[filtered['Comp'].isin(filt_league)]

    if filtered.empty:
        st.warning("No players match the selected filters.")
    else:
        keep_cols = [c for c in ['Player','Pos','Squad','Comp','Age','Goals','Assists',
                                  'Min','Market Value Euros'] if c in filtered.columns]
        top_players = filtered.nlargest(top_n, 'Market Value Euros')[keep_cols].copy()
        top_players['Value (€m)']   = (top_players['Market Value Euros'] / 1e6).round(2)
        top_players['Market Value'] = top_players['Market Value Euros'].apply(fmt)
        top_sorted = top_players.sort_values('Value (€m)', ascending=True)

        hover_cols = [c for c in ['Squad','Comp','Age','Goals','Assists','Market Value']
                      if c in top_sorted.columns]
        fig_top = px.bar(top_sorted, x='Value (€m)', y='Player', orientation='h',
                         color='Pos' if 'Pos' in top_sorted.columns else None,
                         hover_data=hover_cols,
                         color_discrete_sequence=PLOT_THEME['colorway'],
                         labels={'Value (€m)':'Market Value (€m)'})
        fig_top.update_layout(**PLOT_THEME, height=max(400, top_n * 24),
                               margin=dict(t=20,b=20,l=20,r=20),
                               legend=dict(bgcolor="#0d1628", bordercolor="#1e3a5f"))
        fig_top.update_traces(marker_line_width=0)
        st.plotly_chart(fig_top, use_container_width=True)

    # Similar players
    if pred_ok and pred_df is not None:
        st.markdown('<div class="section-header">Similar Players to Your Input</div>',
                    unsafe_allow_html=True)
        has_cols = {'Predicted Market Value','Age'}.issubset(pred_df.columns)
        if has_cols:
            similar = pred_df[
                pred_df['Predicted Market Value'].between(pred_euros * 0.6, pred_euros * 1.4) &
                pred_df['Age'].between(age - 3, age + 3)
            ].copy()
            show_cols = [c for c in ['Player','Pos','Squad','Comp','Age','Goals','Assists',
                                      'Actual Market Value','Predicted Market Value']
                         if c in similar.columns]
            similar = similar[show_cols].nlargest(8, 'Predicted Market Value')
            if not similar.empty:
                for col in ['Actual Market Value','Predicted Market Value']:
                    if col in similar.columns:
                        similar[col] = similar[col].apply(fmt)
                st.dataframe(similar.reset_index(drop=True), use_container_width=True, height=280)
            else:
                st.info("No similar players found — try adjusting the sliders.")
        else:
            st.info("Generate `player_data_with_predictions.xlsx` from the modelling notebook first.")

# ── TAB 4: MODEL PERFORMANCE ─────────────
with tab4:
    st.markdown('<div class="section-header">Model Diagnostics</div>', unsafe_allow_html=True)

    req_cols = {'Player', 'Pos', 'Actual Market Value', 'Predicted Market Value'}
    if pred_df is not None and req_cols.issubset(pred_df.columns):
        diag_df = pred_df[list(req_cols)].dropna().copy()
        diag_df['Error (%)'] = (
            (diag_df['Predicted Market Value'] - diag_df['Actual Market Value']) /
            diag_df['Actual Market Value'].replace(0, np.nan) * 100
        )

        dc1, dc2 = st.columns(2)
        with dc1:
            sample_diag = diag_df.sample(min(500, len(diag_df)), random_state=42)
            max_v = float(diag_df['Actual Market Value'].max())
            fig_avp = px.scatter(
                sample_diag, x='Actual Market Value', y='Predicted Market Value',
                color='Pos', opacity=0.6, hover_data=['Player'],
                color_discrete_sequence=PLOT_THEME['colorway'],
                labels={'Actual Market Value':'Actual (€)','Predicted Market Value':'Predicted (€)'},
                title="Actual vs Predicted",
            )
            fig_avp.add_trace(go.Scatter(
                x=[0, max_v], y=[0, max_v], mode='lines',
                line=dict(color='#ff6b6b', dash='dash', width=1.5), name='Perfect Fit'
            ))
            fig_avp.update_layout(**PLOT_THEME, height=400,
                                   margin=dict(t=40,b=20,l=20,r=20),
                                   legend=dict(bgcolor="#0d1628", bordercolor="#1e3a5f"))
            fig_avp.update_traces(marker_size=5)
            st.plotly_chart(fig_avp, use_container_width=True)

        with dc2:
            fig_err = px.histogram(diag_df.dropna(subset=['Error (%)']),
                                    x='Error (%)', nbins=50,
                                    color_discrete_sequence=["#3a9fff"],
                                    title="Prediction Error Distribution (%)")
            fig_err.add_vline(x=0, line_dash="dash", line_color="#ff6b6b")
            fig_err.update_layout(**PLOT_THEME, height=400, margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig_err, use_container_width=True)

        err_clean = diag_df['Error (%)'].dropna()
        mae_med   = np.abs(diag_df['Predicted Market Value'] - diag_df['Actual Market Value']).median()
        within_20 = (err_clean.abs() <= 20).mean() * 100
        within_50 = (err_clean.abs() <= 50).mean() * 100

        m1, m2, m3, m4 = st.columns(4)
        for col, label, val, sub in [
            (m1, "Median Abs Error",  fmt(mae_med),        "On full dataset"),
            (m2, "Within ±20% Error", f"{within_20:.1f}%", "Of all predictions"),
            (m3, "Within ±50% Error", f"{within_50:.1f}%", "Of all predictions"),
            (m4, "Players Evaluated", f"{len(diag_df):,}", "Full dataset"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Run the full modelling pipeline to generate `player_data_with_predictions.xlsx`, then reload.")

# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#2a4a6a;font-size:0.8rem;padding:8px 0 20px 0;
            font-family:'Barlow Condensed',sans-serif;letter-spacing:0.08em;">
    TRANSFERIQ · FOOTBALL MARKET VALUE INTELLIGENCE · 2022–23 DATA · POWERED BY XGBOOST
</div>
""", unsafe_allow_html=True)