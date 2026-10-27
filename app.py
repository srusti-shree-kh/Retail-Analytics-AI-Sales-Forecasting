"""
RETAIL INTELLIGENCE  |  Executive Sales Analytics
-------------------------------------------------
Single-file Streamlit app. Place next to your CSV (any file named
Retail_Sales_Data*.csv) and run:   streamlit run app.py
"""
import re
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Retail Intelligence | Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESIGN TOKENS
# ============================================================
BG, SURFACE, BORDER, GRID = "#07090f", "#0d121c", "#1c2536", "#182131"
TXT, MUTED = "#e6ebf5", "#8693a8"
AMBER, TEAL, VIOLET, RED, GREEN, BLUE = "#f5a524", "#2dd4bf", "#8b7cf6", "#f4647a", "#4ade80", "#60a5fa"
SEQ = [TEAL, AMBER, VIOLET, RED, GREEN, BLUE, "#f472b6", "#fb923c"]
FONT = "Inter, Segoe UI, sans-serif"
NEW_ST = tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 50)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg:#07090f;--surface:#0d121c;--surface2:#111827;--border:#1c2536;--txt:#e6ebf5;--muted:#8693a8;
--amber:#f5a524;--teal:#2dd4bf;--violet:#8b7cf6;--red:#f4647a;--green:#4ade80;}
html,body,[class*="css"],.stApp{font-family:'Inter','Segoe UI',sans-serif;}
.stApp{background:radial-gradient(1200px 500px at 70% -10%,rgba(139,124,246,.10),transparent 60%),
radial-gradient(900px 400px at 0% 0%,rgba(45,212,191,.06),transparent 55%),var(--bg);color:var(--txt);}
header[data-testid="stHeader"]{background:transparent;}
footer,#MainMenu,.stDeployButton{visibility:hidden;}
.block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1500px;}
/* sidebar */
[data-testid="stSidebar"]{background:#080b12;border-right:1px solid var(--border);}
[data-testid="stSidebar"] .block-container{padding-top:1.4rem;}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:2px;}
.brand-dot{width:30px;height:30px;border-radius:9px;background:linear-gradient(135deg,var(--amber),#fb7185);
display:flex;align-items:center;justify-content:center;font-weight:800;color:#0a0a0a;font-size:15px;}
.brand-name{font-family:'Fraunces',serif;font-size:20px;font-weight:600;letter-spacing:.2px;}
.brand-sub{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1.6px;color:var(--muted);text-transform:uppercase;margin:2px 0 14px 40px;}
.side-label{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1.6px;color:var(--muted);text-transform:uppercase;margin:18px 0 6px;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:2px;}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:8px 12px;border-radius:9px;border:1px solid transparent;width:100%;transition:all .15s;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#101724;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(90deg,rgba(245,165,36,.16),rgba(245,165,36,.02));border-color:rgba(245,165,36,.35);}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child{display:none;}
[data-testid="stSidebar"] [role="radiogroup"] p{font-size:14px;font-weight:500;}
/* headings */
.eyebrow{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:2px;color:var(--amber);text-transform:uppercase;margin-bottom:6px;}
.h1{font-family:'Fraunces',serif;font-size:38px;font-weight:600;line-height:1.12;margin:0 0 8px;color:#f4f6fb;}
.h1 em{color:var(--amber);font-style:italic;font-weight:500;}
.lede{color:var(--muted);font-size:14.5px;max-width:860px;line-height:1.6;margin-bottom:18px;}
.hero{border:1px solid rgba(139,124,246,.35);border-radius:20px;padding:30px 34px 26px;margin-bottom:18px;
background:linear-gradient(120deg,rgba(139,124,246,.22) 0%,rgba(45,212,191,.10) 60%,rgba(13,18,28,.9) 100%);
box-shadow:0 10px 50px rgba(139,124,246,.12);}
.hero .h1{font-size:44px;}
.chip-row{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 0;}
/* kpis */
.kpi-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:6px 0 18px;}
@media(max-width:1250px){.kpi-grid{grid-template-columns:repeat(3,1fr);}}
@media(max-width:700px){.kpi-grid{grid-template-columns:repeat(2,1fr);}.h1{font-size:28px;}.hero .h1{font-size:30px;}}
.kpi{position:relative;background:linear-gradient(180deg,#111826,#0c111b);border:1px solid var(--border);border-top:2px solid var(--teal);
border-radius:14px;padding:14px 16px 12px;box-shadow:0 6px 24px rgba(0,0,0,.35);overflow:hidden;}
.kpi:nth-child(2){border-top-color:var(--violet);}.kpi:nth-child(3){border-top-color:var(--amber);}
.kpi:nth-child(4){border-top-color:var(--blue);}.kpi:nth-child(5){border-top-color:var(--red);}.kpi:nth-child(6){border-top-color:var(--green);}
.kpi-l{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase;}
.kpi-v{font-size:27px;font-weight:700;margin:6px 0 2px;letter-spacing:-.4px;}
.kpi-s{font-size:11.5px;color:var(--muted);}
.up{color:var(--green);}.down{color:var(--red);}.flat{color:var(--muted);}
.kpi svg{position:absolute;right:10px;bottom:8px;opacity:.9;}
/* cards */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] .ctitle){background:#0c111b;border:1px solid var(--border)!important;border-radius:14px;padding:6px 8px;}
.ctitle{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:#aab6cb;margin:4px 0 0;}
.csub{font-size:12px;color:var(--muted);margin:2px 0 4px;}
.insights{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:4px 0 18px;}
@media(max-width:1250px){.insights{grid-template-columns:repeat(2,1fr);}}
.ins{background:#0c111b;border:1px solid var(--border);border-top:3px solid var(--teal);border-radius:12px;padding:14px 16px;min-height:150px;}
.ins-t{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1.6px;text-transform:uppercase;margin-bottom:8px;}
.ins-b{font-size:13.5px;line-height:1.55;color:#d5dcea;}
.answer{border-left:3px solid var(--amber);padding:6px 0 6px 18px;margin:10px 0 14px;}
.answer .a-k{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase;}
.answer .a-h{font-family:'Fraunces',serif;font-size:30px;font-weight:600;margin:4px 0 2px;}
.answer .a-s{font-size:12.5px;color:var(--muted);}
.pill{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:10.5px;padding:2px 9px;border-radius:99px;border:1px solid;}
.pass{color:var(--green);border-color:rgba(74,222,128,.4);background:rgba(74,222,128,.08);}
.warn{color:var(--amber);border-color:rgba(245,165,36,.45);background:rgba(245,165,36,.08);}
.fail{color:var(--red);border-color:rgba(244,100,122,.45);background:rgba(244,100,122,.08);}
table.chk{width:100%;border-collapse:collapse;font-size:13px;}
table.chk th{text-align:left;font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase;padding:8px 10px;border-bottom:1px solid var(--border);}
table.chk td{padding:9px 10px;border-bottom:1px solid #131b29;color:#cfd7e6;}
.note{font-size:12.5px;color:var(--muted);line-height:1.6;}
.note b{color:#cfd7e6;}
/* widgets */
span[data-baseweb="tag"]{background:rgba(45,212,191,.14)!important;border:1px solid rgba(45,212,191,.4);color:#b8f5ec!important;border-radius:7px;}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,.stTextInput input{background:#0c111b!important;border-color:var(--border)!important;border-radius:10px!important;}
.stButton>button{background:#0c111b;border:1px solid var(--border);color:#c7d0e0;border-radius:99px;font-size:12px;padding:3px 14px;transition:all .15s;}
.stButton>button:hover{border-color:var(--amber);color:var(--amber);}
.stDownloadButton>button{background:linear-gradient(90deg,var(--amber),#fb923c);color:#111;font-weight:600;border:none;border-radius:10px;}
[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:12px;overflow:hidden;}
div[role="radiogroup"][aria-orientation="horizontal"] label{background:#0c111b;border:1px solid var(--border);border-radius:99px;padding:3px 14px;margin-right:6px;}
div[role="radiogroup"][aria-orientation="horizontal"] label:has(input:checked){border-color:var(--teal);background:rgba(45,212,191,.12);}
hr{border-color:var(--border);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def cr(v, d=0):
    return f"₹{v / 1e7:,.{d}f} Cr"


def lakh(v):
    return f"₹{v / 1e5:,.2f} L"


def pct(v, d=1, sign=False):
    if v is None or pd.isna(v) or np.isinf(v):
        return "—"
    return f"{v:+,.{d}f}%" if sign else f"{v:,.{d}f}%"


def show(fig):
    cfg = {"displayModeBar": False}
    if NEW_ST:
        st.plotly_chart(fig, width="stretch", config=cfg)
    else:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def table(df, **kw):
    if NEW_ST:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def style(fig, h=340, legend=False, **kw):
    ax = dict(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED, size=11))
    fig.update_layout(
        height=h, margin=dict(l=8, r=8, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TXT, size=12), showlegend=legend,
        hoverlabel=dict(bgcolor="#141b29", bordercolor=BORDER, font=dict(color=TXT)),
        xaxis=ax, yaxis=ax,
        legend=dict(orientation="h", y=1.12, x=0, font=dict(color="#c3cde0")),
    )
    fig.update_layout(**kw)
    return fig


def head(eyebrow, title, lede=""):
    st.markdown(
        f"<div class='eyebrow'>{eyebrow}</div><div class='h1'>{title}</div><div class='lede'>{lede}</div>",
        unsafe_allow_html=True,
    )


def ctitle(title, sub=""):
    st.markdown(f"<div class='ctitle'>{title}</div><div class='csub'>{sub}</div>", unsafe_allow_html=True)


def spark(values, color):
    v = np.asarray(values, dtype=float)
    if len(v) < 2 or np.nanmax(v) == np.nanmin(v):
        return ""
    x = np.linspace(2, 78, len(v))
    y = 24 - (v - v.min()) / (v.max() - v.min()) * 20
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    return f"<svg width='80' height='28' viewBox='0 0 80 28'><polyline fill='none' stroke='{color}' stroke-width='1.8' points='{pts}'/></svg>"


def delta_html(cur, prev, suffix="vs prior year", invert=False):
    if prev is None or prev == 0 or pd.isna(prev):
        return "<span class='flat'>no prior-year period</span>"
    ch = (cur / prev - 1) * 100
    good = ch >= 0 if not invert else ch <= 0
    cls = "up" if good else "down"
    arrow = "▲" if ch >= 0 else "▼"
    return f"<span class='{cls}'>{arrow} {abs(ch):.1f}%</span> <span class='flat'>{suffix}</span>"


# ============================================================
# DATA
# ============================================================
REQUIRED = ["Date", "Store_Location", "Product_Category", "Revenue", "Total_Sales", "Units_Sold", "Region"]


@st.cache_data(show_spinner="Loading and preparing retail data...")
def load_data(file_bytes=None):
    if file_bytes is not None:
        raw = pd.read_csv(BytesIO(file_bytes))
    else:
        folders = [Path(__file__).parent, Path.cwd()]
        found = None
        for fo in folders:
            hits = sorted(fo.glob("Retail_Sales_Data*.csv")) or sorted(fo.glob("*.csv"))
            if hits:
                found = hits[0]
                break
        if found is None:
            return None
        raw = pd.read_csv(found)
    miss = [c for c in REQUIRED if c not in raw.columns]
    if miss:
        raise ValueError("Dataset is missing required columns: " + ", ".join(miss))
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date", "Revenue"]).copy()
    for c in ["Revenue", "Total_Sales", "Units_Sold", "Unit_Price", "Discount_Percentage",
              "Stock_On_Hand", "Store_Rating", "Holiday_Flag"]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    for c in raw.select_dtypes(include=["object", "string"]).columns:
        raw[c] = raw[c].astype(str).str.strip()
    for c, default in [("Brand", "Unknown"), ("Product_Subcategory", "Other"), ("Customer_Type", "Unknown"),
                       ("Payment_Mode", "Unknown"), ("Promotion_Applied", "No")]:
        if c not in raw.columns:
            raw[c] = default
    if "Discount_Percentage" not in raw.columns:
        raw["Discount_Percentage"] = ((1 - raw["Revenue"] / raw["Total_Sales"]) * 100).round()
    if "Store_Rating" not in raw.columns:
        raw["Store_Rating"] = np.nan
    raw["Year"] = raw["Date"].dt.year
    raw["Quarter"] = raw["Date"].dt.quarter
    raw["Month_Start"] = raw["Date"].dt.to_period("M").dt.to_timestamp()
    raw["Month_Label"] = raw["Date"].dt.strftime("%b %Y")
    raw["Discount_Amount"] = raw["Total_Sales"] - raw["Revenue"]
    return raw


def kpis(f):
    n = len(f)
    days = max(f["Date"].nunique(), 1)
    rev = float(f["Revenue"].sum())
    gross = float(f["Total_Sales"].sum())
    return dict(rev=rev, gross=gross, tx=n, units=int(f["Units_Sold"].sum()),
                disc=float(f["Discount_Amount"].sum()), ticket=rev / n if n else 0,
                per_day=rev / days, days=days)


# ============================================================
# SIDEBAR  (brand, navigation, filters, data source)
# ============================================================
st.sidebar.markdown(
    "<div class='brand'><div class='brand-dot'>R</div><div class='brand-name'>Retail Intelligence</div></div>"
    "<div class='brand-sub'>Sales analytics observatory</div>",
    unsafe_allow_html=True,
)
PAGES = ["01  Briefing", "02  Pulse", "03  Mix", "04  Stores", "05  Forecast", "06  Discount Lab", "07  Integrity & Method"]
st.sidebar.markdown("<div class='side-label'>Navigate</div>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
try:
    df = load_data(uploaded.getvalue()) if uploaded else load_data()
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()
if df is None:
    st.markdown("<div class='h1'>Retail Intelligence</div>", unsafe_allow_html=True)
    st.warning("No CSV found. Upload your dataset in the sidebar, or place a file named "
               "'Retail_Sales_Data_Unlox.csv' in the same folder as app.py.")
    st.stop()

st.sidebar.markdown("<div class='side-label'>Filters</div>", unsafe_allow_html=True)
dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
drange = st.sidebar.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
if not isinstance(drange, (tuple, list)) or len(drange) != 2:
    st.info("Select both a start and an end date in the sidebar.")
    st.stop()
d0, d1 = pd.Timestamp(drange[0]), pd.Timestamp(drange[1])


def opts(col):
    return sorted(df[col].dropna().astype(str).unique().tolist())


sel_region = st.sidebar.multiselect("Region", opts("Region"), default=opts("Region"))
sel_store = st.sidebar.multiselect("Store", opts("Store_Location"), default=opts("Store_Location"))
sel_cat = st.sidebar.multiselect("Category", opts("Product_Category"), default=opts("Product_Category"))
sel_brand = st.sidebar.multiselect("Brand", opts("Brand"), default=opts("Brand"))

base = df[df["Region"].isin(sel_region) & df["Store_Location"].isin(sel_store)
          & df["Product_Category"].isin(sel_cat) & df["Brand"].isin(sel_brand)]
f = base[(base["Date"] >= d0) & (base["Date"] <= d1)]
if f.empty:
    st.warning("No rows match the current filters. Widen the selection in the sidebar.")
    st.stop()
p0, p1 = d0 - pd.DateOffset(years=1), d1 - pd.DateOffset(years=1)
prev = base[(base["Date"] >= p0) & (base["Date"] <= p1)] if p0 >= df["Date"].min() else base.iloc[0:0]

st.sidebar.markdown(
    f"<div class='note' style='margin-top:18px'>Source<br><b>{len(df):,}</b> rows · {dmin:%d %b %Y} – {dmax:%d %b %Y}<br>"
    f"In scope: <b>{len(f):,}</b> rows</div>", unsafe_allow_html=True)

K = kpis(f)
KP = kpis(prev) if len(prev) else None
n_months = f["Month_Start"].nunique()
scope_txt = f"{d0:%d %b %Y} – {d1:%d %b %Y} · {len(sel_region)} region(s) · {len(sel_store)} store(s) · {len(sel_cat)} categor{'y' if len(sel_cat)==1 else 'ies'}"


# ============================================================
# ASK-THE-DATA ENGINE
# ============================================================
DIMS = [("subcategory", "Product_Subcategory", "subcategory"), ("category", "Product_Category", "category"),
        ("store", "Store_Location", "store"), ("city", "Store_Location", "store"), ("location", "Store_Location", "store"),
        ("brand", "Brand", "brand"), ("region", "Region", "region"), ("month", "Month_Label", "month"),
        ("payment", "Payment_Mode", "payment mode"), ("quarter", "Quarter", "quarter"), ("year", "Year", "year")]
METRIC_LABEL = {"revenue": "Net revenue", "units": "Units sold", "ticket": "Average ticket",
                "transactions": "Transactions", "discount": "Discount given"}


def fmt_metric(m, v):
    if m in ("revenue", "discount"):
        return cr(v, 2) if v < 1e9 else cr(v, 0)
    if m == "ticket":
        return lakh(v)
    return f"{v:,.0f}"


def answer_question(q, data):
    ql = q.lower().strip()
    sub, scope, used = data, [], set()
    for y in sorted(data["Year"].unique()):
        if re.search(rf"\b{y}\b", ql):
            sub = sub[sub["Year"] == y]; scope.append(str(y)); used.add("Year")
    mq = re.search(r"\bq([1-4])\b", ql)
    if mq:
        sub = sub[sub["Quarter"] == int(mq.group(1))]; scope.append(f"Q{mq.group(1)}"); used.add("Quarter")
    for col, label in [("Product_Category", "category"), ("Region", "region"), ("Store_Location", "store"),
                       ("Brand", "brand"), ("Product_Subcategory", "subcategory"), ("Payment_Mode", "payment")]:
        for val in data[col].dropna().unique():
            if re.search(rf"\b{re.escape(str(val).lower())}\b", ql):
                sub = sub[sub[col] == val]; scope.append(str(val)); used.add(col)
    if "forecast" in ql:
        return dict(h="Open the Forecast tab", s="Forecasts are built on the 05 Forecast page, with a model back-test.", t=None, m="revenue")
    metric = "revenue"
    if "unit" in ql: metric = "units"
    elif re.search(r"ticket|average|avg|per transaction", ql): metric = "ticket"
    elif re.search(r"transaction|orders|how many sales", ql): metric = "transactions"
    elif "discount" in ql: metric = "discount"
    dim = None
    if re.search(r"returning|\bnew\b|customer", ql):
        dim = "Customer_Type"
    for key, col, _ in DIMS:
        if re.search(rf"\b{key}s?\b", ql) and col not in used:
            dim = col; break
    if sub.empty:
        return dict(h="No data for that question", s="The filters in your question matched no rows.", t=None, m=metric)
    agg = {"revenue": ("Revenue", "sum"), "units": ("Units_Sold", "sum"), "ticket": ("Revenue", "mean"),
           "transactions": ("Revenue", "size"), "discount": ("Discount_Amount", "sum")}[metric]
    scope_s = " · ".join(scope) if scope else "all data"
    if dim is None:
        v = {"revenue": sub["Revenue"].sum(), "units": sub["Units_Sold"].sum(), "ticket": sub["Revenue"].mean(),
             "transactions": len(sub), "discount": sub["Discount_Amount"].sum()}[metric]
        return dict(h=fmt_metric(metric, v), s=f"{METRIC_LABEL[metric]} · {scope_s} · {len(sub):,} transactions", t=None, m=metric)
    g = sub.groupby(dim)[agg[0]].agg(agg[1]).reset_index(name="v")
    asc = bool(re.search(r"worst|lowest|least|bottom|min", ql))
    g = g.sort_values("v", ascending=asc).head(12)
    top = g.iloc[0]
    word = "Lowest" if asc else "Top"
    return dict(h=f"{word}: {top[dim]} — {fmt_metric(metric, top['v'])}",
                s=f"{METRIC_LABEL[metric]} by {dim.replace('_', ' ').lower()} · {scope_s}", t=g.rename(columns={dim: "k"}), m=metric)


# ============================================================
# PAGE 01  BRIEFING
# ============================================================
def compute_insights():
    out = []
    # 1 Growth
    yr = f.groupby("Year")["Revenue"].sum()
    if len(yr) >= 2:
        a, b = yr.index[-2], yr.index[-1]
        g = (yr[b] / yr[a] - 1) * 100
        mm = f.pivot_table(index=f["Date"].dt.month, columns="Year", values="Revenue", aggfunc="sum")
        both = mm[[a, b]].dropna()
        ahead = int((both[b] > both[a]).sum())
        tone = "Revenue is broadly flat" if abs(g) < 2 else ("Revenue is growing" if g > 0 else "Revenue is shrinking")
        out.append(("GROWTH", TEAL, f"<b>{tone}:</b> {b} vs {a} is <b>{g:+.1f}%</b> ({cr(yr[a])} → {cr(yr[b])}). "
                    f"{b} was ahead of {a} in <b>{ahead} of {len(both)}</b> comparable months."))
    else:
        out.append(("GROWTH", TEAL, f"Only one year is in scope, so year-over-year growth cannot be measured. Revenue is <b>{cr(K['rev'])}</b>."))
    # 2 Margin leak
    leak = K["disc"] / K["gross"] * 100 if K["gross"] else 0
    ctr = f[["Discount_Percentage", "Units_Sold"]].corr().iloc[0, 1]
    verdict = "no evidence that discounting buys volume" if abs(ctr) < 0.05 else "a measurable link between discount and volume"
    out.append(("MARGIN LEAK", RED, f"Discounts cost <b>{cr(K['disc'])}</b> (<b>{leak:.1f}%</b> of gross sales). "
                f"Discount-to-units correlation is <b>{ctr:+.3f}</b>: {verdict}."))
    # 3 Forecastability
    d = f.groupby("Date")["Revenue"].sum().asfreq("D", fill_value=0)
    if len(d) > 60:
        l7, l1 = d.autocorr(7), d.autocorr(1)
        if max(abs(l7), abs(l1)) < 0.15:
            msg = (f"Daily revenue is close to <b>noise around a level</b> (lag-1 autocorr {l1:+.2f}, lag-7 {l7:+.2f}). "
                   "Plan on totals over 30+ days, not single days.")
        else:
            msg = f"Daily revenue has structure (lag-1 {l1:+.2f}, lag-7 {l7:+.2f}); a model can add value over a flat average."
    else:
        msg = "Too few days in scope to judge how predictable revenue is."
    out.append(("FORECASTABILITY", AMBER, msg))
    # 4 Stores
    s = f.groupby("Store_Location")["Revenue"].sum().sort_values(ascending=False)
    if len(s) >= 2:
        spread = (s.iloc[0] - s.iloc[-1]) / s.mean() * 100
        out.append(("STORES", VIOLET, f"<b>{s.index[0]}</b> leads at {cr(s.iloc[0])}; <b>{s.index[-1]}</b> trails at {cr(s.iloc[-1])}. "
                    f"Gap between best and worst store is {spread:.1f}% of the average store."))
    else:
        out.append(("STORES", VIOLET, f"Only <b>{s.index[0]}</b> is in scope."))
    return out


def page_briefing():
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    ns = f["Store_Location"].nunique()
    st.markdown(
        f"<div class='hero'><div class='eyebrow'>Retail Intelligence · Briefing</div>"
        f"<div class='h1'>{cr(K['rev'])} in {n_months} months, {words.get(ns, ns)} store{'s' if ns != 1 else ''} — and <em>one honest forecast.</em></div>"
        f"<div class='lede' style='margin-bottom:0'>{K['tx']:,} transactions · {scope_txt}. This dashboard cleans the data, tests whether the future is "
        f"predictable, lets forecasting models compete, segments your stores and answers questions in plain English.</div></div>",
        unsafe_allow_html=True)

    # Ask the data
    q = st.text_input("Ask the data", key="ask_q", placeholder="e.g. best store in Q4 2024 by revenue", label_visibility="collapsed")
    chips = ["best store in Q4 2024 by revenue", "revenue of Fashion in 2023", "which category sells most units?",
             "how much discount did we give in Mumbai?", "worst month by revenue", "returning vs new customers", "average ticket by store"]
    cols = st.columns(len(chips))
    for c, s in zip(cols, chips):
        c.button(s, key="chip_" + s, on_click=lambda s=s: st.session_state.update(ask_q=s), use_container_width=True)
    if q:
        ans = answer_question(q, df)
        st.markdown(f"<div class='answer'><div class='a-k'>Answer</div><div class='a-h'>{ans['h']}</div><div class='a-s'>{ans['s']}</div></div>",
                    unsafe_allow_html=True)
        if ans["t"] is not None and len(ans["t"]) > 1:
            t = ans["t"].sort_values("v", ascending=True)
            fig = go.Figure(go.Bar(x=t["v"], y=t["k"].astype(str), orientation="h",
                                   marker=dict(color=t["v"], colorscale=[[0, "#6b4a0f"], [1, AMBER]]),
                                   text=[fmt_metric(ans["m"], v) for v in t["v"]], textposition="outside", textfont=dict(color=MUTED, size=11)))
            style(fig, h=max(180, 30 * len(t)), xaxis=dict(visible=False), yaxis=dict(gridcolor="rgba(0,0,0,0)"))
            show(fig)

    # KPI cards
    mon = f.groupby("Month_Start").agg(rev=("Revenue", "sum"), tx=("Revenue", "size"), units=("Units_Sold", "sum"), disc=("Discount_Amount", "sum"))
    mon["ticket"] = mon["rev"] / mon["tx"]
    mon["perday"] = f.groupby("Month_Start").apply(lambda x: x["Revenue"].sum() / max(x["Date"].nunique(), 1)) if False else mon["rev"] / 30
    P = KP
    cards = [
        ("Net revenue", cr(K["rev"]), delta_html(K["rev"], P and P["rev"]), mon["rev"], TEAL),
        ("Transactions", f"{K['tx']:,}", delta_html(K["tx"], P and P["tx"]), mon["tx"], VIOLET),
        ("Average ticket", lakh(K["ticket"]), delta_html(K["ticket"], P and P["ticket"]), mon["ticket"], AMBER),
        ("Revenue per day", cr(K["per_day"], 2), f"<span class='flat'>{K['days']:,} days in scope</span>", mon["perday"], BLUE),
        ("Discount given", cr(K["disc"]), f"<span class='down'>{K['disc'] / K['gross'] * 100:.1f}%</span> <span class='flat'>of gross sales</span>", mon["disc"], RED),
        ("Units sold", f"{K['units']:,}", delta_html(K["units"], P and P["units"]), mon["units"], GREEN),
    ]
    html = "".join(f"<div class='kpi'><div class='kpi-l'>{l}</div><div class='kpi-v'>{v}</div><div class='kpi-s'>{s}</div>{spark(sp, c)}</div>"
                   for l, v, s, sp, c in cards)
    st.markdown(f"<div class='kpi-grid'>{html}</div>", unsafe_allow_html=True)

    # Insights
    st.markdown("<div class='eyebrow'>The brief · computed from the data in scope</div>", unsafe_allow_html=True)
    ins = "".join(f"<div class='ins' style='border-top-color:{c}'><div class='ins-t' style='color:{c}'>{t}</div><div class='ins-b'>{b}</div></div>"
                  for t, c, b in compute_insights())
    st.markdown(f"<div class='insights'>{ins}</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.6, 1])
    with c1.container(border=True):
        ctitle("Monthly revenue", "Net revenue with 3-month moving average")
        m = mon["rev"].reset_index()
        m["ma"] = m["rev"].rolling(3, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m["Month_Start"], y=m["rev"] / 1e7, name="Revenue", mode="lines+markers", line=dict(color=TEAL, width=2.6),
                                 marker=dict(size=6), fill="tozeroy", fillcolor="rgba(45,212,191,.10)", hovertemplate="%{x|%b %Y}<br>₹%{y:,.1f} Cr<extra></extra>"))
        fig.add_trace(go.Scatter(x=m["Month_Start"], y=m["ma"] / 1e7, name="3-mo avg", line=dict(color=AMBER, width=2, dash="dot"), hoverinfo="skip"))
        style(fig, 330, legend=True, yaxis=dict(title="₹ Cr", rangemode="tozero", gridcolor=GRID))
        show(fig)
    with c2.container(border=True):
        ctitle("Revenue by store", "Ranked, ₹ Cr")
        s = f.groupby("Store_Location")["Revenue"].sum().sort_values().reset_index()
        fig = go.Figure(go.Bar(x=s["Revenue"] / 1e7, y=s["Store_Location"], orientation="h",
                               marker=dict(color=s["Revenue"], colorscale=[[0, "#3b2f7a"], [1, VIOLET]]),
                               text=[f"{v/1e7:,.0f}" for v in s["Revenue"]], textposition="inside", textfont=dict(color="#fff", size=11)))
        style(fig, 330, xaxis=dict(visible=False), yaxis=dict(gridcolor="rgba(0,0,0,0)"))
        show(fig)

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        ctitle("Category contribution", "Share of net revenue")
        c = f.groupby("Product_Category")["Revenue"].sum().reset_index()
        fig = go.Figure(go.Pie(labels=c["Product_Category"], values=c["Revenue"], hole=0.64, sort=True,
                               marker=dict(colors=SEQ, line=dict(color=BG, width=3)), textinfo="percent", textfont=dict(color="#fff", size=12)))
        fig.add_annotation(text=f"<b>{cr(K['rev'])}</b><br><span style='font-size:11px;color:{MUTED}'>net revenue</span>", showarrow=False, font=dict(size=18, color=TXT))
        style(fig, 320, legend=True, legend_orientation="v", legend_y=0.5, legend_x=1.0)
        show(fig)
    with c2.container(border=True):
        ctitle("Regional split", "Revenue by region and year")
        r = f.groupby(["Region", "Year"])["Revenue"].sum().reset_index()
        r["Year"] = r["Year"].astype(str)
        fig = px.bar(r, x="Region", y=r["Revenue"] / 1e7, color="Year", barmode="group", color_discrete_sequence=[VIOLET, TEAL, AMBER])
        fig.update_traces(marker_line_width=0)
        style(fig, 320, legend=True, yaxis=dict(title="₹ Cr", gridcolor=GRID), xaxis=dict(title=""))
        show(fig)


# ============================================================
# PAGE 02  PULSE
# ============================================================
def page_pulse():
    head("02 · Pulse", "How the business breathes",
         "Daily, weekly and monthly performance for the current filters. Everything here recalculates instantly when you change the sidebar.")
    c1, c2 = st.columns([1, 1])
    gran = c1.radio("Granularity", ["Daily", "Weekly", "Monthly"], horizontal=True, index=1)
    metric = c2.radio("Metric", ["Revenue", "Transactions", "Units"], horizontal=True)
    col = {"Revenue": "Revenue", "Transactions": "Revenue", "Units": "Units_Sold"}[metric]
    how = "size" if metric == "Transactions" else "sum"
    rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[gran]
    s = f.set_index("Date")[col].resample(rule).agg(how)
    div, unit = (1e7, "₹ Cr") if metric == "Revenue" else (1, metric.lower())
    with st.container(border=True):
        ctitle(f"{gran} {metric.lower()}", "Teal = actual, amber = 28-day average (daily view only)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s / div, mode="lines", line=dict(color=TEAL, width=1.6 if gran == "Daily" else 2.4),
                                 fill="tozeroy", fillcolor="rgba(45,212,191,.09)", name=metric))
        if gran == "Daily":
            fig.add_trace(go.Scatter(x=s.index, y=s.rolling(28, min_periods=7).mean() / div, mode="lines",
                                     line=dict(color=AMBER, width=2), name="28-day avg"))
        style(fig, 380, legend=True, yaxis=dict(title=unit, gridcolor=GRID), hovermode="x unified")
        show(fig)

    a, b = st.columns(2)
    with a.container(border=True):
        ctitle("Year over year · monthly revenue", "₹ Cr by calendar month")
        m = f.assign(M=f["Date"].dt.month).groupby(["Year", "M"])["Revenue"].sum().reset_index()
        fig = go.Figure()
        for i, (y, g) in enumerate(m.groupby("Year")):
            fig.add_trace(go.Scatter(x=g["M"], y=g["Revenue"] / 1e7, name=str(y), mode="lines+markers", line=dict(color=SEQ[i], width=2.4)))
        style(fig, 300, legend=True, xaxis=dict(tickmode="array", tickvals=list(range(1, 13)),
              ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], gridcolor=GRID), yaxis=dict(gridcolor=GRID, title="₹ Cr"))
        show(fig)
    with b.container(border=True):
        ctitle("Seasonality check", "Average daily revenue by month vs overall average")
        d = f.groupby("Date")["Revenue"].sum().asfreq("D", fill_value=0)
        mo = d.groupby(d.index.month).mean() / d.mean() * 100 - 100
        names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig = go.Figure(go.Bar(x=[names[i - 1] for i in mo.index], y=mo.values, marker_color=[TEAL if v >= 0 else RED for v in mo.values],
                               text=[f"{v:+.1f}%" for v in mo.values], textposition="outside", textfont=dict(color=MUTED, size=10)))
        style(fig, 300, yaxis=dict(title="% vs average", gridcolor=GRID, zeroline=True, zerolinecolor=BORDER))
        show(fig)
        groups = [g.values for _, g in d.groupby(d.index.month)]
        if len(groups) > 2:
            p = stats.f_oneway(*groups).pvalue
            msg = "Month effect is statistically detectable." if p < 0.05 else "Month-to-month bars wobble within what chance produces — no proven seasonality."
            st.markdown(f"<div class='note'><b>ANOVA p = {p:.2f}.</b> {msg}</div>", unsafe_allow_html=True)

    with st.container(border=True):
        ctitle("Calendar heatmap", "Each square is a day · amber = above typical, teal = below typical")
        d = f.groupby("Date")["Revenue"].sum().asfreq("D", fill_value=0)
        years = sorted(d.index.year.unique())
        fig = make_subplots(rows=len(years), cols=1, subplot_titles=[str(y) for y in years], vertical_spacing=0.14)
        mean = d.mean()
        for r, y in enumerate(years, start=1):
            sy = d[d.index.year == y]
            jan1 = pd.Timestamp(y, 1, 1)
            wk = ((sy.index - jan1).days + jan1.weekday()) // 7
            z = np.full((7, 54), np.nan)
            txt = np.full((7, 54), "", dtype=object)
            z[sy.index.weekday, wk] = np.clip(sy.values / mean - 1, -0.7, 0.7)
            for dt, w, v in zip(sy.index, wk, sy.values):
                txt[dt.weekday(), w] = f"{dt:%d %b %Y}<br>₹{v/1e7:,.2f} Cr"
            ticks = [((pd.Timestamp(y, m, 1) - jan1).days + jan1.weekday()) // 7 for m in range(1, 13)]
            fig.add_trace(go.Heatmap(z=z, text=txt, hoverinfo="text", xgap=2, ygap=2, showscale=False, zmin=-0.7, zmax=0.7,
                                     colorscale=[[0, "#0f766e"], [0.5, "#161d2b"], [1, AMBER]]), row=r, col=1)
            fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                             showgrid=False, zeroline=False, row=r, col=1, tickfont=dict(color=MUTED, size=10))
            fig.update_yaxes(tickmode="array", tickvals=[0, 2, 4], ticktext=["Mon", "Wed", "Fri"], autorange="reversed", showgrid=False,
                             zeroline=False, row=r, col=1, tickfont=dict(color=MUTED, size=10))
        style(fig, 210 * len(years) + 20)
        for a in fig.layout.annotations:
            a.font = dict(color="#c3cde0", size=13)
            a.x = 0; a.xanchor = "left"
        show(fig)


# ============================================================
# PAGE 03  MIX
# ============================================================
def page_mix():
    head("03 · Mix", "What sells, to whom, and how",
         "Category and subcategory share, customer behaviour, payment preferences and how concentrated revenue is across brands.")
    a, b = st.columns([1.1, 1])
    with a.container(border=True):
        ctitle("Revenue sunburst", "Category → subcategory")
        g = f.groupby(["Product_Category", "Product_Subcategory"])["Revenue"].sum().reset_index()
        fig = px.sunburst(g, path=["Product_Category", "Product_Subcategory"], values="Revenue", color_discrete_sequence=SEQ)
        fig.update_traces(insidetextorientation="radial", marker=dict(line=dict(color=BG, width=2)), textfont=dict(size=12),
                          hovertemplate="%{label}<br>₹%{value:,.0f}<extra></extra>")
        style(fig, 470)
        show(fig)
    with b.container(border=True):
        ctitle("Customers, payments & promotions", "Share of transactions and average ticket")
        rows = []
        for title, col in [("Customer type", "Customer_Type"), ("Payment mode", "Payment_Mode"), ("Promotion applied", "Promotion_Applied")]:
            g = f.groupby(col).agg(tx=("Revenue", "size"), t=("Revenue", "mean")).reset_index()
            g["share"] = g["tx"] / g["tx"].sum() * 100
            rows.append((title, g.sort_values("share", ascending=False)))
        fig = make_subplots(rows=3, cols=1, subplot_titles=[r[0] for r in rows], vertical_spacing=0.13,
                            row_heights=[len(r[1]) for r in rows])
        for i, (_, g) in enumerate(rows, start=1):
            fig.add_trace(go.Bar(x=g["share"], y=g.iloc[:, 0], orientation="h", marker_color=[AMBER, TEAL, VIOLET, RED][: len(g)],
                                 text=[f"{s:.1f}% · {lakh(t)}" for s, t in zip(g["share"], g["t"])], textposition="outside",
                                 textfont=dict(color=MUTED, size=11)), row=i, col=1)
            fig.update_xaxes(visible=False, range=[0, 100], row=i, col=1)
            fig.update_yaxes(autorange="reversed", row=i, col=1, gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#c3cde0"))
        style(fig, 470)
        for an in fig.layout.annotations:
            an.font = dict(color="#aab6cb", size=11); an.x = 0; an.xanchor = "left"
        show(fig)

    a, b = st.columns([1.3, 1])
    br = f.groupby("Brand")["Revenue"].sum().sort_values(ascending=False).reset_index()
    br["cum"] = br["Revenue"].cumsum() / br["Revenue"].sum() * 100
    with a.container(border=True):
        ctitle("Brand concentration (Pareto)", "Bars = revenue, line = cumulative share")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=br["Brand"], y=br["Revenue"] / 1e7, marker_color=VIOLET, name="₹ Cr"), secondary_y=False)
        fig.add_trace(go.Scatter(x=br["Brand"], y=br["cum"], mode="lines+markers", line=dict(color=AMBER, width=2.4), name="Cumulative %"), secondary_y=True)
        style(fig, 340, legend=True)
        fig.update_yaxes(title_text="₹ Cr", secondary_y=False, gridcolor=GRID)
        fig.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True, showgrid=False)
        show(fig)
        top5 = br["cum"].iloc[min(4, len(br) - 1)]
        st.markdown(f"<div class='note'>Top 5 brands hold <b>{top5:.1f}%</b> of revenue across <b>{len(br)}</b> brands. "
                    f"{'Revenue is spread evenly — no single-brand dependency.' if top5 < 40 else 'Revenue leans on a few brands.'}</div>", unsafe_allow_html=True)
    with b.container(border=True):
        ctitle("Brand leaderboard", "Top 10 by net revenue")
        t = f.groupby("Brand").agg(Revenue=("Revenue", "sum"), Units=("Units_Sold", "sum"), Ticket=("Revenue", "mean")).reset_index()
        t["Revenue (₹ Cr)"] = t["Revenue"] / 1e7
        t["Ticket (₹ L)"] = t["Ticket"] / 1e5
        t = t.sort_values("Revenue", ascending=False).head(10)
        table(t[["Brand", "Revenue (₹ Cr)", "Units", "Ticket (₹ L)"]], column_config={
            "Revenue (₹ Cr)": st.column_config.ProgressColumn("Revenue (₹ Cr)", format="%.1f", min_value=0, max_value=float(t["Revenue (₹ Cr)"].max())),
            "Ticket (₹ L)": st.column_config.NumberColumn(format="%.2f")})

    with st.container(border=True):
        ctitle("Category × year", "How each category moved")
        c = f.groupby(["Product_Category", "Year"])["Revenue"].sum().reset_index()
        c["Year"] = c["Year"].astype(str)
        fig = px.bar(c, x="Product_Category", y=c["Revenue"] / 1e7, color="Year", barmode="group", color_discrete_sequence=[VIOLET, TEAL, AMBER])
        style(fig, 300, legend=True, yaxis=dict(title="₹ Cr", gridcolor=GRID), xaxis=dict(title=""))
        show(fig)


# ============================================================
# PAGE 04  STORES
# ============================================================
def store_table(data):
    g = data.groupby("Store_Location")
    t = pd.DataFrame({
        "Region": g["Region"].first(),
        "Revenue_Cr": g["Revenue"].sum() / 1e7,
        "Ticket": g["Revenue"].mean(),
        "Rating": g["Store_Rating"].mean(),
        "Returning_%": g["Customer_Type"].apply(lambda x: (x == "Returning").mean() * 100),
        "Promo_%": g["Promotion_Applied"].apply(lambda x: (x == "Yes").mean() * 100),
        "Avg_Discount_%": g["Discount_Percentage"].mean(),
        "Units_per_Sale": g["Units_Sold"].mean(),
    })
    yr = data.pivot_table(index="Store_Location", columns="Year", values="Revenue", aggfunc="sum")
    if yr.shape[1] >= 2:
        t["YoY_%"] = (yr.iloc[:, -1] / yr.iloc[:, -2].replace(0, np.nan) - 1) * 100
    else:
        t["YoY_%"] = 0.0
    t["YoY_%"] = t["YoY_%"].fillna(0)
    t["Rating"] = t["Rating"].fillna(t["Rating"].mean() if t["Rating"].notna().any() else 0)
    t["vs_Avg_%"] = (t["Revenue_Cr"] / t["Revenue_Cr"].mean() - 1) * 100
    comp = t[["Revenue_Cr", "YoY_%", "Rating", "Ticket", "Returning_%"]].rank(pct=True)
    t["Health"] = (comp.mean(axis=1) * 100).round(0)
    return t


def page_stores():
    head("04 · Stores", "The store constellation",
         "Every store compared on revenue, growth, ticket size, rating and discount discipline. Segments come from K-Means on behavioural features.")
    t = store_table(f)
    with st.container(border=True):
        ctitle("Store × category heatmap", "Revenue index vs the average cell (1.00 = average)")
        p = f.pivot_table(index="Store_Location", columns="Product_Category", values="Revenue", aggfunc="sum").fillna(0)
        idx = p / p.values.mean()
        fig = go.Figure(go.Heatmap(z=idx.values, x=idx.columns, y=idx.index, colorscale=[[0, "#4c1d95"], [0.5, "#0f766e"], [1, "#facc15"]],
                                   text=np.round(idx.values, 2), texttemplate="%{text}", textfont=dict(size=12, color="#fff"), xgap=3, ygap=3,
                                   colorbar=dict(thickness=10, tickfont=dict(color=MUTED)), hovertemplate="%{y} · %{x}<br>index %{z:.2f}<extra></extra>"))
        style(fig, 380, yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"), xaxis=dict(gridcolor="rgba(0,0,0,0)", side="top"))
        show(fig)

    a, b = st.columns([1.35, 1])
    with a.container(border=True):
        ctitle("Store ranking", "Health score blends revenue, growth, rating, ticket and returning-customer share")
        tt = t.reset_index().sort_values("Revenue_Cr", ascending=False)
        table(tt[["Store_Location", "Region", "Revenue_Cr", "vs_Avg_%", "YoY_%", "Rating", "Ticket", "Health"]].rename(columns={"Store_Location": "Store"}),
              column_config={
                  "Revenue_Cr": st.column_config.NumberColumn("Revenue (₹ Cr)", format="%.1f"),
                  "vs_Avg_%": st.column_config.NumberColumn("vs avg", format="%+.1f%%"),
                  "YoY_%": st.column_config.NumberColumn("YoY", format="%+.1f%%"),
                  "Rating": st.column_config.NumberColumn(format="%.2f ★"),
                  "Ticket": st.column_config.NumberColumn("Ticket (₹)", format="%,.0f"),
                  "Health": st.column_config.ProgressColumn("Health", min_value=0, max_value=100, format="%d")})
    with b.container(border=True):
        ctitle("Store profile radar", "Min-max scaled across stores · dashed = network average")
        pick = st.selectbox("Store", t.index.tolist(), label_visibility="collapsed")
        axes = ["Revenue_Cr", "YoY_%", "Rating", "Ticket", "Returning_%", "Units_per_Sale"]
        labels = ["Revenue", "Growth", "Rating", "Ticket", "Returning", "Basket"]
        sc = (t[axes] - t[axes].min()) / (t[axes].max() - t[axes].min()).replace(0, 1)
        sc = sc.fillna(0.5)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=list(sc.mean()) + [sc.mean().iloc[0]], theta=labels + [labels[0]], line=dict(color=MUTED, dash="dash"), name="Network avg"))
        fig.add_trace(go.Scatterpolar(r=list(sc.loc[pick]) + [sc.loc[pick].iloc[0]], theta=labels + [labels[0]], fill="toself",
                                      fillcolor="rgba(45,212,191,.25)", line=dict(color=TEAL, width=2.4), name=pick))
        style(fig, 330, legend=True, polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False, range=[0, 1]),
              angularaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color="#c3cde0", size=11)), gridshape="linear"))
        show(fig)

    with st.container(border=True):
        ctitle("Behavioural segments (K-Means + PCA)", "Stores with similar behaviour sit close together")
        feats = ["Revenue_Cr", "YoY_%", "Ticket", "Rating", "Returning_%", "Promo_%", "Avg_Discount_%", "Units_per_Sale"]
        X = t[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        if len(X) < 4:
            st.info("Select at least 4 stores in the sidebar to compute segments.")
        else:
            k = st.slider("Number of segments", 2, min(4, len(X) - 1), min(3, len(X) - 1))
            Z = StandardScaler().fit_transform(X)
            lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Z)
            pc = PCA(n_components=2, random_state=42).fit_transform(Z)
            seg = pd.DataFrame({"Store": X.index, "x": pc[:, 0], "y": pc[:, 1], "Segment": [f"Segment {chr(65 + i)}" for i in lab], "Rev": t["Revenue_Cr"].values})
            c1, c2 = st.columns([1.4, 1])
            with c1:
                fig = px.scatter(seg, x="x", y="y", color="Segment", text="Store", size="Rev", size_max=34, color_discrete_sequence=SEQ)
                fig.update_traces(textposition="top center", textfont=dict(color="#c3cde0", size=11), marker=dict(line=dict(width=1.5, color=BG)))
                style(fig, 340, legend=True, xaxis=dict(title="PC1", gridcolor=GRID), yaxis=dict(title="PC2", gridcolor=GRID))
                show(fig)
            with c2:
                sil = silhouette_score(Z, lab) if len(set(lab)) > 1 else float("nan")
                prof = X.assign(Segment=seg["Segment"].values).groupby("Segment").mean().round(2).reset_index()
                st.markdown(f"<div class='note'>Silhouette score <b>{sil:.2f}</b> — "
                            f"{'clear separation.' if sil > 0.4 else 'groups overlap; read segments as tendencies, not hard classes.'}</div>", unsafe_allow_html=True)
                table(prof)


# ============================================================
# PAGE 05  FORECAST
# ============================================================
MODELS = ["Long-run mean", "28-day mean", "Weekday profile", "Holt-Winters (weekly)", "Ridge (calendar)"]


def _cal(idx, origin):
    return np.hstack([np.eye(7)[idx.dayofweek], np.eye(12)[idx.month - 1], ((idx - origin).days.values / 365.0)[:, None]])


def predict(name, train, fidx):
    n = len(fidx)
    if name == "Long-run mean":
        return np.full(n, train.mean())
    if name == "28-day mean":
        return np.full(n, train.iloc[-28:].mean())
    if name == "Weekday profile":
        t = train.iloc[-56:]
        prof = t.groupby(t.index.dayofweek).mean()
        return np.array([prof.get(d, t.mean()) for d in fidx.dayofweek])
    if name == "Holt-Winters (weekly)":
        m = ExponentialSmoothing(train.values, seasonal="add", seasonal_periods=7, initialization_method="estimated").fit()
        return np.asarray(m.forecast(n))
    sc = train.mean() or 1.0
    r = Ridge(alpha=1.0).fit(_cal(train.index, train.index[0]), train.values / sc)
    return r.predict(_cal(fidx, train.index[0])) * sc


@st.cache_data(show_spinner="Running the model arena (rolling-origin back-test)...")
def arena(series):
    folds, h = 4, 30
    n = len(series)
    rows, errs = [], {m: [] for m in MODELS}
    for m in MODELS:
        a_all, p_all, tot = [], [], []
        for k in range(folds, 0, -1):
            cut = n - k * h
            train, test = series.iloc[:cut], series.iloc[cut:cut + h]
            try:
                p = predict(m, train, test.index)
            except Exception:
                p = np.full(len(test), train.mean())
            a_all.append(test.values); p_all.append(p)
            tot.append(abs(p.sum() - test.values.sum()) / max(test.values.sum(), 1) * 100)
            errs[m].append(test.values - p)
        a, p = np.concatenate(a_all), np.concatenate(p_all)
        rows.append({"Model": m, "Daily WAPE %": np.abs(a - p).sum() / np.abs(a).sum() * 100, "30-day total error %": float(np.mean(tot))})
    lb = pd.DataFrame(rows).sort_values("Daily WAPE %").reset_index(drop=True)
    champ = lb.iloc[0]["Model"]
    sigma = float(np.std(np.concatenate(errs[champ])))
    return lb, champ, sigma


@st.cache_data(show_spinner=False)
def future(series, champ, horizon):
    fidx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return fidx, np.maximum(predict(champ, series, fidx), 0)


def page_forecast():
    head("05 · Forecast", "The forecast arena",
         "Five models compete in a rolling-origin back-test (4 folds × 30 days). The winner forecasts forward, with prediction bands taken from its own historical errors.")
    opts_s = ["Total (current filters)"] + [f"Category · {c}" for c in sorted(f["Product_Category"].unique())] + [f"Store · {s}" for s in sorted(f["Store_Location"].unique())]
    c1, c2 = st.columns([1, 2])
    pick = c1.selectbox("Series", opts_s)
    horizon = c2.slider("Forecast horizon (days)", 14, 180, 90)
    src = f
    if pick.startswith("Category"):
        src = f[f["Product_Category"] == pick.split("· ")[1]]
    elif pick.startswith("Store"):
        src = f[f["Store_Location"] == pick.split("· ")[1]]
    series = src.groupby("Date")["Revenue"].sum().asfreq("D", fill_value=0).astype(float)
    if len(series) < 200:
        st.warning("At least ~200 days of data are needed in scope for a reliable back-test. Widen the date range.")
        return
    lb, champ, sigma = arena(series)
    wm = float(lb.loc[lb["Model"] == "Long-run mean", "Daily WAPE %"].iloc[0])
    wc = float(lb.iloc[0]["Daily WAPE %"])
    skill = max(0.0, (1 - wc / wm) * 100) if wm else 0.0
    fidx, fc = future(series, champ, horizon)
    k = [("Champion", champ, ""), ("Skill vs long-run mean", f"{skill:+.1f}%", "positive = beats a flat average"),
         ("Back-test daily WAPE", f"{wc:.1f}%", "lower is better"),
         ("30-day total error", f"{float(lb.iloc[0]['30-day total error %']):.1f}%", "how well totals are predicted"),
         (f"Next {horizon} days", cr(fc.sum(), 1), "point forecast")]
    html = "".join(f"<div class='kpi' style='min-height:92px'><div class='kpi-l'>{a}</div><div class='kpi-v' style='font-size:22px'>{b}</div><div class='kpi-s'>{c}</div></div>" for a, b, c in k)
    st.markdown(f"<div class='kpi-grid' style='grid-template-columns:repeat(5,1fr)'>{html}</div>", unsafe_allow_html=True)

    with st.container(border=True):
        ctitle(f"Next {horizon} days · champion forecast with 80% / 95% bands", pick)
        hist = series.iloc[-180:]
        fig = go.Figure()
        for z, a_, nm in [(1.96, 0.10, "95% band"), (1.2816, 0.18, "80% band")]:
            fig.add_trace(go.Scatter(x=list(fidx) + list(fidx[::-1]), y=list(np.maximum(fc + z * sigma, 0) / 1e7) + list(np.maximum(fc - z * sigma, 0)[::-1] / 1e7),
                                     fill="toself", fillcolor=f"rgba(245,165,36,{a_})", line=dict(width=0), name=nm, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hist.index, y=hist / 1e7, mode="lines", line=dict(color=TEAL, width=1.6), name="Actual"))
        fig.add_trace(go.Scatter(x=fidx, y=fc / 1e7, mode="lines", line=dict(color=AMBER, width=3), name="Forecast"))
        fig.add_shape(type="line", x0=fidx[0].strftime("%Y-%m-%d"), x1=fidx[0].strftime("%Y-%m-%d"), y0=0, y1=1, yref="paper", line=dict(color=MUTED, dash="dot"))
        style(fig, 400, legend=True, yaxis=dict(title="₹ Cr / day", gridcolor=GRID), hovermode="x unified")
        show(fig)

    a, b = st.columns([1.2, 1])
    with a.container(border=True):
        ctitle("Model leaderboard", "Lower error is better · 🏆 = champion")
        show_lb = lb.copy()
        show_lb["Model"] = [("🏆 " + m) if m == champ else m for m in show_lb["Model"]]
        table(show_lb, column_config={"Daily WAPE %": st.column_config.NumberColumn(format="%.2f"), "30-day total error %": st.column_config.NumberColumn(format="%.2f")})
    with b.container(border=True):
        ctitle("Reading this honestly")
        if champ in ("Long-run mean", "28-day mean") or skill < 2:
            st.markdown("<div class='note'><b>No model meaningfully beats a flat average.</b> Daily revenue behaves like noise around a stable level, so a day-by-day forecast "
                        "cannot be sharper than the bands shown. Use the <b>30-day total</b> for planning and treat the bands as the realistic daily range.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='note'><b>{champ}</b> improves on a flat average by <b>{skill:.1f}%</b> in back-testing, so there is real structure (weekday or seasonal pattern) to exploit. "
                        "Prediction bands come from out-of-sample errors, so they reflect realistic accuracy rather than in-sample fit.</div>", unsafe_allow_html=True)
    out = pd.DataFrame({"Date": fidx, "Forecast (₹)": fc, "Lower 95% (₹)": np.maximum(fc - 1.96 * sigma, 0), "Upper 95% (₹)": fc + 1.96 * sigma})
    st.download_button("Download forecast CSV", out.to_csv(index=False).encode("utf-8"), "revenue_forecast.csv", "text/csv")


# ============================================================
# PAGE 06  DISCOUNT LAB
# ============================================================
def page_discount():
    head("06 · Discount Lab", "What is a discount point worth?",
         "Tests whether discounts and promotions actually move volume, then lets you simulate a change in discount policy.")
    g = f.groupby("Discount_Percentage").agg(units=("Units_Sold", "mean"), tx=("Revenue", "size"), ticket=("Revenue", "mean")).reset_index()
    groups = [x["Units_Sold"].values for _, x in f.groupby("Discount_Percentage") if len(x) > 1]
    p_anova = stats.f_oneway(*groups).pvalue if len(groups) > 1 else np.nan
    pr, npr = f[f["Promotion_Applied"] == "Yes"]["Revenue"], f[f["Promotion_Applied"] == "No"]["Revenue"]
    p_promo = stats.ttest_ind(pr, npr, equal_var=False).pvalue if len(pr) > 1 and len(npr) > 1 else np.nan
    lr = stats.linregress(f["Discount_Percentage"], f["Units_Sold"])
    mean_units = f["Units_Sold"].mean()
    slope_pct = lr.slope / mean_units * 100
    ci = 1.96 * lr.stderr / mean_units * 100

    a, b = st.columns([1, 1])
    with a.container(border=True):
        ctitle("Units per sale by discount tier", "Average basket size at each discount level")
        fig = go.Figure(go.Bar(x=[f"{int(x)}% off" for x in g["Discount_Percentage"]], y=g["units"], marker_color=SEQ[: len(g)],
                               text=[f"{v:.1f}" for v in g["units"]], textposition="outside", textfont=dict(color=MUTED)))
        style(fig, 330, yaxis=dict(title="Units / sale", rangemode="tozero", gridcolor=GRID))
        show(fig)
        verdict = "Basket size differs by tier." if p_anova < 0.05 else "Basket size is flat across tiers."
        st.markdown(f"<div class='note'><b>{verdict}</b> ANOVA p = {p_anova:.2f}; promoted sales vs unpromoted ticket p = {p_promo:.2f}. "
                    f"Estimated volume response: <b>{slope_pct:+.2f}% per discount point</b> (95% CI ±{ci:.2f}%).</div>", unsafe_allow_html=True)

    gross = f["Total_Sales"].sum()
    d0_ = f["Discount_Amount"].sum() / gross * 100
    years = max((f["Date"].max() - f["Date"].min()).days / 365.25, 0.1)
    with b.container(border=True):
        ctitle("Scenario simulator", "Annualised, all stores in scope")
        d_new = st.slider("Average discount (%)", 0.0, 25.0, float(round(d0_, 1)), 0.5)
        resp = st.slider("Volume response per +1 discount point (%)", -2.0, 4.0, float(round(slope_pct, 2)), 0.05)
        delta = d_new - d0_
        g_new = gross * (1 + resp / 100 * delta)
        rev_now = gross * (1 - d0_ / 100)
        rev_new = g_new * (1 - d_new / 100)
        be = (((1 - d0_ / 100) / (1 - d_new / 100)) - 1) / delta * 100 if abs(delta) > 1e-9 else float("nan")
        ch = (rev_new - rev_now) / years
        k = [("Net revenue today", cr(rev_now / years)), ("Scenario", cr(rev_new / years)),
             ("Change per year", f"<span class='{'up' if ch >= 0 else 'down'}'>{'+' if ch >= 0 else '−'}{cr(abs(ch))}</span>"),
             ("Break-even response", f"{be:.2f}% / pt" if not np.isnan(be) else "—")]
        html = "".join(f"<div class='kpi' style='min-height:78px'><div class='kpi-l'>{a_}</div><div class='kpi-v' style='font-size:20px'>{b_}</div></div>" for a_, b_ in k)
        st.markdown(f"<div class='kpi-grid' style='grid-template-columns:repeat(2,1fr)'>{html}</div>", unsafe_allow_html=True)
        if abs(delta) > 1e-9:
            st.markdown(f"<div class='note'>Moving the average discount from <b>{d0_:.1f}%</b> to <b>{d_new:.1f}%</b> only pays off if each extra point lifts volume by more than "
                        f"<b>{be:.2f}%</b>. History shows <b>{slope_pct:+.2f}% ± {ci:.2f}%</b>.</div>", unsafe_allow_html=True)

    with st.container(border=True):
        ctitle("Discount leakage by category", "Gross sales vs discount given, ₹ Cr")
        c = f.groupby("Product_Category").agg(gross=("Total_Sales", "sum"), disc=("Discount_Amount", "sum")).reset_index()
        c["rate"] = c["disc"] / c["gross"] * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=c["Product_Category"], y=c["gross"] / 1e7, name="Gross sales", marker_color=VIOLET))
        fig.add_trace(go.Bar(x=c["Product_Category"], y=c["disc"] / 1e7, name="Discount given", marker_color=RED,
                             text=[f"{r:.1f}%" for r in c["rate"]], textposition="outside", textfont=dict(color=MUTED)))
        style(fig, 320, legend=True, barmode="group", yaxis=dict(title="₹ Cr", gridcolor=GRID))
        show(fig)


# ============================================================
# PAGE 07  INTEGRITY & METHOD
# ============================================================
def page_integrity():
    head("07 · Integrity & Method", "Trust, but verify",
         "Automated data-quality checks run on the full dataset (ignoring sidebar filters), followed by the methods used on every page.")
    d = df
    checks = []

    def add(name, ok, detail, warn=False):
        checks.append((name, "PASS" if ok else ("WARN" if warn else "FAIL"), detail))

    add("Row count", len(d) > 0, f"{len(d):,} rows × {d.shape[1]} columns")
    nulls = int(d.isna().sum().sum())
    add("Missing values", nulls == 0, f"{nulls:,} empty cells", warn=True)
    dup = int(d.duplicated().sum())
    add("Duplicate rows", dup == 0, f"{dup:,} exact duplicates", warn=True)
    span = (d["Date"].max() - d["Date"].min()).days + 1
    miss_days = span - d["Date"].nunique()
    add("Date coverage", miss_days == 0, f"{d['Date'].min():%d %b %Y} → {d['Date'].max():%d %b %Y} · {miss_days} days without sales", warn=True)
    if "Discount_Percentage" in d:
        err = (d["Total_Sales"] * (1 - d["Discount_Percentage"] / 100) - d["Revenue"]).abs().max()
        add("Revenue reconciliation", err < 1, f"Revenue = Total sales × (1 − discount). Max difference ₹{err:,.2f}")
    if "Unit_Price" in d:
        err2 = (d["Unit_Price"] * d["Units_Sold"] - d["Total_Sales"]).abs().max()
        add("Total sales = price × units", err2 < 1, f"Max difference ₹{err2:,.2f}")
        add("Positive prices & units", bool(((d["Unit_Price"] > 0) & (d["Units_Sold"] > 0)).all()), "No zero or negative prices/units")
    if "Store_ID" in d:
        one = d.groupby("Store_ID")["Store_Location"].nunique().max() == 1
        add("Store ID ↔ location", bool(one), "Each store ID maps to exactly one city")
    add("Store ↔ region", bool(d.groupby("Store_Location")["Region"].nunique().max() == 1), "Each city belongs to exactly one region")
    rows = "".join(f"<tr><td>{n}</td><td><span class='pill {r.lower()}'>{r}</span></td><td>{t}</td></tr>" for n, r, t in checks)
    with st.container(border=True):
        ctitle("Data quality checks")
        st.markdown(f"<table class='chk'><tr><th>Check</th><th>Result</th><th>Detail</th></tr>{rows}</table>", unsafe_allow_html=True)

    a, b = st.columns(2)
    with a.container(border=True):
        ctitle("Methods")
        st.markdown("""<div class='note'>
<b>Net revenue</b> = Total sales minus discount. <b>Average ticket</b> = net revenue per transaction.<br>
<b>Year-over-year</b> compares the selected dates with the same dates one year earlier (same filters).<br>
<b>Insights</b> are computed from the filtered data: growth, discount-volume correlation, autocorrelation and store spread.<br>
<b>Forecast arena</b>: five models, 4-fold rolling-origin back-test of 30 days each, ranked by daily WAPE; bands use out-of-sample error spread.<br>
<b>Segments</b>: K-Means on 8 standardised store features, projected to 2-D with PCA.<br>
<b>Discount lab</b>: ANOVA, Welch t-test and linear regression (95% CI) from SciPy.</div>""", unsafe_allow_html=True)
    with b.container(border=True):
        ctitle("Export")
        st.markdown("<div class='note'>Download exactly what you see in the dashboard (current filters).</div>", unsafe_allow_html=True)
        st.download_button("Download filtered data (CSV)", f.to_csv(index=False).encode("utf-8"), "filtered_retail_data.csv", "text/csv")
        with st.expander("Preview filtered rows"):
            table(f.head(500))


# ============================================================
# ROUTER
# ============================================================
if page.endswith("Briefing"):
    page_briefing()
elif page.endswith("Pulse"):
    page_pulse()
elif page.endswith("Mix"):
    page_mix()
elif page.endswith("Stores"):
    page_stores()
elif page.endswith("Forecast"):
    page_forecast()
elif page.endswith("Discount Lab"):
    page_discount()
else:
    page_integrity()

st.markdown(
    "<div class='note' style='margin-top:28px;border-top:1px solid #1c2536;padding-top:12px'>"
    "Retail Intelligence · Built with Python, Streamlit, Plotly, scikit-learn and statsmodels. "
    "Forecasts and clusters are decision-support outputs; validate before operational use.</div>", unsafe_allow_html=True)
