# =============================================================================
# 💧 Water Quality Classification — Streamlit App
# Deploy: streamlit run app.py
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💧 AquaGuard — Water Quality AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES  — dark industrial + electric-teal accent
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0e14;
    color: #e2e8f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2d40;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label { color: #38bdf8 !important; }

/* ── Main background ── */
.main .block-container {
    background: #0a0e14;
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0a1628 0%, #0f2744 50%, #071521 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    color: #38bdf8;
    letter-spacing: -1px;
    margin: 0 0 0.5rem 0;
    line-height: 1;
}
.hero-sub {
    font-size: 1.05rem;
    color: #64748b;
    font-family: 'Space Mono', monospace;
    font-weight: 400;
    margin: 0;
}
.hero-tag {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid #38bdf8;
    color: #38bdf8;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 1px;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.4rem;
    color: #f1f5f9;
    border-left: 4px solid #38bdf8;
    padding-left: 1rem;
    margin: 2rem 0 1rem 0;
    letter-spacing: -0.3px;
}

/* ── Metric cards ── */
.metric-card {
    background: #0d1929;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #38bdf8; }
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1;
}
.metric-label {
    font-size: 0.8rem;
    color: #475569;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* ── Result box ── */
.result-safe {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-unsafe {
    background: linear-gradient(135deg, #2d0a0a, #450a0a);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
}
.result-desc {
    font-size: 0.9rem;
    font-family: 'Space Mono', monospace;
    color: #94a3b8;
    margin-top: 0.5rem;
}

/* ── Confidence bar ── */
.conf-bar-bg {
    background: #1e293b;
    border-radius: 4px;
    height: 8px;
    width: 100%;
    margin: 0.5rem 0;
}
.conf-bar-fill {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #38bdf8, #22c55e);
}

/* ── Data table ── */
.stDataFrame { border: 1px solid #1e3a5f !important; border-radius: 8px !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #475569;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #38bdf8 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0284c7);
    border: none;
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.7rem 2rem;
    border-radius: 10px;
    width: 100%;
    transition: all 0.2s;
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0284c7, #38bdf8);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(56,189,248,0.3);
}

/* ── Expander ── */
details { border: 1px solid #1e3a5f !important; border-radius: 10px !important; }
summary { color: #38bdf8 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e14; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1929",
    "axes.facecolor":    "#0d1929",
    "axes.edgecolor":    "#1e3a5f",
    "axes.labelcolor":   "#94a3b8",
    "xtick.color":       "#475569",
    "ytick.color":       "#475569",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1e3a5f",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
})


# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA + MODEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(path):
    df = pd.read_csv(path)
    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        df[col] = df[col].fillna(df[col].mean())
    df["pH_Hardness_ratio"]      = df["ph"] / (df["Hardness"] + 1e-6)
    df["TDS_Conductivity_ratio"] = df["Solids"] / (df["Conductivity"] + 1e-6)
    df["Chloramines_THM_ratio"]  = df["Chloramines"] / (df["Trihalomethanes"] + 1e-6)
    df["ph_sq"]                  = df["ph"] ** 2
    df["Turbidity_sq"]           = df["Turbidity"] ** 2
    df["Organic_carbon_sq"]      = df["Organic_carbon"] ** 2
    return df


@st.cache_resource(show_spinner=False)
def train_models(path):
    df = load_and_preprocess(path)
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_sm)
    X_te_sc = scaler.transform(X_test)

    models = {
        "Random Forest":      RandomForestClassifier(n_estimators=300, min_samples_leaf=0.03, random_state=42, n_jobs=-1),
        "XGBoost":            XGBClassifier(max_depth=8, n_estimators=125, learning_rate=0.03, random_state=0, n_jobs=-1, eval_metric="logloss", verbosity=0),
        "SVM":                SVC(kernel="rbf", random_state=42, probability=True),
        "AdaBoost":           AdaBoostClassifier(learning_rate=0.002, n_estimators=205, random_state=42),
        "KNeighbors":         KNeighborsClassifier(n_neighbors=9, n_jobs=-1),
        "Decision Tree":      DecisionTreeClassifier(max_depth=4, random_state=42),
        "Logistic Regression":LogisticRegression(max_iter=1000, random_state=0, n_jobs=-1),
    }

    results, roc_data, trained = {}, {}, {}
    for name, model in models.items():
        model.fit(X_tr_sc, y_tr_sm)
        preds = model.predict(X_te_sc)
        probs = (model.predict_proba(X_te_sc)[:, 1]
                 if hasattr(model, "predict_proba")
                 else model.decision_function(X_te_sc))
        results[name] = {
            "Accuracy": accuracy_score(y_test, preds),
            "ROC-AUC":  roc_auc_score(y_test, probs),
            "Report":   classification_report(y_test, preds, target_names=["Not Potable", "Potable"], output_dict=True),
            "CM":       confusion_matrix(y_test, preds),
        }
        roc_data[name] = (probs, roc_auc_score(y_test, probs))
        trained[name]  = model

    return df, X, y, scaler, trained, results, roc_data, X_test, y_test, X.columns.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">ML · WATER SAFETY · CLASSIFICATION</div>
  <p class="hero-title">💧 AquaGuard</p>
  <p class="hero-sub">AI-powered water potability analysis — 7 models, 15 features, real-time prediction</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — DATA UPLOAD + PREDICTOR INPUTS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    uploaded = st.file_uploader("Upload `water_potability.csv`", type="csv")
    st.markdown("---")

    st.markdown("### 🔬 Predict Water Sample")
    st.caption("Adjust sliders to describe your water sample:")

    ph_val      = st.slider("pH",              0.0, 14.0, 7.0, 0.01)
    hard_val    = st.slider("Hardness",         50.0, 350.0, 196.0, 0.5)
    solids_val  = st.slider("Solids (TDS)",    300.0, 62000.0, 22000.0, 100.0)
    chlor_val   = st.slider("Chloramines",       0.0, 13.0, 7.1, 0.1)
    sulf_val    = st.slider("Sulfate",         100.0, 500.0, 333.0, 1.0)
    cond_val    = st.slider("Conductivity",    180.0, 800.0, 426.0, 1.0)
    org_val     = st.slider("Organic Carbon",    2.0, 28.0, 14.0, 0.1)
    thm_val     = st.slider("Trihalomethanes",   8.0, 124.0, 66.0, 0.5)
    turb_val    = st.slider("Turbidity",         1.5, 7.0, 3.97, 0.01)

    st.markdown("---")
    chosen_model = st.selectbox("Model for Prediction", [
        "Random Forest", "XGBoost", "SVM",
        "AdaBoost", "KNeighbors", "Decision Tree", "Logistic Regression"
    ])

if uploaded:
    DATA_PATH = uploaded
else:
    DATA_PATH = "water_potability.csv"

try:
    df_raw, X_all, y_all, scaler, trained_models, results, roc_data, X_test, y_test, feat_names = train_models(DATA_PATH)
    data_loaded = True
except FileNotFoundError:
    st.error("⚠️  `water_potability.csv` not found. Upload the file via the sidebar.")
    data_loaded = False

if not data_loaded:
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────────────────────────────────────
best_model_name = max(results, key=lambda k: results[k]["Accuracy"])
best_acc  = results[best_model_name]["Accuracy"]
best_auc  = results[best_model_name]["ROC-AUC"]
potable_pct = (y_all == 1).mean() * 100

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, f"{len(df_raw):,}",          "TOTAL SAMPLES"),
    (c2, f"{best_acc*100:.1f}%",      "BEST ACCURACY"),
    (c3, f"{best_auc:.3f}",           "BEST ROC-AUC"),
    (c4, f"{potable_pct:.1f}%",       "POTABLE RATE"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-value">{val}</div>
      <div class="metric-label">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🎯 Predict",
    "📊 EDA",
    "🏆 Models",
    "📈 ROC Curves",
    "🔍 Feature Importance",
    "📋 Dataset"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Real-Time Potability Prediction</div>', unsafe_allow_html=True)

    col_form, col_res = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("##### Selected Parameters")
        params = {
            "ph": ph_val, "Hardness": hard_val, "Solids": solids_val,
            "Chloramines": chlor_val, "Sulfate": sulf_val,
            "Conductivity": cond_val, "Organic_carbon": org_val,
            "Trihalomethanes": thm_val, "Turbidity": turb_val,
        }
        param_df = pd.DataFrame(params.items(), columns=["Parameter", "Value"])
        param_df["Value"] = param_df["Value"].round(3)
        st.dataframe(param_df, use_container_width=True, hide_index=True)

        predict_btn = st.button("🔬 Analyze Water Sample")

    with col_res:
        if predict_btn:
            sample = np.array([[
                ph_val, hard_val, solids_val, chlor_val, sulf_val,
                cond_val, org_val, thm_val, turb_val,
                ph_val / (hard_val + 1e-6),
                solids_val / (cond_val + 1e-6),
                chlor_val / (thm_val + 1e-6),
                ph_val ** 2,
                turb_val ** 2,
                org_val ** 2,
            ]])
            sample_sc  = scaler.transform(sample)
            model      = trained_models[chosen_model]
            pred       = model.predict(sample_sc)[0]
            prob       = model.predict_proba(sample_sc)[0] if hasattr(model, "predict_proba") else None

            if pred == 1:
                conf = float(prob[1]) * 100 if prob is not None else None
                st.markdown(f"""
                <div class="result-safe">
                  <p class="result-title" style="color:#22c55e">✅ POTABLE</p>
                  <p class="result-desc">Water appears safe for drinking</p>
                  {'<p style="color:#4ade80;font-family:monospace;font-size:1.1rem;margin-top:1rem">Confidence: ' + f"{conf:.1f}%" + '</p>' if conf else ''}
                </div>""", unsafe_allow_html=True)
            else:
                conf = float(prob[0]) * 100 if prob is not None else None
                st.markdown(f"""
                <div class="result-unsafe">
                  <p class="result-title" style="color:#ef4444">⚠️ NOT POTABLE</p>
                  <p class="result-desc">Water is unsafe for drinking</p>
                  {'<p style="color:#f87171;font-family:monospace;font-size:1.1rem;margin-top:1rem">Confidence: ' + f"{conf:.1f}%" + '</p>' if conf else ''}
                </div>""", unsafe_allow_html=True)

            if prob is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Probability Breakdown**")
                fig_p, ax_p = plt.subplots(figsize=(6, 1.5))
                ax_p.barh(["Not Potable", "Potable"], [prob[0]*100, prob[1]*100],
                          color=["#ef4444", "#22c55e"], height=0.4)
                ax_p.set_xlim(0, 100)
                ax_p.set_xlabel("Probability (%)")
                ax_p.axvline(50, color="#38bdf8", linestyle="--", alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig_p, use_container_width=True)
                plt.close()
        else:
            st.info("👈 Adjust sliders in the sidebar, then click **Analyze Water Sample**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    eda1, eda2 = st.columns(2)

    # Missing values
    with eda1:
        st.markdown("##### Missing Values (%)")
        miss = load_and_preprocess.__wrapped__(DATA_PATH) if hasattr(load_and_preprocess, "__wrapped__") else pd.read_csv(DATA_PATH)
        raw_df = pd.read_csv(DATA_PATH) if isinstance(DATA_PATH, str) else pd.read_csv(DATA_PATH)
        miss_pct = (raw_df.isnull().mean() * 100).round(2)
        fig_m, ax_m = plt.subplots(figsize=(6, 3.5))
        miss_pct.plot.bar(ax=ax_m, color="#38bdf8", edgecolor="#0a0e14", width=0.6)
        ax_m.set_ylabel("Missing (%)")
        ax_m.set_title("Missing Data by Feature", color="#e2e8f0")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_m, use_container_width=True)
        plt.close()

    # Class distribution
    with eda2:
        st.markdown("##### Class Distribution")
        fig_c, axes_c = plt.subplots(1, 2, figsize=(6, 3.5))
        vc = raw_df["Potability"].value_counts()
        axes_c[0].bar(["Not Potable", "Potable"], vc.values,
                      color=["#ef4444", "#22c55e"], edgecolor="#0a0e14", width=0.5)
        axes_c[0].set_title("Count", color="#e2e8f0")
        axes_c[1].pie(vc.values, labels=["Not Potable", "Potable"],
                      autopct="%1.1f%%", colors=["#ef4444", "#22c55e"],
                      startangle=90, textprops={"color": "#e2e8f0"})
        axes_c[1].set_title("Proportion", color="#e2e8f0")
        plt.tight_layout()
        st.pyplot(fig_c, use_container_width=True)
        plt.close()

    # Correlation heatmap
    st.markdown("##### Correlation Matrix")
    fig_heat, ax_heat = plt.subplots(figsize=(11, 5))
    numeric_cols = [c for c in raw_df.columns if raw_df[c].dtype in [np.float64, np.int64]]
    sns.heatmap(raw_df[numeric_cols].corr(), annot=True, fmt=".2f",
                cmap="YlOrRd", linewidths=0.4, ax=ax_heat,
                annot_kws={"size": 7}, linecolor="#0a0e14")
    ax_heat.set_title("Feature Correlation Matrix", color="#e2e8f0")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig_heat, use_container_width=True)
    plt.close()

    # Feature distributions
    st.markdown("##### Feature Distributions by Potability")
    base_features = ["ph", "Hardness", "Solids", "Chloramines",
                     "Sulfate", "Conductivity", "Organic_carbon",
                     "Trihalomethanes", "Turbidity"]
    fig_dist, axes_d = plt.subplots(3, 3, figsize=(14, 9))
    axes_d = axes_d.flatten()
    for i, col in enumerate(base_features):
        for label, colour, alpha in [(0, "#ef4444", 0.45), (1, "#22c55e", 0.45)]:
            subset = raw_df[raw_df["Potability"] == label][col].dropna()
            sns.kdeplot(subset, ax=axes_d[i],
                        label="Potable" if label else "Not Potable",
                        color=colour, fill=True, alpha=alpha)
        axes_d[i].set_title(col, fontsize=9)
        axes_d[i].legend(fontsize=7)
    plt.suptitle("Feature Distributions by Potability Class", y=1.01,
                 fontsize=12, color="#e2e8f0")
    plt.tight_layout()
    st.pyplot(fig_dist, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)

    res_df = (pd.DataFrame({k: {"Accuracy": v["Accuracy"], "ROC-AUC": v["ROC-AUC"]}
                             for k, v in results.items()})
              .T.sort_values("Accuracy", ascending=False))
    res_df_show = res_df.copy()
    res_df_show["Accuracy"] = (res_df_show["Accuracy"] * 100).round(2).astype(str) + " %"
    res_df_show["ROC-AUC"]  = res_df_show["ROC-AUC"].round(4)
    st.dataframe(res_df_show, use_container_width=True)

    # Bar comparison
    fig_bar, axes_b = plt.subplots(1, 2, figsize=(13, 4))
    colors = ["#38bdf8" if i == 0 else "#1e3a5f"
              for i in range(len(res_df))]

    res_df["Accuracy"].sort_values().plot.barh(
        ax=axes_b[0], color=colors[::-1], edgecolor="#0a0e14")
    axes_b[0].set_xlabel("Accuracy"); axes_b[0].set_title("Accuracy", color="#e2e8f0")
    axes_b[0].set_xlim(0.5, 1.0)

    res_df["ROC-AUC"].sort_values().plot.barh(
        ax=axes_b[1], color=colors[::-1], edgecolor="#0a0e14")
    axes_b[1].set_xlabel("ROC-AUC"); axes_b[1].set_title("ROC-AUC", color="#e2e8f0")
    axes_b[1].set_xlim(0.5, 1.0)

    plt.suptitle("Model Leaderboard", fontsize=12, color="#e2e8f0")
    plt.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close()

    # Per-model confusion matrices
    st.markdown("##### Confusion Matrices")
    cm_cols = st.columns(3)
    idx = 0
    for name, res in sorted(results.items(), key=lambda x: -x[1]["Accuracy"]):
        with cm_cols[idx % 3]:
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            cm = res["CM"]
            sns.heatmap(cm / cm.sum(), annot=True, fmt=".1%",
                        cmap="Blues", linewidths=0.5, ax=ax_cm,
                        xticklabels=["Not Potable", "Potable"],
                        yticklabels=["Not Potable", "Potable"],
                        annot_kws={"size": 9})
            ax_cm.set_title(name, fontsize=9, color="#e2e8f0")
            ax_cm.set_ylabel("True", fontsize=8)
            ax_cm.set_xlabel("Predicted", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)
            plt.close()
        idx += 1


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ROC CURVES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">ROC Curves — All Models</div>', unsafe_allow_html=True)

    palette = ["#38bdf8", "#22c55e", "#f59e0b", "#a78bfa",
               "#f87171", "#fb923c", "#67e8f9"]
    fig_roc, ax_roc = plt.subplots(figsize=(9, 6))
    for (name, (probs, auc)), color in zip(roc_data.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax_roc.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})",
                    color=color, linewidth=1.8)
    ax_roc.plot([0, 1], [0, 1], "w--", alpha=0.3, label="Random Classifier")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves — All 7 Models", color="#e2e8f0")
    ax_roc.legend(loc="lower right", fontsize=7.5,
                  facecolor="#0d1929", edgecolor="#1e3a5f")
    ax_roc.fill_between([0, 1], [0, 0], [1, 1], alpha=0.03, color="#38bdf8")
    plt.tight_layout()
    st.pyplot(fig_roc, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">Feature Importance Analysis</div>', unsafe_allow_html=True)

    fi_col1, fi_col2 = st.columns(2)
    for col_widget, model_name, bar_color in [
        (fi_col1, "Random Forest", "#38bdf8"),
        (fi_col2, "XGBoost",       "#22c55e"),
    ]:
        with col_widget:
            m = trained_models[model_name]
            fi = pd.DataFrame({
                "Feature":    feat_names,
                "Importance": m.feature_importances_
            }).sort_values("Importance", ascending=True)
            fig_fi, ax_fi = plt.subplots(figsize=(6, 6))
            ax_fi.barh(fi["Feature"], fi["Importance"],
                       color=bar_color, edgecolor="#0a0e14", height=0.6)
            ax_fi.set_title(f"{model_name} — Feature Importance",
                            color="#e2e8f0", fontsize=10)
            ax_fi.set_xlabel("Importance Score")
            plt.tight_layout()
            st.pyplot(fig_fi, use_container_width=True)
            plt.close()

            top5 = fi.sort_values("Importance", ascending=False).head(5)
            st.markdown(f"**Top 5 — {model_name}**")
            st.dataframe(top5.reset_index(drop=True), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">Dataset Explorer</div>', unsafe_allow_html=True)

    c_search, c_n = st.columns([3, 1])
    with c_n:
        n_rows = st.number_input("Rows to show", 5, 500, 50, step=5)
    with c_search:
        filter_pot = st.selectbox("Filter by Potability",
                                  ["All", "Potable (1)", "Not Potable (0)"])

    display_df = raw_df.copy()
    if filter_pot == "Potable (1)":
        display_df = display_df[display_df["Potability"] == 1]
    elif filter_pot == "Not Potable (0)":
        display_df = display_df[display_df["Potability"] == 0]

    st.dataframe(display_df.head(n_rows), use_container_width=True)

    st.markdown("##### Summary Statistics")
    st.dataframe(raw_df.describe().round(3), use_container_width=True)

    with st.expander("📐 Dataset Schema"):
        schema = pd.DataFrame({
            "Feature":     ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                            "Conductivity", "Organic_carbon", "Trihalomethanes",
                            "Turbidity", "Potability"],
            "Description": ["pH value of water", "Water hardness (mg/L)",
                            "Total dissolved solids (ppm)", "Chloramine concentration (ppm)",
                            "Sulfate concentration (mg/L)", "Electrical conductivity (μS/cm)",
                            "Organic carbon content (ppm)", "Trihalomethane concentration (μg/L)",
                            "Water turbidity (NTU)", "Target: 1=Potable, 0=Not Potable"],
            "Missing?":    ["Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No"],
        })
        st.dataframe(schema, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<br><hr style="border-color:#1e3a5f; margin: 2rem 0">
<p style="text-align:center; font-family:'Space Mono',monospace; font-size:0.75rem; color:#334155">
  💧 AquaGuard &nbsp;·&nbsp; Water Quality Classification &nbsp;·&nbsp;
  Random Forest · XGBoost · SVM · KNN · AdaBoost · Decision Tree · Logistic Regression
</p>
""", unsafe_allow_html=True)
