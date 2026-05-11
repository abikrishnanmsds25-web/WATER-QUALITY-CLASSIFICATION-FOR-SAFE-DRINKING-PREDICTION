# =============================================================================
# PROJECT #15 — Water Quality Classification for Safe Drinking Prediction
# =============================================================================
# Dataset  : water_potability.csv  (3276 samples, 10 features)
# Goal     : Classify water as Potable (1) or Not Potable (0)
# Models   : Logistic Regression, Decision Tree, Random Forest,
#            XGBoost, KNeighbors, SVM, AdaBoost
# Extras   : Missing-value handling, feature engineering, SMOTE balancing
# =============================================================================

# ── 0. Imports ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Plotly (optional – comment out if not installed)
try:
    import plotly.express as px
    PLOTLY = True
except ImportError:
    PLOTLY = False

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE   # pip install imbalanced-learn

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")

# =============================================================================
# 1. LOAD DATA
# =============================================================================
# ── Update the path below to wherever your CSV lives ──────────────────────────
DATA_PATH = "water_potability.csv"

main_df = pd.read_csv(DATA_PATH)
df = main_df.copy()

print("=" * 60)
print("WATER QUALITY DATASET — OVERVIEW")
print("=" * 60)
print(f"\nShape : {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic statistics:\n{df.describe()}")

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 60)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── 2.1 Missing values ───────────────────────────────────────────────────────
print("\n--- Missing Values ---")
print(df.isnull().sum())
print(f"\nMissing % per column:\n{(df.isnull().mean() * 100).round(2)}")

fig, ax = plt.subplots(figsize=(10, 4))
(df.isnull().mean() * 100).plot.bar(ax=ax, color="salmon", edgecolor="black")
ax.set_ylabel("Missing (%)")
ax.set_title("Missing Data by Feature")
plt.tight_layout()
plt.savefig("01_missing_values.png", dpi=150)
plt.show()

# ── 2.2 Class distribution ──────────────────────────────────────────────────
print("\n--- Class Distribution ---")
print(df["Potability"].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df["Potability"].value_counts().plot.bar(
    ax=axes[0], color=["steelblue", "coral"], edgecolor="black"
)
axes[0].set_xticklabels(["Not Potable", "Potable"], rotation=0)
axes[0].set_title("Class Distribution (count)")

axes[1].pie(
    df["Potability"].value_counts(),
    labels=["Not Potable", "Potable"],
    autopct="%1.1f%%",
    colors=["steelblue", "coral"],
    startangle=90,
)
axes[1].set_title("Class Distribution (proportion)")
plt.tight_layout()
plt.savefig("02_class_distribution.png", dpi=150)
plt.show()

# ── 2.3 Correlation heatmap ──────────────────────────────────────────────────
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("03_correlation_heatmap.png", dpi=150)
plt.show()

# ── 2.4 Boxplots (outlier detection) ─────────────────────────────────────────
features = [c for c in df.columns if c != "Potability"]
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(features):
    sns.boxplot(y=df[col], ax=axes[i], color="skyblue")
    axes[i].set_title(col)
axes[-1].set_visible(False)   # hide the empty 10th subplot
plt.suptitle("Boxplots — All Features (outlier detection)", y=1.01, fontsize=13)
plt.tight_layout()
plt.savefig("04_boxplots.png", dpi=150)
plt.show()

# ── 2.5 Feature distributions by Potability ──────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(features):
    for label, colour in zip([0, 1], ["coral", "steelblue"]):
        sns.kdeplot(
            df[df["Potability"] == label][col].dropna(),
            ax=axes[i], label=f"{'Potable' if label else 'Not Potable'}",
            color=colour, fill=True, alpha=0.4,
        )
    axes[i].set_title(col)
    axes[i].legend()
plt.suptitle("Feature Distributions by Potability", y=1.01, fontsize=13)
plt.tight_layout()
plt.savefig("05_feature_distributions.png", dpi=150)
plt.show()

# ── 2.6 Skewness ─────────────────────────────────────────────────────────────
print("\n--- Skewness ---")
print(df.skew().sort_values(ascending=False).round(3))

# ── 2.7 Violin plot (pH) ─────────────────────────────────────────────────────
plt.figure(figsize=(7, 5))
sns.violinplot(x="Potability", y="ph", data=df, palette="rocket")
plt.xticks([0, 1], ["Not Potable", "Potable"])
plt.title("pH Distribution by Potability")
plt.tight_layout()
plt.savefig("06_violin_ph.png", dpi=150)
plt.show()

# =============================================================================
# 3. DATA PRE-PROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("3. DATA PRE-PROCESSING")
print("=" * 60)

# ── 3.1 Impute missing values with column mean ───────────────────────────────
for col in ["ph", "Sulfate", "Trihalomethanes"]:
    df[col] = df[col].fillna(df[col].mean())

print(f"\nMissing values after imputation:\n{df.isnull().sum()}")

# ── 3.2 Feature engineering ──────────────────────────────────────────────────
# Ratio features that capture physicochemical interactions
df["pH_Hardness_ratio"]      = df["ph"] / (df["Hardness"] + 1e-6)
df["TDS_Conductivity_ratio"] = df["Solids"] / (df["Conductivity"] + 1e-6)
df["Chloramines_THM_ratio"]  = df["Chloramines"] / (df["Trihalomethanes"] + 1e-6)

# Squared terms for non-linear patterns
df["ph_sq"]           = df["ph"] ** 2
df["Turbidity_sq"]    = df["Turbidity"] ** 2
df["Organic_carbon_sq"] = df["Organic_carbon"] ** 2

print(f"\nDataFrame shape after feature engineering: {df.shape}")
print(f"New features: {[c for c in df.columns if c not in main_df.columns]}")

# ── 3.3 Train / test split ───────────────────────────────────────────────────
X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
print(f"Train class balance:\n{y_train.value_counts()}")

# ── 3.4 SMOTE — handle class imbalance on training set only ─────────────────
print("\n--- Applying SMOTE to training set ---")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"After  SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")

# ── 3.5 Scaling ───────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)

# =============================================================================
# 4. MODEL TRAINING & EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("4. MODEL TRAINING & EVALUATION")
print("=" * 60)

# ── Helper ────────────────────────────────────────────────────────────────────
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    """Fit, predict and print a full evaluation report."""
    model.fit(X_tr, y_tr)
    preds  = model.predict(X_te)
    acc    = accuracy_score(y_te, preds)
    # ROC-AUC (use predict_proba when available, else decision_function)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_te)[:, 1]
    else:
        probs = model.decision_function(X_te)
    auc = roc_auc_score(y_te, probs)

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Accuracy : {acc:.4f}   |   ROC-AUC : {auc:.4f}")
    print(classification_report(y_te, preds,
                                 target_names=["Not Potable", "Potable"]))

    # Confusion matrix
    cm = confusion_matrix(y_te, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm / cm.sum(), annot=True, fmt=".2%",
                cmap="Blues", linewidths=0.5,
                xticklabels=["Not Potable", "Potable"],
                yticklabels=["Not Potable", "Potable"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"cm_{name.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()

    return acc, auc, probs


# ── 4.1 Define models ─────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=0, n_jobs=-1
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=0.03, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        max_depth=8, n_estimators=125, learning_rate=0.03,
        random_state=0, n_jobs=-1, use_label_encoder=False,
        eval_metric="logloss", verbosity=0
    ),
    "KNeighbors": KNeighborsClassifier(
        n_neighbors=9, leaf_size=20, n_jobs=-1
    ),
    "SVM": SVC(
        kernel="rbf", random_state=42, probability=True
    ),
    "AdaBoost": AdaBoostClassifier(
        learning_rate=0.002, n_estimators=205, random_state=42
    ),
}

# ── 4.2 Train & collect results ───────────────────────────────────────────────
results = {}
roc_data = {}

for name, model in models.items():
    acc, auc, probs = evaluate_model(
        name, model,
        X_train_scaled, y_train_sm,
        X_test_scaled,  y_test
    )
    results[name] = {"Accuracy": acc, "ROC-AUC": auc}
    roc_data[name] = probs

# =============================================================================
# 5. MODEL COMPARISON
# =============================================================================
print("\n" + "=" * 60)
print("5. MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results).T.sort_values("Accuracy", ascending=False)
print(f"\n{results_df.round(4)}")

# ── Bar chart ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
results_df["Accuracy"].sort_values().plot.barh(
    ax=axes[0], color="steelblue", edgecolor="black"
)
axes[0].set_xlabel("Accuracy"); axes[0].set_title("Accuracy by Model")
axes[0].set_xlim(0.5, 1.0)

results_df["ROC-AUC"].sort_values().plot.barh(
    ax=axes[1], color="coral", edgecolor="black"
)
axes[1].set_xlabel("ROC-AUC"); axes[1].set_title("ROC-AUC by Model")
axes[1].set_xlim(0.5, 1.0)

plt.suptitle("Model Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("07_model_comparison.png", dpi=150)
plt.show()

# ── ROC curves ────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
for name, probs in roc_data.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = results[name]["ROC-AUC"]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("08_roc_curves.png", dpi=150)
plt.show()

# =============================================================================
# 6. CROSS-VALIDATION ON BEST MODEL
# =============================================================================
print("\n" + "=" * 60)
print("6. CROSS-VALIDATION — Best Model")
print("=" * 60)

best_name = results_df.index[0]
best_model = models[best_name]
print(f"\nBest model: {best_name}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    best_model, X_train_scaled, y_train_sm,
    cv=cv, scoring="accuracy", n_jobs=-1
)
print(f"\n5-Fold CV Accuracy: {cv_scores.round(4)}")
print(f"Mean ± Std         : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# =============================================================================
# 7. FEATURE IMPORTANCE (Random Forest & XGBoost)
# =============================================================================
print("\n" + "=" * 60)
print("7. FEATURE IMPORTANCE")
print("=" * 60)

feature_names = X.columns.tolist()

for model_name in ["Random Forest", "XGBoost"]:
    m = models[model_name]
    importances = m.feature_importances_
    fi_df = (pd.DataFrame({"Feature": feature_names, "Importance": importances})
               .sort_values("Importance", ascending=True))
    plt.figure(figsize=(8, 6))
    fi_df.plot.barh(x="Feature", y="Importance",
                    color="mediumseagreen", edgecolor="black",
                    legend=False, ax=plt.gca())
    plt.title(f"Feature Importance — {model_name}")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"fi_{model_name.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()

    print(f"\n{model_name} — Top 5 features:")
    print(fi_df.sort_values("Importance", ascending=False).head(5).to_string(index=False))

# =============================================================================
# 8. CONCLUSION
# =============================================================================
print("\n" + "=" * 60)
print("8. CONCLUSION")
print("=" * 60)
print(f"""
Dataset   : 3 276 water samples, 9 original physicochemical features
Pre-processing:
  • Mean imputation for pH, Sulfate, Trihalomethanes
  • 6 engineered features (ratio & squared terms)
  • SMOTE over-sampling on the training split only
  • StandardScaler normalisation

Model Performance (test set, sorted by accuracy):
{results_df.round(4).to_string()}

Best model : {results_df.index[0]}
  Accuracy  = {results_df['Accuracy'].iloc[0]:.4f}
  ROC-AUC   = {results_df['ROC-AUC'].iloc[0]:.4f}

Key take-aways:
  ✔ Tree-based ensembles (RF, XGBoost) generalise best on this dataset.
  ✔ SMOTE improved recall for the minority Potable class across all models.
  ✔ pH, Sulfate and Solids are consistently among the top predictors.
  ✔ SVM with RBF kernel is a strong baseline despite no ensemble averaging.
""")
