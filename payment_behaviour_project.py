"""
Payment Behaviour & Digital Payment Intelligence
Fundamentals of Machine Learning (MLE2503)

IMPORTANT:
- The included CSV is a synthetic teaching dataset generated for this project.
- Replace it with faculty-provided/approved real data if available.
- This project performs unsupervised customer segmentation; it does not predict fraud.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# 1. Load and inspect transaction-level data
df = pd.read_csv(ROOT / "synthetic_payment_transactions.csv")
print("First five rows:\n", df.head())
print("\nLast five rows:\n", df.tail())
print("\nShape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nDescriptive statistics:\n", df.describe(include="all").T)
print("\nMissing values:\n", df.isna().sum())
print("Exact duplicate rows:", df.duplicated().sum())

# 2. Clean records. A duplicate transaction_id indicates a duplicated transaction.
df = df.drop_duplicates(subset="transaction_id", keep="first").copy()
df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
df["payment_mode"] = df["payment_mode"].fillna("Unknown")
df["channel"] = df["channel"].fillna("Unknown")
df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce")
df = df.dropna(subset=["customer_id", "transaction_amount", "transaction_date"])
df = df[df["transaction_amount"] > 0]

# Cap extreme transaction values at the 99th percentile. This limits the effect of
# extreme values on distance-based clustering while retaining most observations.
upper = df["transaction_amount"].quantile(.99)
df["transaction_amount"] = df["transaction_amount"].clip(upper=upper)
df["discount_percent"] = df["discount_percent"].clip(0, 100)

# 3. EDA plots
plt.figure()
df["payment_mode"].value_counts().plot(kind="bar")
plt.title("Transactions by Payment Mode"); plt.xlabel("Payment mode"); plt.ylabel("Transaction count")
plt.tight_layout(); plt.savefig(PLOT_DIR/"payment_mode_counts.png", dpi=160); plt.close()

plt.figure()
df["transaction_amount"].plot(kind="hist", bins=35)
plt.title("Transaction Amount Distribution"); plt.xlabel("Amount"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(PLOT_DIR/"transaction_amount_distribution.png", dpi=160); plt.close()

plt.figure()
df.groupby("payment_mode")["transaction_amount"].median().sort_values().plot(kind="bar")
plt.title("Median Transaction Amount by Payment Mode"); plt.xlabel("Payment mode"); plt.ylabel("Median amount")
plt.tight_layout(); plt.savefig(PLOT_DIR/"median_amount_by_mode.png", dpi=160); plt.close()

# 4. Aggregate transactions to customer-level features
df["is_digital"] = df["payment_mode"].isin(["UPI", "Wallet"]).astype(int)
df["is_card"] = (df["payment_mode"] == "Card").astype(int)
df["is_cash"] = (df["payment_mode"] == "Cash").astype(int)
df["is_emi_mode"] = (df["payment_mode"] == "EMI/BNPL").astype(int)
df["is_unknown_mode"] = (df["payment_mode"] == "Unknown").astype(int)
df["month"] = df["transaction_date"].dt.month

customer = df.groupby("customer_id").agg(
    transaction_count=("transaction_id", "nunique"),
    total_spend=("transaction_amount", "sum"),
    average_transaction_value=("transaction_amount", "mean"),
    median_transaction_value=("transaction_amount", "median"),
    cashback_usage_rate=("cashback_used", "mean"),
    average_discount_percent=("discount_percent", "mean"),
    emi_usage_rate=("emi_used", "mean"),
    digital_payment_ratio=("is_digital", "mean"),
    card_payment_ratio=("is_card", "mean"),
    cash_payment_ratio=("is_cash", "mean"),
    emi_payment_ratio=("is_emi_mode", "mean"),
    unknown_payment_ratio=("is_unknown_mode", "mean"),
    channel_diversity=("channel", "nunique"),
    active_months=("month", "nunique")
).reset_index()

# Ratios should sum to approximately 1 across the four mutually exclusive modes
# when all mode values are known; Unknown is handled separately in raw data.
customer.to_csv(ROOT/"customer_level_features.csv", index=False)
print("\nCustomer-level feature sample:\n", customer.head())
print("\nCustomer feature dimensions:", customer.shape)

# 5. Compare two feature sets:
# A: payment-mode behaviour only; B: payment + spending + frequency + offer usage.
payment_features = [
    "digital_payment_ratio", "card_payment_ratio", "cash_payment_ratio", "emi_payment_ratio",
    "unknown_payment_ratio"
]
combined_features = payment_features + [
    "transaction_count", "total_spend", "average_transaction_value",
    "cashback_usage_rate", "average_discount_percent", "emi_usage_rate",
    "channel_diversity", "active_months"
]

def prepare_matrix(frame, features):
    X = frame[features].copy()
    for col in X.columns:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        X[col] = X[col].fillna(X[col].median())
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

# No train/test split is used: this is an unsupervised clustering task without a target label.
# Internal metrics compare cluster structure on the same data; external validation on new/approved data
# would still be needed before any real business deployment.
# 6. Inspect inertia (Elbow Method) and silhouette across K=2..7, then compare KMeans and Agglomerative.
results = []
best_models = {}
for feature_label, features in [("Payment-only", payment_features),
                                ("Payment+Behaviour", combined_features)]:
    X_scaled, scaler = prepare_matrix(customer, features)
    inertias = []
    silhouettes = []
    k_values = range(2, 8)
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    plt.figure()
    plt.plot(list(k_values), inertias, marker="o")
    plt.title(f"Elbow Method — {feature_label}")
    plt.xlabel("Number of clusters (K)"); plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(PLOT_DIR/(feature_label.lower().replace("+","_").replace("-","_")+"_elbow.png"), dpi=160)
    plt.close()

    # Select the maximum-silhouette candidate among K=2..7 as a reproducible starting point.
    # Inspect the Elbow plot too; business interpretability matters, so do not rely on one metric alone.
    best_k = list(k_values)[int(np.argmax(silhouettes))]
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    results.append({
        "feature_set": feature_label, "model": "K-Means", "k": best_k,
        "silhouette": silhouette_score(X_scaled, km_labels),
        "davies_bouldin": davies_bouldin_score(X_scaled, km_labels),
        "inertia": km.inertia_
    })
    best_models[(feature_label, "K-Means")] = (X_scaled, km_labels, km, features)

    # Hierarchical comparison with same K for a like-for-like comparison.
    ag = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    ag_labels = ag.fit_predict(X_scaled)
    results.append({
        "feature_set": feature_label, "model": "Agglomerative",
        "k": best_k, "silhouette": silhouette_score(X_scaled, ag_labels),
        "davies_bouldin": davies_bouldin_score(X_scaled, ag_labels),
        "inertia": np.nan
    })
    best_models[(feature_label, "Agglomerative")] = (X_scaled, ag_labels, ag, features)

    # PCA visualization of K-Means solution
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    plt.figure()
    plt.scatter(coords[:,0], coords[:,1], c=km_labels, s=18)
    plt.title(f"PCA View — {feature_label} K-Means")
    plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(PLOT_DIR/(feature_label.lower().replace("+","_").replace("-","_")+"_pca.png"), dpi=160)
    plt.close()

results_df = pd.DataFrame(results)
results_df.to_csv(ROOT/"model_comparison.csv", index=False)
print("\nModel comparison:\n", results_df.to_string(index=False))

# 7. Choose combined-feature KMeans for business profiling, while acknowledging
# final choice depends on metrics and interpretability.
X_scaled, labels, model, features = best_models[("Payment+Behaviour", "K-Means")]
customer["cluster"] = labels
customer.to_csv(ROOT/"customer_segments.csv", index=False)

profile_cols = [
    "transaction_count", "total_spend", "average_transaction_value",
    "cashback_usage_rate", "average_discount_percent", "emi_usage_rate",
    "digital_payment_ratio", "card_payment_ratio", "cash_payment_ratio",
    "emi_payment_ratio", "unknown_payment_ratio", "channel_diversity", "active_months"
]
profile = customer.groupby("cluster")[profile_cols].mean().round(2)
profile["customer_count"] = customer.groupby("cluster").size()
profile.to_csv(ROOT/"cluster_profiles.csv")
print("\nCluster profile table:\n", profile)

plt.figure()
profile["customer_count"].plot(kind="bar")
plt.title("Customer Count by Cluster"); plt.xlabel("Cluster"); plt.ylabel("Number of customers")
plt.tight_layout(); plt.savefig(PLOT_DIR/"cluster_sizes.png", dpi=160); plt.close()

# Normalized heatmap of profile features (for interpretation, not model training).
heat_cols = ["transaction_count","average_transaction_value","cashback_usage_rate",
             "emi_usage_rate","digital_payment_ratio","card_payment_ratio","cash_payment_ratio"]
heat = profile[heat_cols].copy()
heat_norm = (heat - heat.mean()) / heat.std(ddof=0).replace(0, 1)
plt.figure(figsize=(10, 5))
plt.imshow(heat_norm.fillna(0).values, aspect="auto")
plt.xticks(range(len(heat_cols)), heat_cols, rotation=35, ha="right")
plt.yticks(range(len(heat_norm.index)), [f"Cluster {i}" for i in heat_norm.index])
plt.colorbar(label="Relative feature level (standardized)")
plt.title("Cluster Behaviour Profile Heatmap")
plt.tight_layout(); plt.savefig(PLOT_DIR/"cluster_profile_heatmap.png", dpi=160); plt.close()

print("\nFiles generated. Interpret clusters using the profile table; cluster numbers themselves have no inherent meaning.")
