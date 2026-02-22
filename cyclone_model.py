"""
=============================================================
Cyclone Path Prediction Using Time Series & Machine Learning
B.Tech Final Year Project
Data Source: IBTrACS (International Best Track Archive for
             Climate Stewardship) - NOAA
Note: No Kaggle or GitHub datasets used.
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import GradientBoostingRegressor   # proxy for XGBoost
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. DATA COLLECTION
# ─────────────────────────────────────────────
print("=" * 60)
print("CYCLONE PATH PREDICTION - B.Tech Project")
print("Data: IBTrACS / NOAA (Official Meteorological Source)")
print("=" * 60)

RAW_FILE = "cyclone.csv"

# ─────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\n[STEP 1] Loading and preprocessing data ...")

df_raw = pd.read_csv(RAW_FILE, skiprows=[1], low_memory=False)   # row 1 is units

# Keep only essential columns
cols_needed = ["SID", "ISO_TIME", "LAT", "LON", "NAME", "BASIN", "SEASON"]
df = df_raw[cols_needed].copy()

# Convert types
df["LAT"]  = pd.to_numeric(df["LAT"],  errors="coerce")
df["LON"]  = pd.to_numeric(df["LON"],  errors="coerce")
df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")

# Drop missing lat/lon
df.dropna(subset=["LAT", "LON", "ISO_TIME"], inplace=True)

# Sort by storm ID then time  
df.sort_values(["SID", "ISO_TIME"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"  Total records after cleaning : {len(df):,}")
print(f"  Unique cyclones              : {df['SID'].nunique():,}")
print(f"  Date range                   : {df['ISO_TIME'].min().date()} → {df['ISO_TIME'].max().date()}")
print(f"  Basins covered               : {sorted([b for b in df['BASIN'].unique() if isinstance(b, str)])}")

# ─────────────────────────────────────────────
# 3. TIME SERIES TRANSFORMATION (Sliding Window)
# ─────────────────────────────────────────────
WINDOW = 5   # past 5 positions as input

def make_sequences(group, window=WINDOW):
    """
    Convert a single cyclone's track into sliding-window samples.
    Input:  [lat_t-4, lon_t-4, ..., lat_t,   lon_t]   (10 features)
    Target: [lat_t+1, lon_t+1]                          (2 targets)
    """
    lats = group["LAT"].values
    lons = group["LON"].values
    X_rows, y_rows = [], []
    for i in range(window, len(lats) - 1):
        features = []
        for j in range(i - window, i):
            features.extend([lats[j], lons[j]])
        X_rows.append(features)
        y_rows.append([lats[i + 1], lons[i + 1]])
    return X_rows, y_rows

print("\n[STEP 2] Creating sliding-window sequences (window=5) ...")
X_all, y_all = [], []
for sid, grp in df.groupby("SID"):
    if len(grp) > WINDOW + 1:
        Xg, yg = make_sequences(grp)
        X_all.extend(Xg)
        y_all.extend(yg)

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.float32)
print(f"  Total samples : {len(X_all):,}")
print(f"  Feature shape : {X_all.shape}  → 5 past (lat,lon) pairs")
print(f"  Target shape  : {y_all.shape}  → next (lat,lon)")

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT  (80 / 20)
# ─────────────────────────────────────────────
split = int(0.8 * len(X_all))
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

scaler_X = StandardScaler().fit(X_train)
X_train_s = scaler_X.transform(X_train)
X_test_s  = scaler_X.transform(X_test)

print(f"\n[STEP 3] Train/Test Split")
print(f"  Training samples : {len(X_train):,}")
print(f"  Testing  samples : {len(X_test):,}")

# ─────────────────────────────────────────────
# 5. MODEL TRAINING — XGBoost-style GBM
#    (In production environment, replace with:
#     from xgboost import XGBRegressor)
# ─────────────────────────────────────────────
print("\n[STEP 4] Training XGBoost Regressor (Gradient Boosted Trees) ...")

def train_model(X_tr, y_col):
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_tr, y_col)
    return model

model_lat = train_model(X_train_s, y_train[:, 0])
model_lon = train_model(X_train_s, y_train[:, 1])
print("  Latitude  model : trained")
print("  Longitude model : trained")

# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
pred_lat = model_lat.predict(X_test_s)
pred_lon = model_lon.predict(X_test_s)

rmse_lat = np.sqrt(mean_squared_error(y_test[:, 0], pred_lat))
rmse_lon = np.sqrt(mean_squared_error(y_test[:, 1], pred_lon))
overall  = np.sqrt(mean_squared_error(
    np.stack([y_test[:, 0], y_test[:, 1]], axis=1),
    np.stack([pred_lat, pred_lon],         axis=1)
))

print(f"\n[STEP 5] Model Evaluation (RMSE)")
print(f"  Latitude  RMSE : {rmse_lat:.4f} °")
print(f"  Longitude RMSE : {rmse_lon:.4f} °")
print(f"  Overall   RMSE : {overall:.4f} °")

# ─────────────────────────────────────────────
# 7. VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n[STEP 6] Generating visualizations ...")

os_sep = "/"
import os
os.makedirs("/home/claude/plots", exist_ok=True)

# --- (a) Actual vs Predicted Scatter ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Actual vs Predicted: Cyclone Position", fontsize=14, fontweight='bold')

n_show = min(2000, len(y_test))
for ax, actual, predicted, label, color in zip(
        axes,
        [y_test[:n_show, 0], y_test[:n_show, 1]],
        [pred_lat[:n_show],   pred_lon[:n_show]],
        ["Latitude (°)",      "Longitude (°)"],
        ["#2196F3",           "#FF5722"]):
    ax.scatter(actual, predicted, alpha=0.3, s=8, color=color)
    mn, mx = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, label='Perfect fit')
    ax.set_xlabel(f"Actual {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(f"{label} — RMSE={rmse_lat:.3f}°" if "Lat" in label else f"{label} — RMSE={rmse_lon:.3f}°")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/plots/actual_vs_predicted.png", dpi=150, bbox_inches='tight')
plt.close()

# --- (b) Error Distribution ---
err_lat = y_test[:, 0] - pred_lat
err_lon = y_test[:, 1] - pred_lon

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Prediction Error Distribution", fontsize=14, fontweight='bold')
for ax, err, label, color in zip(
        axes,
        [err_lat, err_lon],
        ["Latitude Error (°)", "Longitude Error (°)"],
        ["#4CAF50", "#9C27B0"]):
    ax.hist(err, bins=80, color=color, alpha=0.75, edgecolor='white')
    ax.axvline(0, color='black', linestyle='--', lw=1.5)
    ax.set_xlabel(label)
    ax.set_ylabel("Frequency")
    ax.set_title(f"{label}\nMean={err.mean():.3f}  Std={err.std():.3f}")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/plots/error_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# --- (c) Sample Cyclone Track: Actual vs Predicted ---
# Pick a single cyclone with enough points
example_sid = None
for sid, grp in df.groupby("SID"):
    if len(grp) >= 20:
        example_sid = sid
        example_grp = grp
        break

if example_sid:
    Xg, yg = make_sequences(example_grp)
    Xg = np.array(Xg, dtype=np.float32)
    yg = np.array(yg, dtype=np.float32)
    Xg_s = scaler_X.transform(Xg)
    pred_lat_ex = model_lat.predict(Xg_s)
    pred_lon_ex = model_lon.predict(Xg_s)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(yg[:, 1], yg[:, 0], 'b-o', markersize=4, label='Actual Path', lw=2)
    ax.plot(pred_lon_ex, pred_lat_ex, 'r--s', markersize=4, label='Predicted Path', lw=2)
    ax.plot(yg[0, 1], yg[0, 0], 'g^', markersize=12, label='Start', zorder=5)
    ax.plot(yg[-1, 1], yg[-1, 0], 'k*', markersize=12, label='End', zorder=5)
    ax.set_xlabel("Longitude (°)", fontsize=12)
    ax.set_ylabel("Latitude (°)", fontsize=12)
    ax.set_title(f"Cyclone Track: Actual vs Predicted\nStorm: {example_sid}", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/claude/plots/track_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

# --- (d) Global Cyclone Track Map ---
fig, ax = plt.subplots(figsize=(16, 8), facecolor='#0d1b2a')
ax.set_facecolor('#0d1b2a')

# Draw simple coastline approximation using lat/lon scatter
basins = df['BASIN'].unique()
cmap = plt.cm.get_cmap('tab10', len(basins))
for i, basin in enumerate(basins):
    sub = df[df['BASIN'] == basin]
    ax.scatter(sub['LON'], sub['LAT'], s=0.3, alpha=0.4, color=cmap(i), label=basin)

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.axhline(0, color='white', lw=0.5, alpha=0.3)
ax.set_xlabel("Longitude", color='white', fontsize=11)
ax.set_ylabel("Latitude",  color='white', fontsize=11)
ax.set_title("Global Cyclone Tracks — IBTrACS Dataset", color='white', fontsize=14, fontweight='bold')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('white')
legend = ax.legend(title="Basin", loc='lower left', fontsize=8,
                   facecolor='#1a2a3a', labelcolor='white', title_fontsize=9)
legend.get_title().set_color('white')
plt.tight_layout()
plt.savefig("/home/claude/plots/global_tracks.png", dpi=150, bbox_inches='tight')
plt.close()

print("  Plots saved to /home/claude/plots/")

# ─────────────────────────────────────────────
# 8. SAVE MODEL ARTIFACTS (for Streamlit app)
# ─────────────────────────────────────────────
import pickle, os

os.makedirs("/home/claude/model_artifacts", exist_ok=True)
with open("/home/claude/model_artifacts/model_lat.pkl", "wb") as f:
    pickle.dump(model_lat, f)
with open("/home/claude/model_artifacts/model_lon.pkl", "wb") as f:
    pickle.dump(model_lon, f)
with open("/home/claude/model_artifacts/scaler_X.pkl", "wb") as f:
    pickle.dump(scaler_X, f)

print("\n  Model artifacts saved to /home/claude/model_artifacts/")
print("\n✅ Pipeline complete!")
