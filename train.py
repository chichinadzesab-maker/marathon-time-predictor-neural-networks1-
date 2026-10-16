"""
train.py — Run this once to train FC-FFNN and RBF models
and save weights for the app.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

np.random.seed(42)

# ── Features ──────────────────────────────────────────────
FEATURES = [
    'age', 'running_experience_months', 'previous_marathon_count',
    'motivation_level', 'weekly_mileage_km', 'runs_per_week',
    'long_run_distance_km', 'speed_work_sessions_per_week',
    'rest_days_per_week', 'training_adherence_pct',
    'consecutive_weeks_no_miss', 'resting_heart_rate_bpm',
    'bmi', 'recovery_score', 'injury_count',
    'training_streak_days', 'missed_workout_pct',
    'goal_completion_rate', 'mental_preparation_score'
]
TARGET = 'actual_finish_time_minutes'

# ── Load data ──────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('train.csv').dropna(subset=[TARGET])
for col in FEATURES:
    df[col] = df[col].fillna(df[col].median())

df = df.sample(n=10000, random_state=42).reset_index(drop=True)
split = 8000

X_train = df[FEATURES].values[:split]
y_train = df[TARGET].values[:split].reshape(-1, 1)
X_val   = df[FEATURES].values[split:]
y_val   = df[TARGET].values[split:].reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_val_s   = scaler_X.transform(X_val)
y_train_s = scaler_y.fit_transform(y_train)
y_val_s   = scaler_y.transform(y_val)

INPUT_DIM = X_train_s.shape[1]

# ── FC-FFNN ────────────────────────────────────────────────
def relu(x):      return np.maximum(0, x)
def relu_g(x):    return (x > 0).astype(float)

class FCFFNN:
    def __init__(self, layers, lr=0.001, epochs=100, batch=256):
        self.layers = layers
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.W, self.b = [], []
        for i in range(len(layers)-1):
            self.W.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i]))
            self.b.append(np.zeros((1, layers[i+1])))

    def forward(self, X):
        self.A, self.Z = [X], []
        cur = X
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = cur @ w + b
            self.Z.append(z)
            a = relu(z) if i < len(self.W)-1 else z
            self.A.append(a); cur = a
        return cur

    def backward(self, y):
        m = y.shape[0]
        dA = (self.A[-1] - y) / m
        for i in reversed(range(len(self.W))):
            dZ = dA * relu_g(self.Z[i]) if i < len(self.W)-1 else dA
            dW = self.A[i].T @ dZ
            db = dZ.sum(0, keepdims=True)
            dA = dZ @ self.W[i].T
            self.W[i] -= self.lr * dW
            self.b[i]  -= self.lr * db

    def fit(self, X, y):
        n = X.shape[0]
        for ep in range(self.epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, self.batch):
                Xb, yb = X[idx[s:s+self.batch]], y[idx[s:s+self.batch]]
                self.forward(Xb); self.backward(yb)
            if (ep+1) % 25 == 0:
                loss = np.mean((self.forward(X) - y)**2)
                print(f"  Epoch {ep+1}/{self.epochs}  MSE: {loss:.4f}")

    def predict(self, X): return self.forward(X)

# ── RBF ───────────────────────────────────────────────────
class RBFNetwork:
    def __init__(self, n_centers=80):
        self.n_centers = n_centers

    def _kmeans(self, X, k, max_iter=50):
        C = X[np.random.choice(len(X), k, replace=False)].copy()
        for _ in range(max_iter):
            D = np.linalg.norm(X[:,None]-C[None], axis=2)
            L = D.argmin(1)
            Cn = np.array([X[L==j].mean(0) if (L==j).any() else C[j] for j in range(k)])
            if np.allclose(C, Cn, atol=1e-5): break
            C = Cn
        return C

    def fit(self, X, y):
        print(f"  K-means: {self.n_centers} centers...")
        self.centers = self._kmeans(X, self.n_centers)
        D = np.linalg.norm(X[:,None]-self.centers[None], axis=2)
        self.width = D.min(1).mean() + 1e-6
        Phi = np.exp(-D**2 / (2*self.width**2))
        Phi_b = np.hstack([Phi, np.ones((len(Phi),1))])
        lam = 1e-3
        self.W = np.linalg.solve(Phi_b.T@Phi_b + lam*np.eye(Phi_b.shape[1]), Phi_b.T@y)
        loss = np.mean((Phi_b@self.W - y)**2)
        print(f"  Train MSE: {loss:.4f}")

    def predict(self, X):
        D   = np.linalg.norm(X[:,None]-self.centers[None], axis=2)
        Phi = np.exp(-D**2 / (2*self.width**2))
        return np.hstack([Phi, np.ones((len(Phi),1))]) @ self.W

# ── Train ─────────────────────────────────────────────────
print("\nTraining FC-FFNN...")
ffnn = FCFFNN([INPUT_DIM, 64, 32, 1], lr=0.001, epochs=100, batch=256)
ffnn.fit(X_train_s, y_train_s)

print("\nTraining RBF...")
rbf = RBFNetwork(n_centers=80)
rbf.fit(X_train_s, y_train_s)

# ── Validate ──────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, r2_score

for model, name in [(ffnn, "FC-FFNN"), (rbf, "RBF")]:
    pred_s = model.predict(X_val_s)
    pred   = scaler_y.inverse_transform(pred_s.reshape(-1,1)).flatten()
    rmse   = np.sqrt(mean_squared_error(y_val.flatten(), pred))
    r2     = r2_score(y_val.flatten(), pred)
    print(f"{name}  RMSE: {rmse:.2f} min  R²: {r2:.4f}")

# ── Architecture experiments ──────────────────────────────
print("\nRunning architecture experiments...")
from sklearn.metrics import mean_absolute_error

def get_rmse(model, X, y_orig):
    pred_s = model.predict(X)
    pred   = scaler_y.inverse_transform(pred_s.reshape(-1,1)).flatten()
    return np.sqrt(mean_squared_error(y_orig.flatten(), pred))

arch_configs = [
    [INPUT_DIM,16,1],[INPUT_DIM,32,1],[INPUT_DIM,64,1],
    [INPUT_DIM,64,32,1],[INPUT_DIM,128,64,1],[INPUT_DIM,128,64,32,1]
]
arch_labels = ["1L-16","1L-32","1L-64","2L-64/32","2L-128/64","3L-128/64/32"]
arch_rmses  = []
for cfg, lbl in zip(arch_configs, arch_labels):
    m = FCFFNN(cfg, lr=0.001, epochs=60, batch=256)
    m.fit(X_train_s, y_train_s)
    arch_rmses.append(get_rmse(m, X_val_s, y_val))
    print(f"  {lbl}: RMSE {arch_rmses[-1]:.2f}")

rbf_centers_list = [20, 40, 60, 80, 100]
rbf_rmses = []
for nc in rbf_centers_list:
    m = RBFNetwork(n_centers=nc)
    m.fit(X_train_s, y_train_s)
    rbf_rmses.append(get_rmse(m, X_val_s, y_val))
    print(f"  RBF centers={nc}: RMSE {rbf_rmses[-1]:.2f}")

# ── Other factors ─────────────────────────────────────────
print("\nRunning other factor experiments...")
lr_values, lr_rmses = [0.1, 0.01, 0.001, 0.0001], []
for lr in lr_values:
    m = FCFFNN([INPUT_DIM,64,32,1], lr=lr, epochs=60, batch=256)
    m.fit(X_train_s, y_train_s)
    lr_rmses.append(get_rmse(m, X_val_s, y_val))
    print(f"  LR={lr}: RMSE {lr_rmses[-1]:.2f}")

size_fracs, size_rmses = [0.1, 0.25, 0.5, 0.75, 1.0], []
n_full = X_train_s.shape[0]
for frac in size_fracs:
    n = max(100, int(n_full*frac))
    m = FCFFNN([INPUT_DIM,64,32,1], lr=0.001, epochs=60, batch=256)
    m.fit(X_train_s[:n], y_train_s[:n])
    size_rmses.append(get_rmse(m, X_val_s, y_val))
    print(f"  Data {int(frac*100)}%: RMSE {size_rmses[-1]:.2f}")

# ── Final predictions for plots ───────────────────────────
ffnn_pred_s = ffnn.predict(X_val_s)
ffnn_pred   = scaler_y.inverse_transform(ffnn_pred_s.reshape(-1,1)).flatten()
rbf_pred_s  = rbf.predict(X_val_s)
rbf_pred    = scaler_y.inverse_transform(rbf_pred_s.reshape(-1,1)).flatten()
y_true      = y_val.flatten()

ffnn_rmse = np.sqrt(mean_squared_error(y_true, ffnn_pred))
ffnn_mae  = mean_absolute_error(y_true, ffnn_pred)
ffnn_r2   = r2_score(y_true, ffnn_pred)
rbf_rmse  = np.sqrt(mean_squared_error(y_true, rbf_pred))
rbf_mae   = mean_absolute_error(y_true, rbf_pred)
rbf_r2    = r2_score(y_true, rbf_pred)

# ── Generate training loss ─────────────────────────────────
print("\nGenerating loss history...")
ffnn_train_losses, ffnn_val_losses = [], []
ffnn2 = FCFFNN([INPUT_DIM,64,32,1], lr=0.001, epochs=100, batch=256)
n = X_train_s.shape[0]
for ep in range(100):
    idx = np.random.permutation(n)
    for s in range(0, n, 256):
        Xb = X_train_s[idx[s:s+256]]
        yb = y_train_s[idx[s:s+256]]
        ffnn2.forward(Xb); ffnn2.backward(yb)
    tl = np.mean((ffnn2.forward(X_train_s) - y_train_s)**2)
    vl = np.mean((ffnn2.forward(X_val_s)   - y_val_s)**2)
    ffnn_train_losses.append(tl)
    ffnn_val_losses.append(vl)

rbf2 = RBFNetwork(n_centers=80)
rbf2.fit(X_train_s, y_train_s)
rbf_train_loss = float(np.mean((rbf2.predict(X_train_s) - y_train_s)**2))
rbf_val_loss   = float(np.mean((rbf2.predict(X_val_s)   - y_val_s)**2))

# ── PLOTS ─────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
C_FFNN, C_RBF, C_ACC = '#2196F3', '#FF5722', '#4CAF50'

print("\nGenerating plots...")

# Plot 1: Loss curves
fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle('Training Loss Curves', fontsize=15, fontweight='bold')
axes[0].plot(ffnn_train_losses, color=C_FFNN, label='Train')
axes[0].plot(ffnn_val_losses,   color=C_FFNN, ls='--', label='Val')
axes[0].set_title('FC-FFNN (MSE per epoch, normalized scale)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE'); axes[0].legend()
axes[1].bar(['Train MSE','Val MSE'], [rbf_train_loss, rbf_val_loss], color=C_RBF, alpha=0.8)
axes[1].set_title('RBF — Closed-form solution (single-shot training)')
axes[1].set_ylabel('MSE (normalized)')
plt.tight_layout()
plt.savefig('plot1_loss_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Pred vs Actual
fig, axes = plt.subplots(1,2,figsize=(14,6))
fig.suptitle('Predicted vs Actual Finish Time', fontsize=15, fontweight='bold')
lims = [y_true.min()-10, y_true.max()+10]
for ax, pred, name, col, r2 in [
    (axes[0], ffnn_pred,'FC-FFNN',C_FFNN,ffnn_r2),
    (axes[1], rbf_pred, 'RBF',    C_RBF, rbf_r2)]:
    ax.scatter(y_true, pred, alpha=0.15, s=5, color=col)
    ax.plot(lims, lims, 'k--', lw=1.5, label='Perfect')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual (min)'); ax.set_ylabel('Predicted (min)')
    ax.set_title(f'{name}  R²={r2:.3f}'); ax.legend()
plt.tight_layout()
plt.savefig('plot2_pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Architecture
fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle('Architecture Impact on RMSE', fontsize=15, fontweight='bold')
bars = axes[0].bar(arch_labels, arch_rmses, color=C_FFNN, alpha=0.85)
axes[0].set_title('FC-FFNN Configurations')
axes[0].set_ylabel('RMSE (min)'); axes[0].tick_params(axis='x', rotation=25)
for bar,v in zip(bars,arch_rmses): axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f'{v:.1f}', ha='center', fontsize=9)
bars = axes[1].bar([str(n) for n in rbf_centers_list], rbf_rmses, color=C_RBF, alpha=0.85)
axes[1].set_title('RBF: Number of Centers')
axes[1].set_ylabel('RMSE (min)')
for bar,v in zip(bars,rbf_rmses): axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f'{v:.1f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('plot3_architecture.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Other factors
fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle('Other Factors Affecting Quality', fontsize=15, fontweight='bold')
axes[0].plot([str(lr) for lr in lr_values], lr_rmses, marker='o', color=C_FFNN, lw=2, ms=8)
axes[0].set_title('Learning Rate vs RMSE'); axes[0].set_xlabel('Learning Rate'); axes[0].set_ylabel('RMSE (min)')
for x,yp in zip([str(lr) for lr in lr_values], lr_rmses):
    axes[0].annotate(f'{yp:.1f}', (x,yp), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)
axes[1].plot([int(f*100) for f in size_fracs], size_rmses, marker='s', color=C_ACC, lw=2, ms=8)
axes[1].set_title('Training Size vs RMSE'); axes[1].set_xlabel('Data Used (%)'); axes[1].set_ylabel('RMSE (min)')
for x,yp in zip([int(f*100) for f in size_fracs], size_rmses):
    axes[1].annotate(f'{yp:.1f}', (x,yp), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('plot4_other_factors.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 5: Comparison
fig, axes = plt.subplots(1,3,figsize=(15,5))
fig.suptitle('FC-FFNN vs RBF Final Comparison', fontsize=15, fontweight='bold')
for ax, metric, fv, rv in zip(axes,
    ['RMSE (min)','MAE (min)','R²'],
    [ffnn_rmse,ffnn_mae,ffnn_r2],
    [rbf_rmse, rbf_mae, rbf_r2]):
    bars = ax.bar(['FC-FFNN','RBF'],[fv,rv], color=[C_FFNN,C_RBF], alpha=0.85, width=0.4)
    ax.set_title(metric)
    for bar,v in zip(bars,[fv,rv]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01, f'{v:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('plot5_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 6: Error distribution
fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle('Prediction Error Distribution', fontsize=15, fontweight='bold')
for ax, pred, name, col in [(axes[0],ffnn_pred,'FC-FFNN',C_FFNN),(axes[1],rbf_pred,'RBF',C_RBF)]:
    errs = pred - y_true
    ax.hist(errs, bins=60, color=col, alpha=0.75, edgecolor='white')
    ax.axvline(0, color='black', lw=1.5, ls='--')
    ax.axvline(errs.mean(), color='red', lw=1.5, label=f'Mean: {errs.mean():.1f} min')
    ax.set_title(f'{name} Errors'); ax.set_xlabel('Error (min)'); ax.set_ylabel('Count'); ax.legend()
plt.tight_layout()
plt.savefig('plot6_error_dist.png', dpi=150, bbox_inches='tight')
plt.close()

print("All 6 plots saved to marathon folder.")

# ── Save everything with pickle ───────────────────────────
import pickle
print("\nSaving models...")

data = {
    'ffnn_W': ffnn.W,
    'ffnn_b': ffnn.b,
    'ffnn_layers': ffnn.layers,
    'rbf_centers': rbf.centers,
    'rbf_W': rbf.W,
    'rbf_width': rbf.width,
    'scaler_X_mean': scaler_X.mean_,
    'scaler_X_scale': scaler_X.scale_,
    'scaler_y_mean': scaler_y.mean_,
    'scaler_y_scale': scaler_y.scale_,
}
with open('model.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Done! model.pkl saved.")
print("You can now run: python marathon_app.py")