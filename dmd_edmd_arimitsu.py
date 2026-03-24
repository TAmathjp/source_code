import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats.qmc import Sobol

MU = -0.05
LDA = -1.0

T_END = 20.0     # シミュレーションの終了時刻
N_T = 60         # 時間点の数
BOX = 0.8        # 初期値の範囲 [-BOX, BOX]
N_TRAJ_TRAIN = 64  # 軌道の本数
POLY_DEGREE = 2
x0_test = np.array([0.65, -0.2])


# ----------
# 1. slow manifold
# ----------
def slow_manifold(x, t, mu=MU, lda=LDA):
    x1, x2 = x
    return [mu * x1, lda * (x2 - x1**2)]


# ----------
# 2. 学習データ
# ----------
t = np.linspace(0.0, T_END, N_T, endpoint=False)
dt = t[1] - t[0]

sampler = Sobol(d=2, scramble=True, seed=1)
x0_train = (sampler.random(N_TRAJ_TRAIN) - 0.5) * (2.0 * BOX)

X_list, Y_list = [], []
for x0 in x0_train:
    sol = odeint(slow_manifold, x0, t)

    X_list.append(sol[:-1])
    Y_list.append(sol[1:])

X_train = np.vstack(X_list)
Y_train = np.vstack(Y_list)


# ----------
# 3. DMD
# ----------

def dmd_predict(X):
    return X @ A_dmd


# ----------
# 4. EDMD
# ----------
def poly_features_2d(X, degree=POLY_DEGREE):
    X = np.asarray(X)
    x1 = X[:, 0]
    x2 = X[:, 1]

    feats = []
    for total_deg in range(degree + 1):
        for e1 in range(total_deg + 1):
            e2 = total_deg - e1
            feats.append((x1**e1) * (x2**e2))

    return np.column_stack(feats)

Phi_X = poly_features_2d(X_train)
Phi_Y = poly_features_2d(Y_train)


K_edmd = np.linalg.lstsq(Phi_X, Phi_Y, rcond=None)[0]


IDX_X2 = 1
IDX_X1 = 2

def edmd_predict(X):
    Phi_next = poly_features_2d(X) @ K_edmd
    return Phi_next[:, [IDX_X1, IDX_X2]]


# ----------
# 5. 軌道生成
# ----------
def rollout(predict_fn, x0, n_steps):
    x = np.asarray(x0, dtype=float).reshape(1, -1)
    xs = [x.copy()]

    for _ in range(n_steps):
        x = predict_fn(x)
        xs.append(x.copy())

    return np.vstack(xs)


# ----------
# 6. 未知の1本の軌道に対するテスト
# ----------
t_test = np.arange(0.0, T_END, dt)
true_traj = odeint(slow_manifold, x0_test, t_test)

pred_dmd = rollout(dmd_predict, x0_test, len(t_test) - 1)
pred_edmd = rollout(edmd_predict, x0_test, len(t_test) - 1)

def mse(a, b):
    return np.mean((a - b) ** 2)

mse_dmd = mse(pred_dmd, true_traj)
mse_edmd = mse(pred_edmd, true_traj)

print(f"DMD  multi-step MSE  = {mse_dmd:.3e}")
print(f"EDMD multi-step MSE  = {mse_edmd:.3e}")




# ----------
# 7. visualize
# ----------
# (a) 状態空間での軌道比較
plt.figure(figsize=(6, 5))
plt.plot(true_traj[:, 0], true_traj[:, 1], "-k", lw=2, label="True")
plt.plot(pred_dmd[:, 0], pred_dmd[:, 1], "--b", lw=2, label="DMD")
plt.plot(pred_edmd[:, 0], pred_edmd[:, 1], "--r", lw=2, label=f"EDMD(Poly{POLY_DEGREE})")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Slow manifold: DMD vs EDMD")
plt.legend()
plt.tight_layout()
plt.show()

# (b) x2 の時系列比較
plt.figure(figsize=(7, 3))
plt.plot(t_test, true_traj[:, 1], "-k", lw=2, label="True x2")
plt.plot(t_test, pred_dmd[:, 1], "--b", lw=2, label="DMD x2")
plt.plot(t_test, pred_edmd[:, 1], "--r", lw=2, label=f"EDMD x2")
plt.xlabel("t")
plt.ylabel("x2")
plt.title("Time series comparison")
plt.legend()
plt.tight_layout()
plt.show()

# (c) Indicator
# 厳密な slow manifold は x2 = (10/9) x1^2 である。
# ここでは簡単のため，その近似的な指標として x2 - x1^2 を描く。
# したがって，この図は厳密な manifold からのずれではなく，「x2 が x1^2 型の関係からどれだけ外れるか」の目安を与える。
plt.figure(figsize=(7, 3))
plt.plot(t_test, true_traj[:, 1] - true_traj[:, 0]**2, "-k", label="True")
plt.plot(t_test, pred_dmd[:, 1] - pred_dmd[:, 0]**2, "--b", label="DMD")
plt.plot(t_test, pred_edmd[:, 1] - pred_edmd[:, 0]**2, "--r", label="EDMD")
plt.xlabel("t")
plt.ylabel(r"$x_2 - x_1^2$")
plt.title("Indicator")
plt.legend()
plt.tight_layout()
plt.show()
