import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5
x0, y0, T = 10.0, 5.0, 20.0

def f(t, s):
    x, y = s
    return np.array([alpha*x - beta*x*y, delta*x*y - gamma*y])

def euler(f, s0, T, h):
    n = int(round(T / h))
    t = np.linspace(0, T, n + 1)
    s = np.zeros((n + 1, 2)); s[0] = s0
    for i in range(n):
        s[i+1] = s[i] + h * f(t[i], s[i])
    return t, s

def rk4(f, s0, T, h):
    n = int(round(T / h))
    t = np.linspace(0, T, n + 1)
    s = np.zeros((n + 1, 2)); s[0] = s0
    for i in range(n):
        k1 = f(t[i], s[i])
        k2 = f(t[i] + h/2, s[i] + h/2 * k1)
        k3 = f(t[i] + h/2, s[i] + h/2 * k2)
        k4 = f(t[i] + h, s[i] + h * k3)
        s[i+1] = s[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, s

def reference(T):
    sol = solve_ivp(f, (0, T), [x0, y0], method="DOP853",
                    rtol=1e-12, atol=1e-12)
    return sol.y[:, -1]

def conserved(s):
    x, y = s[:, 0], s[:, 1]
    return delta*x - gamma*np.log(x) + beta*y - alpha*np.log(y)

os.makedirs("results", exist_ok=True)
s0 = [x0, y0]

# Plot 1: population vs time (RK4)
t, s = rk4(f, s0, T, 0.01)
plt.figure(figsize=(8, 4))
plt.plot(t, s[:, 0], label="Prey")
plt.plot(t, s[:, 1], label="Predator")
plt.xlabel("Time"); plt.ylabel("Population")
plt.title("Lotka-Volterra (RK4, h=0.01)")
plt.legend(loc="upper right"); plt.grid(alpha=0.3)
plt.savefig("results/populations.png", dpi=150, bbox_inches="tight")

# Plot 2: phase plane, Euler (2000 steps) vs RK4 (200 steps)
te, se = euler(f, s0, T, 0.01)
tr, sr = rk4(f, s0, T, 0.1)
plt.figure(figsize=(6, 6))
plt.plot(se[:, 0], se[:, 1], label="Euler h=0.01 (2000 steps)")
plt.plot(sr[:, 0], sr[:, 1], label="RK4 h=0.1 (200 steps)")
plt.xlabel("Prey"); plt.ylabel("Predator")
plt.title("Phase plane")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("results/phase_plane.png", dpi=150, bbox_inches="tight")

# Plot 3: error vs step size (measured at t=5)
Te = 5.0
ref = reference(Te)
hs = [0.1, 0.05, 0.02, 0.01, 0.005]
err_e, err_r = [], []
for h in hs:
    err_e.append(np.linalg.norm(euler(f, s0, Te, h)[1][-1] - ref))
    err_r.append(np.linalg.norm(rk4(f, s0, Te, h)[1][-1] - ref))
    print(f"h={h:<6} Euler error={err_e[-1]:.3e}  RK4 error={err_r[-1]:.3e}")

slope_e = np.polyfit(np.log(hs), np.log(err_e), 1)[0]
slope_r = np.polyfit(np.log(hs), np.log(err_r), 1)[0]
print(f"Euler slope: {slope_e:.2f}")
print(f"RK4 slope:   {slope_r:.2f}")

plt.figure(figsize=(6, 4))
plt.loglog(hs, err_e, "o-", label=f"Euler (slope {slope_e:.2f})")
plt.loglog(hs, err_r, "s-", label=f"RK4 (slope {slope_r:.2f})")
plt.xlabel("Step size h"); plt.ylabel("Error at t=5")
plt.title("Error vs step size")
plt.legend(); plt.grid(alpha=0.3, which="both")
plt.savefig("results/error_vs_h.png", dpi=150, bbox_inches="tight")

# Plot 4 (optional): conserved quantity V over time
plt.figure(figsize=(8, 4))
plt.plot(te, conserved(se) - conserved(se)[0], label="Euler h=0.01")
plt.plot(tr, conserved(sr) - conserved(sr)[0], label="RK4 h=0.1")
plt.xlabel("Time"); plt.ylabel("V(t) - V(0)")
plt.title("Conserved quantity drift")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("results/conserved_quantity.png", dpi=150, bbox_inches="tight")
