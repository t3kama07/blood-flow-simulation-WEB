import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

"""
Model: Dimensionless Healthy-Domain PDE (K1 = 0)

Equations:
    a_tau + q_x = 0
    q_tau + a_x + K3*q = 0

Numerical Method:
    MacCormack predictor-corrector scheme
"""

K3 = 0.0002          # damping coefficient
N  = 400             # number of spatial points
dx = 1.0 / N         # grid spacing
x  = (np.arange(N) + 0.5) * dx   # cell-centered coordinates

tau_final = 2.6
CFL = 0.4
dt  = CFL * dx
Nt  = int(tau_final / dt)
dt  = tau_final / Nt             # adjust dt so final time is exact

# solution arrays with ghost cells
a = np.zeros(N + 2)
q = np.zeros(N + 2)

"""
Initial Conditions:
Cosine-shaped bumps for a(x) and q(x)
"""
def bump(x, center, width, amp):
    s = (x - center) / width
    out = np.zeros_like(x)
    mask = np.abs(s) < 1
    out[mask] = amp * 0.5 * (1 + np.cos(np.pi * s[mask]))
    return out

eps = 0.020
a[1:-1] = bump(x, center=0.7, width=eps, amp=0.10)
q[1:-1] = bump(x, center=0.4, width=eps, amp=0.02)

"""
Boundary Conditions:
Open (zero-gradient) boundaries
"""
def apply_bc(a, q):
    a[0]  = a[1]
    q[0]  = q[1]
    a[-1] = a[-2]
    q[-1] = q[-2]

apply_bc(a, q)

"""
MacCormack Predictor-Corrector Method
"""
def mac_cormack(a, q):
    a_p = a.copy()   # predictor arrays
    q_p = q.copy()

    # Predictor step (forward differences)
    for i in range(1, N + 1):
        a_p[i] = a[i] - (dt/dx)*(q[i+1] - q[i])
        q_p[i] = q[i] - (dt/dx)*(a[i+1] - a[i]) - dt*K3*q[i]

    apply_bc(a_p, q_p)

    # Corrector step (backward differences + averaging)
    a_new = a.copy()
    q_new = q.copy()

    for i in range(1, N + 1):
        a_new[i] = 0.5*(a[i] + a_p[i] - (dt/dx)*(q_p[i] - q_p[i-1]))
        q_new[i] = 0.5*(q[i] + q_p[i]
                        - (dt/dx)*(a_p[i] - a_p[i-1])
                        - dt*K3*q_p[i])

    return a_new, q_new


"""
Simulation Loop:
Store snapshots for slider-based visualization
"""
a_hist = []
q_hist = []
t_hist = []

save_every = 50
tau = 0.0

for n in range(Nt + 1):

    # store frames at intervals
    if n % save_every == 0:
        a_hist.append(a[1:-1].copy())
        q_hist.append(q[1:-1].copy())
        t_hist.append(tau)

    apply_bc(a, q)
    a, q = mac_cormack(a, q)
    tau += dt


"""
Interactive Plotting with Slider
"""
a_arr = np.array(a_hist)
q_arr = np.array(q_hist)
times = np.array(t_hist)

fig, ax = plt.subplots(figsize=(10,5))
plt.subplots_adjust(bottom=0.25)

idx0 = 0   # initial frame index

line_q, = ax.plot(x, q_arr[idx0], color='tab:blue', label="q(x, τ)")
line_a, = ax.plot(x, a_arr[idx0], color='tab:orange', label="a(x, τ)")

ax.set_xlabel("x (dimensionless)")
ax.set_ylabel("Dimensionless Value")
ax.set_title(f"τ = {times[idx0]:.5f}")
ax.grid(True)
ax.legend()

"""
Set global y-limits to avoid cropping during animation
"""
ymin = min(a_arr.min(), q_arr.min())
ymax = max(a_arr.max(), q_arr.max())
padding = 0.2 * (ymax - ymin)
ax.set_ylim(ymin - padding, ymax + padding)

"""
Time Slider Widget
"""
slider_ax = plt.axes([0.15, 0.10, 0.70, 0.03])
time_slider = Slider(
    ax=slider_ax,
    label="Time Index",
    valmin=0,
    valmax=len(times) - 1,
    valinit=idx0,
    valstep=1
)

def update(val):
    i = int(time_slider.val)
    line_q.set_ydata(q_arr[i])
    line_a.set_ydata(a_arr[i])
    ax.set_title(f"MacCormack scheme, τ = {times[i]:.5f}")
    fig.canvas.draw_idle()

time_slider.on_changed(update)

plt.show()
