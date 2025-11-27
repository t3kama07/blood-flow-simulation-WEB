import numpy as np
import matplotlib.pyplot as plt

 
"""
1. Physical and Numerical Parameters (Healthy Artery Model)

Includes:
- Geometry
- Time-stepping
- Material properties
- Wall stiffness
- Wave speed and damping
- Windkessel parameters
"""

# Geometry
L = 0.15                 # artery length [m]
dz = 1.0e-3              # spatial step [m]
Nx = int(L/dz) + 1       # number of grid points

# Time (one or multiple heart cycles)
T_heart = 1.0            # heart period [s]
N_cycles = 1             # change this to run more cycles
T_final = N_cycles * T_heart
dt = 1.0e-5              # time step [s]
Nt = int(T_final / dt)

# Material / fluid properties
rho = 1060.0             # density [kg/m³]
mu  = 3.5e-3             # viscosity [Pa·s]

# Reference geometry (ACA artery)
D_ref = 3.0e-3           # diameter [m]
r_ref = D_ref / 2.0
A_ref = np.pi * r_ref**2 # cross-sectional area [m²]

# Wall stiffness (tube law)
E = 1.5e6                # Young's modulus [Pa]
h = 3.0e-4               # wall thickness [m]
alpha = E * h / (2.0 * np.pi * r_ref**3)

# Wave speed and damping
c0 = np.sqrt(alpha * A_ref / rho)
delta = 8.0 * np.pi * mu / (rho * A_ref)

print(f"c0 = {c0:.2f} m/s,   delta = {delta:.3f} 1/s")
print(f"CFL = {c0*dt/dz:.3f}")

# Windkessel (3-element) outlet parameters
Rp   = 6.7e8             # proximal resistance
Rd   = 1.0e10            # distal resistance
Cw   = 1.5e-11           # compliance
Lint = 1.0e4             # inertance

"""
Inlet Pressure (Blackman-Harris Pulse Modulation)
Uses Table 1 timing parameters for systolic & diastolic shapes.
"""

# Timing parameters (Table 1)
LD = 0.60; LP = 0.55; LT = 0.55   
tP = 0.38; tD = 0.05; tT = 0.20    
betaD = 0.4; betaP = 1.0; betaT = 0.3
c_rel = 60.0/60.0                   # 60 BPM normalization

# Amplitudes: mmHg → Pa
mmHg_to_Pa = 133.322
A_mmHg = 50.0
A_P = A_D = A_T = A_mmHg * mmHg_to_Pa

# Base diastolic pressure
P_dias = 87.0 * mmHg_to_Pa
P_ref = P_dias

"""
Blackman-Harris Window Definition
Used to shape the pressure waveform smoothly.
"""

a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168

def bh_window(t_local):
    """4-term Blackman-Harris window, normalized over [0, T_heart]."""
    if t_local < 0 or t_local > T_heart:
        return 0.0
    tau = t_local / T_heart
    return (a0
           - a1 * np.cos(2*np.pi*tau)
           + a2 * np.cos(4*np.pi*tau)
           - a3 * np.cos(6*np.pi*tau))

# normalization constant
tau_grid = np.linspace(0, T_heart, 2001)
w_vals = np.array([bh_window(t) for t in tau_grid])
w_max = np.max(w_vals)

def inlet_pressure(t):
    """
    Full inlet pressure waveform:
    diastolic base + P, T, D pulses,
    each modulated by a Blackman–Harris window.
    """
    t_mod = t % T_heart

    def pulse(Ai, beta, Li, ti):
        arg = (t_mod - ti) / (Li / c_rel)
        return Ai * beta * bh_window(arg) / w_max

    return (P_dias
            + pulse(A_P, betaP, LP, tP)
            + pulse(A_D, betaD, LD, tD)
            + pulse(A_T, betaT, LT, tT))

"""
3. Spatial Grid and Monitoring Setup
"""

z = np.linspace(0, L, Nx)

A_tilde = np.zeros(Nx)   # area perturbation
Q_tilde = np.zeros(Nx)   # flow perturbation

# Monitor at inlet, midpoint, outlet
monitor_z = np.array([0.0, L/2, L])
monitor_idx = [np.argmin(np.abs(z - zz)) for zz in monitor_z]

# histories for each location
A_hist_multi = [[] for _ in monitor_z]
Q_hist_multi = [[] for _ in monitor_z]
P_hist_multi = [[] for _ in monitor_z]

# outlet time derivative buffers for Windkessel
Q_out_hist = np.zeros(3)

# global time histories
P_out_hist = []
Q_out_hist_rec = []
P_wk_hist = []
t_hist = []

"""
4. Linearized Flux Function (β = 0 case)

System:
    A_t + Q_z = 0
    Q_t + c0² A_z = -δQ   
"""

def flux(A_t, Q_t):
    """Return the homogeneous flux components."""
    F1 = Q_t
    F2 = c0**2 * A_t
    return F1, F2

"""
5. MacCormack Predictor-Corrector Time Stepping
"""

for n in range(Nt):
    t = n * dt

    # --- predictor ---
    F1, F2 = flux(A_tilde, Q_tilde)
    A_pred = A_tilde.copy()
    Q_pred = Q_tilde.copy()

    # forward differences on interior
    for i in range(0, Nx-1):
        A_pred[i] = A_tilde[i] - dt/dz * (F1[i+1] - F1[i])
        Q_pred[i] = Q_tilde[i] - dt/dz * (F2[i+1] - F2[i]) - dt * delta * Q_tilde[i]

    # inlet predictor via tube law
    P_in = inlet_pressure(t + dt)
    A_pred[0] = (P_in - P_ref) / alpha
    Q_pred[0] = Q_pred[1]

    # outlet predictor via Windkessel model
    Q_out_hist[2] = Q_out_hist[1]
    Q_out_hist[1] = Q_out_hist[0]
    Q_out_hist[0] = Q_tilde[-1]

    if n >= 2:
        dQdt   = (Q_out_hist[0] - Q_out_hist[1]) / dt
        d2Qdt2 = (Q_out_hist[0] - 2*Q_out_hist[1] + Q_out_hist[2]) / dt**2
    else:
        dQdt = d2Qdt2 = 0.0

    A_out = A_tilde[-1]

    RHS = (Lint/alpha)*d2Qdt2 \
        + (Rp + Lint/(Rd*Cw))/alpha*dQdt \
        + (1.0/Cw + Rp/(Rd*Cw))/alpha*Q_tilde[-1]

    dA_dt_out = - (1.0/(Rd*Cw))*A_out + RHS

    A_pred[-1] = A_out + dt * dA_dt_out
    Q_pred[-1] = Q_pred[-2]

    # --- corrector ---
    F1p, F2p = flux(A_pred, Q_pred)
    A_new = A_tilde.copy()
    Q_new = Q_tilde.copy()

    for i in range(1, Nx):
        A_new[i] = 0.5*(A_tilde[i] + A_pred[i]
                        - dt/dz*(F1p[i] - F1p[i-1]))
        Q_new[i] = 0.5*(Q_tilde[i] + Q_pred[i]
                        - dt/dz*(F2p[i] - F2p[i-1])
                        - dt * delta * (Q_tilde[i] + Q_pred[i]) / 2)

    # inlet corrector
    A_new[0] = (inlet_pressure(t + dt) - P_ref) / alpha
    Q_new[0] = Q_new[1]

    # outlet corrector
    A_new[-1] = A_pred[-1]
    Q_new[-1] = Q_new[-2]

    # update
    A_tilde = A_new
    Q_tilde = Q_new

    # --- record histories ---
    t_curr = t + dt
    t_hist.append(t_curr)

    P_out = P_ref + alpha * A_tilde[-1]
    P_out_hist.append(P_out)
    P_wk_hist.append(P_out)
    Q_out_hist_rec.append(Q_tilde[-1])

    # monitor at inlet, mid, outlet
    for k, idx in enumerate(monitor_idx):
        A_hist_multi[k].append(A_tilde[idx] + A_ref)
        Q_hist_multi[k].append(Q_tilde[idx])
        P_hist_multi[k].append(P_ref + alpha * A_tilde[idx])

"""
6. Convert Recorded Lists to Arrays
"""

t_hist = np.array(t_hist)
P_out_hist = np.array(P_out_hist)
Q_out_hist_rec = np.array(Q_out_hist_rec)
P_wk_hist = np.array(P_wk_hist)

"""
7-9. Visualization of Outlet Conditions and Signals
Contains:
- Outlet pressure & flow
- Windkessel boundary pressure
- Pressure/flow/area at inlet, mid, outlet
"""

# outlet pressure, flow, and area
plt.figure(figsize=(9,7))

# --- Pressure ---
plt.subplot(3,1,1)
plt.plot(t_hist, P_out_hist/mmHg_to_Pa)
plt.ylabel("Outlet Pressure [mmHg]")
plt.title("Outlet Pressure, Flow, and Area (Windkessel BC)")
plt.grid(True)

# --- Flow ---
plt.subplot(3,1,2)
plt.plot(t_hist, Q_out_hist_rec)
plt.ylabel("Outlet Q̃ [m³/s]")
plt.grid(True)

# --- Area ---
plt.subplot(3,1,3)
A_out_hist = A_ref + A_tilde[-1]  # FINAL area only (not time history)

# If you want FULL history:
# A_out_hist = A_hist_multi[-1]   # this was already stored during the loop

plt.plot(t_hist, A_hist_multi[-1])     # <- FULL area time series at outlet
plt.ylabel("Outlet Area [m²]")
plt.xlabel("Time [s]")
plt.grid(True)

plt.tight_layout()
plt.show()


# Windkessel pressure evolution
plt.figure(figsize=(8,4))
plt.plot(t_hist, P_wk_hist/mmHg_to_Pa, label="Windkessel Pressure at Outlet")
plt.xlabel("Time [s]")
plt.ylabel("Pressure [mmHg]")
plt.title("Windkessel Boundary Pressure Over One Heart Cycle")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# multi-location plots
labels = ["Inlet (0 m)", "Mid (0.075 m)", "Outlet (0.15 m)"]

plt.figure(figsize=(12,10))

plt.subplot(3,1,1)
for k in range(3):
    plt.plot(t_hist, np.array(P_hist_multi[k])/mmHg_to_Pa, label=labels[k])
plt.ylabel("Pressure [mmHg]")
plt.title("Pressure at Inlet, Midpoint, and Outlet")
plt.grid(True)
plt.legend()

plt.subplot(3,1,2)
for k in range(3):
    plt.plot(t_hist, Q_hist_multi[k], label=labels[k])
plt.ylabel("Flow Q̃ [m³/s]")
plt.title("Flow at Inlet, Midpoint, and Outlet")
plt.grid(True)
plt.legend()

plt.subplot(3,1,3)
for k in range(3):
    plt.plot(t_hist, A_hist_multi[k], label=labels[k])
plt.ylabel("Area [m²]")
plt.xlabel("Time [s]")
plt.title("Area at Inlet, Midpoint, and Outlet")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
