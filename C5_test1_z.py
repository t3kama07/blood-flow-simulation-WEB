import numpy as np
import matplotlib.pyplot as plt

# PHYSICAL PARAMETERS with SI units

# Vessel geometry
D_ref = 3.0e-3                  # m
R_ref = D_ref / 2.0
A_ref = np.pi * R_ref**2        # m^2

# When flow is steady, Q_ref = 0 
Q_ref = 0                     # m^3/s  

# Wave speed for the linearized system
c = 1.0                         # m/s

# NUMERICAL DOMAIN
L = 1.0
Nx = 400
dz = L / Nx
z = np.linspace(0.0, L, Nx+1)

Tfinal = 2.6
CFL = 0.4
dt = CFL * dz / c
Nt = int(Tfinal / dt) + 1
dt = Tfinal / Nt

epsilon = 0.02

# INITIAL PERTURBATIONS
def bump(z, center, eps, amp=1.0):
    s = (z - center) / eps
    out = np.zeros_like(z)
    mask = np.abs(s) < 1
    out[mask] = amp * 0.5 * (1 + np.cos(np.pi * s[mask]))
    return out

# small perturbations
amp_A = 1 * A_ref
amp_Q = 1 * 1e-6            # considered tiny perturbation for Q since Q_ref=0

Q = amp_Q * bump(z, 0.4, epsilon)
A = amp_A * bump(z, 0.7, epsilon)

# Save times
save_times = [0.0, 0.25, 0.5, 1.0, 1.7, 2.6]
saved = {t: None for t in save_times}
saved[0.0] = (Q.copy(), A.copy())

# Flux function for:
#   Q_t + A_z + (1/5)Q = 0
#   A_t + Q_z = 0
def flux(Q, A):
    return np.vstack((A, Q))   # F = (A, Q)

# Apply boundary conditions
def apply_BC(Q, A):
    # z = 0
    A[0] = A[1]
    Q[0] = Q[1]                 

    # z = L
    A[-1] = A[-2]
    Q[-1] = Q[-2]

# MACCORMACK SOLVER
t = 0.0
for n in range(1, Nt + 1):

    apply_BC(Q, A)

    # Predictor
    FQ = A
    FA = Q

    Qp = Q.copy()
    Ap = A.copy()

    Qp[:-1] = Q[:-1] - dt/dz*(FQ[1:] - FQ[:-1]) - (dt/5)*Q[:-1]
    Ap[:-1] = A[:-1] - dt/dz*(FA[1:] - FA[:-1])

    apply_BC(Qp, Ap)

    # Corrector
    FQp = Ap
    FAp = Qp

    Q_new = 0.5*(Q + Qp - dt/dz*(FQp - np.roll(FQp,1)) - (dt/5)*Qp)
    A_new = 0.5*(A + Ap - dt/dz*(FAp - np.roll(FAp,1)))

    Q = Q_new
    A = A_new

    apply_BC(Q, A)

    t = n * dt

    for ts in save_times:
        if saved[ts] is None and t >= ts - 1e-10:
            saved[ts] = (Q.copy(), A.copy())

# PLOTTING
for ts in save_times:
    Qt, At = saved[ts]

    Q_phys = Qt                 # Q_ref = 0, so this is the physical flow
    A_phys = A_ref + At         # true physical area
    

    plt.figure(figsize=(10,4))
    plt.plot(z, Q_phys, label="Flow Q(z,t) [m³/s]")
    plt.plot(z, A_phys, label="Area A(z,t) [m²]")
    plt.title(f"Test problem at t = {ts:.2f}s")
    plt.xlabel("z [m]")
    plt.ylabel("Flow / Area")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
