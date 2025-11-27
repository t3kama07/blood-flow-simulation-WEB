import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0
Nx = 400
dz = L / Nx
z = np.linspace(0, L, Nx+1)

Tfinal = 2.6
c = 1.0
CFL = 0.4
dt = CFL * dz / c
Nt = int(Tfinal / dt) + 1
dt = Tfinal / Nt

epsilon = 0.02

# Smooth cosine bump
def bump(z, center, eps, amp=1.0):
    s = (z - center) / eps
    out = np.zeros_like(z)
    mask = np.abs(s) < 1
    out[mask] = amp * 0.5 * (1 + np.cos(np.pi * s[mask]))
    return out

# Initial fields
Q = bump(z, 0.4, epsilon, amp=1.0)
A = bump(z, 0.7, epsilon, amp=1.0)

#  points Q(t), A(t)
track_positions = [0.25, 0.50, 0.75]
track_indices = [np.argmin(np.abs(z - xp)) for xp in track_positions]

# Time arrays to store results
time = []
Q_track = {xp: [] for xp in track_positions}
A_track = {xp: [] for xp in track_positions}

# TIME INTEGRATIONMacCormack Schem 
t = 0.0
for n in range(Nt + 1):

    # Store current values
    time.append(t)
    for xp, idx in zip(track_positions, track_indices):
        Q_track[xp].append(Q[idx])
        A_track[xp].append(A[idx])

    # Apply approximate Neumann BCs
    Q[0], Q[-1] = Q[1], Q[-2]
    A[0], A[-1] = A[1], A[-2]

    # Predictor
    Qp = Q.copy()
    Ap = A.copy()

    Qp[:-1] = Q[:-1] - dt/dz * (A[1:] - A[:-1]) - dt*(1/5)*Q[:-1]
    Ap[:-1] = A[:-1] - dt/dz * (Q[1:] - Q[:-1])

    # Reapply BCs to predictors
    Qp[0], Qp[-1] = Qp[1], Qp[-2]
    Ap[0], Ap[-1] = Ap[1], Ap[-2]

    # Corrector
    Qnew = Q.copy()
    Anew = A.copy()

    Qnew[1:] = 0.5 * (Q[1:] + Qp[1:]
                      - dt/dz * (Ap[1:] - Ap[:-1])
                      - dt*(1/5)*Qp[1:])
    Anew[1:] = 0.5 * (A[1:] + Ap[1:]
                      - dt/dz * (Qp[1:] - Qp[:-1]))

    # Apply BCs after correction
    Qnew[0], Qnew[-1] = Qnew[1], Qnew[-2]
    Anew[0], Anew[-1] = Anew[1], Anew[-2]

    Q, A = Qnew, Anew
    t += dt

#plots
for xp in track_positions:
    plt.figure(figsize=(8,4))
    plt.plot(time, Q_track[xp], label=f"Q(t) at z={xp}")
    plt.plot(time, A_track[xp], label=f"A(t) at z={xp}")
    plt.xlabel("Time t")
    plt.ylabel("Value")
    plt.title(f"Time series at z = {xp}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


#need to  add units
#Need to recheck codes 18/11/2025