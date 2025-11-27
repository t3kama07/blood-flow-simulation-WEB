from flask import Flask, render_template, request, jsonify
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from io import StringIO
import sys

app = Flask(__name__)

def run_simulation(param_dict=None):
    """Run blood flow simulation with optional parameters"""
    
    # Default parameters (from simulation_t.py)
    rho = 1060.0
    mu = 3.5e-3
    L = 0.15
    D_ref = 3.0e-3
    r_ref = D_ref / 2.0
    E = 1.5e6
    h = 1.0e-4
    A_ref = math.pi * r_ref**2
    alpha = E * h / (2.0 * math.pi * r_ref**3)
    c = math.sqrt(alpha * A_ref / rho)
    
    # Override with user parameters if provided
    if param_dict:
        L = float(param_dict.get('L', L))
        E = float(param_dict.get('E', E))
        D_ref = float(param_dict.get('D_ref', D_ref))
        r_ref = D_ref / 2.0
        A_ref = math.pi * r_ref**2
        alpha = E * h / (2.0 * math.pi * r_ref**3)
        c = math.sqrt(alpha * A_ref / rho)
    
    dx = 1.0e-3
    dt = 1.0e-5
    T_final = 1.0
    Nz = int(L / dx)
    Nzp1 = Nz + 1
    Nt = int(T_final / dt)
    z = np.linspace(0.0, L, Nzp1)
    
    d = 8.0 * math.pi * mu / (rho * A_ref)
    
    def blackman_harris(x):
        if x < 0.0 or x > 1.0:
            return 0.0
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        return a0 - a1*math.cos(2*math.pi*x) + a2*math.cos(4*math.pi*x) - a3*math.cos(6*math.pi*x)

    def p_inlet(t):
        LD, LP, LT = 0.60, 0.55, 0.55
        tP, tD, tT = 0.38, 0.05, 0.20
        betaP, betaD, betaT = 1.0, 0.4, 0.3
        A_total_mmHg = 50.0
        mmHg_to_Pa = 133.322
        A_total = A_total_mmHg * mmHg_to_Pa
        
        xi_P = (t - tP) / LP
        xi_T = (t - tT) / LT
        xi_D = (t - tD) / LD
        wP = blackman_harris(xi_P)
        wT = blackman_harris(xi_T)
        wD = blackman_harris(xi_D)
        return A_total * (betaP*wP + betaT*wT + betaD*wD)

    def A_inlet(t):
        p = p_inlet(t)
        return A_ref + p / alpha
    
    # Initialize
    A = np.full(Nzp1, A_ref)
    Q = np.zeros(Nzp1)
    
    probe_z = np.array([0.0, 0.2*L, 0.4*L, 0.6*L, 0.8*L, L])
    probe_idx = [int(zp / dx) for zp in probe_z]
    probe_labels = [f"{zp/L:.1f}L" for zp in probe_z]
    
    time_store = []
    p_store = [[] for _ in probe_idx]
    Q_store = [[] for _ in probe_idx]
    A_store = [[] for _ in probe_idx]
    
    # Time integration
    for n in range(Nt):
        t = n * dt
        
        A[0] = A_inlet(t)
        Q[0] = Q[1]
        A[-1] = A[-2]
        Q[-1] = Q[-2]
        
        A_star = A.copy()
        Q_star = Q.copy()
        for i in range(Nz):
            A_star[i] = A[i] - (dt/dx) * (Q[i+1] - Q[i])
            Q_star[i] = Q[i] - (c**2 * dt/dx) * (A[i+1] - A[i]) - d * dt * Q[i]
        
        A_star[0] = A_inlet(t + dt)
        Q_star[0] = Q_star[1]
        A_star[-1] = A_star[-2]
        Q_star[-1] = Q_star[-2]
        
        A_new = A.copy()
        Q_new = Q.copy()
        for i in range(1, Nzp1):
            A_new[i] = 0.5 * (A[i] + A_star[i] - (dt/dx)*(Q_star[i] - Q_star[i-1]))
            Q_new[i] = 0.5 * (Q[i] + Q_star[i]
                              - (c**2 * dt/dx)*(A_star[i] - A_star[i-1])
                              - d*dt*Q_star[i])
        
        A[:] = A_new
        Q[:] = Q_new
        A[0] = A_inlet(t + dt)
        Q[0] = Q[1]
        A[-1] = A[-2]
        Q[-1] = Q[-2]
        
        if n % 1000 == 0:
            p = alpha * (A - A_ref)
            time_store.append(t)
            for j, idx in enumerate(probe_idx):
                p_store[j].append(p[idx])
                Q_store[j].append(Q[idx])
                A_store[j].append(A[idx])
    
    return np.array(time_store), np.array(p_store), np.array(Q_store), np.array(A_store), probe_labels

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        
        # Run simulation
        time_store, p_store, Q_store, A_store, probe_labels = run_simulation(data)
        
        # Create plot
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(probe_labels)))
        
        # Pressure plot
        for j, lbl in enumerate(probe_labels):
            axes[0].plot(time_store, p_store[j]/133.322, color=colors[j], label=f'z={lbl}')
        axes[0].set_ylabel('Pressure [mmHg]')
        axes[0].legend()
        axes[0].grid(True)
        
        # Flow plot
        for j, lbl in enumerate(probe_labels):
            axes[1].plot(time_store, Q_store[j]*1e6, color=colors[j], label=f'z={lbl}')
        axes[1].set_ylabel('Flow Q [mm³/s]')
        axes[1].legend()
        axes[1].grid(True)
        
        # Area plot
        for j, lbl in enumerate(probe_labels):
            axes[2].plot(time_store, A_store[j]*1e6, color=colors[j], label=f'z={lbl}')
        axes[2].set_ylabel('Area [mm²]')
        axes[2].set_xlabel('Time [s]')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        # Convert to base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'plot': f'data:image/png;base64,{plot_url}',
            'message': 'Simulation completed successfully!'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False)
