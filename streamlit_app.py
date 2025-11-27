import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Blood Flow Simulation Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.markdown("# 🏥 Navigation")
page = st.sidebar.radio(
    "Select a Simulation:",
    [
        "🏠 Home",
        "🫀 Simulation T (Time-based)",
        "🌊 Simulation Z (Space-based)",
        "💚 Healthy Artery Model",
        "🧪 Test 1-T (Cosine Bump)",
        "🧪 Test 1-Z (Spatial)",
        "🧪 Dimensionless Model"
    ]
)

# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Home":
    st.title("🫀 Blood Flow Simulation Suite")
    st.markdown("---")
    
    st.markdown("""
    Welcome to the **Blood Flow Simulation Suite**! This platform provides interactive simulations 
    of cardiovascular hemodynamics using advanced mathematical models of arterial blood flow.
    """)
    
    st.subheader("📚 Available Simulations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🫀 Simulation T (Time-based)**
        
        Models blood pressure and flow propagation through an artery over time using the 1D blood 
        flow equations with MacCormack numerical scheme.
        
        ✅ Available for cloud deployment
        """)
        
        st.info("""
        **🌊 Simulation Z (Space-based)**
        
        Interactive visualization showing spatial distribution of pressure, flow, and area 
        along the artery at different time snapshots.
        
        ⚠️ Local execution recommended
        """)
    
    with col2:
        st.info("""
        **💚 Healthy Artery Model**
        
        Advanced model including Windkessel outlet boundary conditions 
        for realistic downstream compliance effects.
        
        ⚠️ Local execution recommended
        """)
        
        st.info("""
        **🧪 Test Simulations**
        
        Various test cases for validation and verification purposes.
        
        ⚠️ Local execution recommended
        """)
    
    st.markdown("---")
    st.subheader("🚀 Getting Started")
    st.markdown("""
    1. Select a simulation from the navigation menu
    2. Adjust the arterial parameters using the sliders
    3. Click "Run Simulation" to start
    4. View results in interactive plots and summary statistics
    """)

# ============================================================
# SIMULATION T (TIME-BASED)
# ============================================================
elif page == "🫀 Simulation T (Time-based)":
    st.title("🫀 Simulation T: Time-Based Blood Flow Analysis")
    
    st.markdown("""
    This simulation models blood pressure and flow wave propagation through an artery using 
    the one-dimensional blood flow equations with the MacCormack numerical scheme.
    """)
    
    # Static parameters (fixed, no sliders)
    L = 0.15  # Artery length [m]
    D_ref = 3.0e-3  # Reference diameter [m]
    E = 1.5e6  # Young's Modulus [Pa]
    
    # Display parameter values
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Artery Length", f"{L*1000:.1f} mm")
    with col2:
        st.metric("Diameter", f"{D_ref*1000:.2f} mm")
    with col3:
        st.metric("Young's Modulus", f"{E/1e6:.1f} MPa")
    
    run_button = st.button("🚀 Run Simulation T", key="run_t", use_container_width=True)
    
    # Physical parameters
    rho = 1060.0
    mu = 3.5e-3
    r_ref = D_ref / 2.0
    h = 1.0e-4
    A_ref = math.pi * r_ref**2
    alpha = E * h / (2.0 * math.pi * r_ref**3)
    c = math.sqrt(alpha * A_ref / rho)
    
    # Numerical setup
    dx = 1.0e-3
    dt = 1.0e-5
    T_final = 1.0
    Nz = int(L / dx)
    Nzp1 = Nz + 1
    Nt = int(T_final / dt)
    z = np.linspace(0.0, L, Nzp1)
    CFL = c * dt / dx
    
    # Inlet pressure waveform
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
    
    if run_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        d = 8.0 * math.pi * mu / (rho * A_ref)
        
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
            
            if n % 10000 == 0:
                progress_bar.progress(min(n / Nt, 1.0))
                status_text.text(f"⏳ Processing... {n/Nt*100:.1f}%")
            
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
        
        progress_bar.progress(1.0)
        status_text.text("✅ Simulation complete!")
        
        time_store = np.array(time_store)
        p_store = np.array(p_store)
        Q_store = np.array(Q_store)
        A_store = np.array(A_store)
        
        st.success("Simulation completed successfully!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Wave Speed", f"{c:.3f} m/s")
        with col2:
            st.metric("CFL Number", f"{CFL:.3f}")
        with col3:
            st.metric("Grid Points", Nz)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(probe_idx)))
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        for j, lbl in enumerate(probe_labels):
            axes[0].plot(time_store, p_store[j]/133.322, color=colors[j], label=f'z={lbl}', linewidth=2)
        axes[0].set_ylabel('Pressure [mmHg]', fontsize=11, fontweight='bold')
        axes[0].legend(loc='best', ncol=3)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Arterial Pressure Wave Propagation', fontsize=12, fontweight='bold')
        
        for j, lbl in enumerate(probe_labels):
            axes[1].plot(time_store, Q_store[j]*1e6, color=colors[j], label=f'z={lbl}', linewidth=2)
        axes[1].set_ylabel('Flow Rate Q [mm³/s]', fontsize=11, fontweight='bold')
        axes[1].legend(loc='best', ncol=3)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Blood Flow Rate', fontsize=12, fontweight='bold')
        
        for j, lbl in enumerate(probe_labels):
            axes[2].plot(time_store, A_store[j]*1e6, color=colors[j], label=f'z={lbl}', linewidth=2)
        axes[2].set_ylabel('Cross-sectional Area [mm²]', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Time [s]', fontsize=11, fontweight='bold')
        axes[2].legend(loc='best', ncol=3)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Arterial Cross-sectional Area', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("📊 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Pressure (inlet)", f"{max(p_store[0])/133.322:.1f} mmHg")
            st.metric("Min Pressure (inlet)", f"{min(p_store[0])/133.322:.1f} mmHg")
        with col2:
            st.metric("Max Flow (inlet)", f"{max(Q_store[0])*1e6:.2f} mm³/s")
            st.metric("Min Flow (inlet)", f"{min(Q_store[0])*1e6:.2f} mm³/s")
        with col3:
            st.metric("Max Area (inlet)", f"{max(A_store[0])*1e6:.2f} mm²")
            st.metric("Min Area (inlet)", f"{min(A_store[0])*1e6:.2f} mm²")
    else:
        st.info("👈 **Adjust parameters in the sidebar and click 'Run Simulation T' to begin**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Wave Speed (preview)", f"{c:.3f} m/s")
        with col2:
            st.metric("CFL Number (preview)", f"{CFL:.3f}")
        with col3:
            st.metric("Grid Points", Nz)

# ============================================================
# SIMULATION Z (SPACE-BASED)
# ============================================================
elif page == "🌊 Simulation Z (Space-based)":
    st.title("🌊 Simulation Z: Space-Based Analysis")
    st.markdown("Shows spatial distribution of pressure, flow, and area along the artery at different time snapshots.")
    
    # Static parameters (fixed, no sliders)
    L = 0.15  # Artery length [m]
    D_ref = 3.0e-3  # Reference diameter [m]
    E = 1.5e6  # Young's Modulus [Pa]
    
    # Display parameter values
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Artery Length", f"{L*1000:.1f} mm")
    with col2:
        st.metric("Diameter", f"{D_ref*1000:.2f} mm")
    with col3:
        st.metric("Young's Modulus", f"{E/1e6:.1f} MPa")
    
    run_button = st.button("🚀 Run Simulation Z", key="run_z", use_container_width=True)
    
    rho = 1060.0
    mu = 3.5e-3
    r_ref = D_ref / 2.0
    h = 1.0e-4
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
    CFL = c * dt / dx
    
    def blackman_harris(x):
        if x < 0.0 or x > 1.0:
            return 0.0
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        return a0 - a1*np.cos(2*np.pi*x) + a2*np.cos(4*np.pi*x) - a3*np.cos(6*np.pi*x)

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
    
    if run_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        d = 8.0 * np.pi * mu / (rho * A_ref)
        
        A = np.full(Nzp1, A_ref)
        Q = np.zeros(Nzp1)
        
        snapshots_A, snapshots_Q, snapshots_P, times = [], [], [], []
        save_every = 2000
        
        for n in range(Nt):
            t = n * dt
            
            if n % 10000 == 0:
                progress_bar.progress(min(n / Nt, 1.0))
                status_text.text(f"⏳ Processing... {n/Nt*100:.1f}%")
            
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
            
            if n % save_every == 0:
                p = alpha * (A - A_ref)
                snapshots_A.append(A.copy())
                snapshots_Q.append(Q.copy())
                snapshots_P.append(p.copy())
                times.append(t)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Simulation complete!")
        
        snapshots_A = np.array(snapshots_A)
        snapshots_Q = np.array(snapshots_Q)
        snapshots_P = np.array(snapshots_P)
        times = np.array(times)
        
        st.success("Simulation completed successfully!")
        
        z_mm = z * 1000
        P_mmHg = snapshots_P / 133.322
        Q_mm3s = snapshots_Q * 1e6
        A_mm2 = snapshots_A * 1e6
        
        selected_time_idx = st.slider("Select time snapshot", 0, len(times)-1, 0, key="z_slider")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        axes[0].plot(z_mm, P_mmHg[selected_time_idx], 'r-', linewidth=2, marker='o', markersize=4)
        axes[0].set_ylabel('Pressure [mmHg]', fontweight='bold')
        axes[0].set_title(f'Spatial Distribution at t={times[selected_time_idx]:.4f}s', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(z_mm, Q_mm3s[selected_time_idx], 'b-', linewidth=2, marker='s', markersize=4)
        axes[1].set_ylabel('Flow [mm³/s]', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(z_mm, A_mm2[selected_time_idx], 'g-', linewidth=2, marker='^', markersize=4)
        axes[2].set_ylabel('Area [mm²]', fontweight='bold')
        axes[2].set_xlabel('Position along artery z [mm]', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("📊 Wave Propagation Over Time")
        
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        
        im0 = axes2[0].contourf(z_mm, times, P_mmHg, levels=15, cmap='RdYlBu_r')
        axes2[0].set_xlabel('Position z [mm]')
        axes2[0].set_ylabel('Time [s]')
        axes2[0].set_title('Pressure Wave Propagation')
        plt.colorbar(im0, ax=axes2[0], label='Pressure [mmHg]')
        
        im1 = axes2[1].contourf(z_mm, times, Q_mm3s, levels=15, cmap='viridis')
        axes2[1].set_xlabel('Position z [mm]')
        axes2[1].set_ylabel('Time [s]')
        axes2[1].set_title('Flow Wave Propagation')
        plt.colorbar(im1, ax=axes2[1], label='Flow [mm³/s]')
        
        im2 = axes2[2].contourf(z_mm, times, A_mm2, levels=15, cmap='plasma')
        axes2[2].set_xlabel('Position z [mm]')
        axes2[2].set_ylabel('Time [s]')
        axes2[2].set_title('Area Wave Propagation')
        plt.colorbar(im2, ax=axes2[2], label='Area [mm²]')
        
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.info("👈 **Adjust parameters and click 'Run Simulation Z' to begin**")

# ============================================================
# HEALTHY ARTERY MODEL
# ============================================================
elif page == "💚 Healthy Artery Model":
    st.title("💚 Healthy Artery Model")
    st.markdown("Advanced model with improved wall properties and compliance.")
    
    # Static parameters (fixed, no sliders)
    L = 0.15  # Artery length [m]
    D_ref = 3.0e-3  # Reference diameter [m]
    h_wall = 3.0e-4  # Wall thickness [m]
    E = 1.5e6  # Young's Modulus [Pa]
    
    # Display parameter values
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Artery Length", f"{L*1000:.1f} mm")
    with col2:
        st.metric("Diameter", f"{D_ref*1000:.2f} mm")
    with col3:
        st.metric("Wall Thickness", f"{h_wall*1e6:.1f} µm")
    with col4:
        st.metric("Young's Modulus", f"{E/1e6:.1f} MPa")
    
    run_button = st.button("🚀 Run Healthy Model", key="run_healthy", use_container_width=True)
    
    rho = 1060.0
    mu = 3.5e-3
    r_ref = D_ref / 2.0
    A_ref = np.pi * r_ref**2
    alpha = E * h_wall / (2.0 * np.pi * r_ref**3)
    c0 = np.sqrt(alpha * A_ref / rho)
    
    dx = 1.0e-3
    dt = 1.0e-5
    T_final = 1.0
    Nz = int(L / dx)
    Nzp1 = Nz + 1
    Nt = int(T_final / dt)
    z = np.linspace(0.0, L, Nzp1)
    delta = 8.0 * np.pi * mu / (rho * A_ref)
    
    def blackman_harris_h(x):
        if x < 0.0 or x > 1.0:
            return 0.0
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        return a0 - a1*np.cos(2*np.pi*x) + a2*np.cos(4*np.pi*x) - a3*np.cos(6*np.pi*x)

    def p_inlet_h(t):
        LD, LP, LT = 0.60, 0.55, 0.55
        tP, tD, tT = 0.38, 0.05, 0.20
        betaP, betaD, betaT = 1.0, 0.4, 0.3
        mmHg_to_Pa = 133.322
        A_total = 50.0 * mmHg_to_Pa
        
        xi_P = (t - tP) / LP
        xi_T = (t - tT) / LT
        xi_D = (t - tD) / LD
        wP = blackman_harris_h(xi_P)
        wT = blackman_harris_h(xi_T)
        wD = blackman_harris_h(xi_D)
        return A_total * (betaP*wP + betaT*wT + betaD*wD)

    def A_inlet_h(t):
        p = p_inlet_h(t)
        return A_ref + p / alpha
    
    if run_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        A = np.full(Nzp1, A_ref)
        Q = np.zeros(Nzp1)
        
        time_store = []
        p_store_h = []
        Q_store_h = []
        A_store_h = []
        
        for n in range(Nt):
            t = n * dt
            
            if n % 10000 == 0:
                progress_bar.progress(min(n / Nt, 1.0))
                status_text.text(f"⏳ Processing... {n/Nt*100:.1f}%")
            
            A[0] = A_inlet_h(t)
            Q[0] = Q[1]
            A[-1] = A[-2]
            Q[-1] = Q[-2]
            
            A_star = A.copy()
            Q_star = Q.copy()
            for i in range(Nz):
                A_star[i] = A[i] - (dt/dx) * (Q[i+1] - Q[i])
                Q_star[i] = Q[i] - (c0**2 * dt/dx) * (A[i+1] - A[i]) - delta * dt * Q[i]
            
            A_star[0] = A_inlet_h(t + dt)
            Q_star[0] = Q_star[1]
            A_star[-1] = A_star[-2]
            Q_star[-1] = Q_star[-2]
            
            A_new = A.copy()
            Q_new = Q.copy()
            for i in range(1, Nzp1):
                A_new[i] = 0.5 * (A[i] + A_star[i] - (dt/dx)*(Q_star[i] - Q_star[i-1]))
                Q_new[i] = 0.5 * (Q[i] + Q_star[i]
                                  - (c0**2 * dt/dx)*(A_star[i] - A_star[i-1])
                                  - delta*dt*Q_star[i])
            
            A[:] = A_new
            Q[:] = Q_new
            A[0] = A_inlet_h(t + dt)
            Q[0] = Q[1]
            A[-1] = A[-2]
            Q[-1] = Q[-2]
            
            if n % 1000 == 0:
                p = alpha * (A - A_ref)
                time_store.append(t)
                p_store_h.append(p.copy())
                Q_store_h.append(Q.copy())
                A_store_h.append(A.copy())
        
        progress_bar.progress(1.0)
        status_text.text("✅ Simulation complete!")
        
        st.success("Healthy model simulation completed!")
        
        time_store = np.array(time_store)
        p_store_h = np.array(p_store_h)
        Q_store_h = np.array(Q_store_h)
        A_store_h = np.array(A_store_h)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Wall Thickness", f"{h_wall*1000:.2f} mm")
        with col2:
            st.metric("Wave Speed", f"{c0:.3f} m/s")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        for i in [0, len(time_store)//2, -1]:
            axes[0].plot(z*1000, p_store_h[i]/133.322, label=f't={time_store[i]:.3f}s', linewidth=2, alpha=0.7)
        axes[0].set_ylabel('Pressure [mmHg]', fontweight='bold')
        axes[0].set_title('Pressure Evolution', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        for i in [0, len(time_store)//2, -1]:
            axes[1].plot(z*1000, Q_store_h[i]*1e6, label=f't={time_store[i]:.3f}s', linewidth=2, alpha=0.7)
        axes[1].set_ylabel('Flow [mm³/s]', fontweight='bold')
        axes[1].set_title('Flow Evolution', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        for i in [0, len(time_store)//2, -1]:
            axes[2].plot(z*1000, A_store_h[i]*1e6, label=f't={time_store[i]:.3f}s', linewidth=2, alpha=0.7)
        axes[2].set_ylabel('Area [mm²]', fontweight='bold')
        axes[2].set_xlabel('Position z [mm]', fontweight='bold')
        axes[2].set_title('Area Evolution', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("👈 **Adjust parameters and click 'Run Healthy Model' to begin**")

# ============================================================
# TEST 1-T: COSINE BUMP
# ============================================================
elif page == "🧪 Test 1-T (Cosine Bump)":
    st.title("🧪 Test 1-T: Cosine Bump Validation")
    st.markdown("A simple validation test with analytical benchmark conditions.")
    
    # Static parameters (fixed, no sliders)
    domain_length = 1.0  # Domain length [m]
    grid_points = 400  # Grid points
    final_time = 2.6  # Final time [s]
    
    # Display parameter values
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Domain Length", f"{domain_length:.2f} m")
    with col2:
        st.metric("Grid Points", grid_points)
    with col3:
        st.metric("Final Time", f"{final_time:.2f} s")
    
    run_test = st.button("🚀 Run Test 1-T", key="run_test_t", use_container_width=True)
    
    if run_test:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        L = domain_length
        Nx = grid_points
        dz = L / Nx
        z = np.linspace(0, L, Nx+1)
        
        Tfinal = final_time
        c = 1.0
        CFL = 0.4
        dt = CFL * dz / c
        Nt = int(Tfinal / dt) + 1
        dt = Tfinal / Nt
        
        epsilon = 0.02
        
        def bump(z_vals, center, eps, amp=1.0):
            s = (z_vals - center) / eps
            out = np.zeros_like(z_vals)
            mask = np.abs(s) < 1
            out[mask] = amp * 0.5 * (1 + np.cos(np.pi * s[mask]))
            return out
        
        Q = bump(z, 0.4, epsilon, amp=1.0)
        A = bump(z, 0.7, epsilon, amp=1.0)
        
        track_positions = [0.25, 0.50, 0.75]
        track_indices = [np.argmin(np.abs(z - xp)) for xp in track_positions]
        
        time_arr = []
        Q_track = {xp: [] for xp in track_positions}
        A_track = {xp: [] for xp in track_positions}
        
        t = 0.0
        for n in range(Nt + 1):
            if n % max(1, Nt//10) == 0:
                progress_bar.progress(min(n / (Nt+1), 1.0))
                status_text.text(f"⏳ Processing... {n/(Nt+1)*100:.1f}%")
            
            time_arr.append(t)
            for xp, idx in zip(track_positions, track_indices):
                Q_track[xp].append(Q[idx])
                A_track[xp].append(A[idx])
            
            Q[0], Q[-1] = Q[1], Q[-2]
            A[0], A[-1] = A[1], A[-2]
            
            Q_new = Q.copy()
            A_new = A.copy()
            for i in range(1, Nx):
                Q_new[i] = Q[i] - 0.5*CFL*(Q[i+1] - Q[i-1])
                A_new[i] = A[i] - 0.5*CFL*(A[i+1] - A[i-1])
            
            Q = Q_new
            A = A_new
            t += dt
        
        progress_bar.progress(1.0)
        status_text.text("✅ Test complete!")
        
        st.success("Test 1-T completed successfully!")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        time_arr = np.array(time_arr)
        for xp in track_positions:
            axes[0].plot(time_arr, Q_track[xp], label=f'z={xp}m', linewidth=2, alpha=0.7)
        axes[0].set_ylabel('Flow Q', fontweight='bold')
        axes[0].set_title('Flow Evolution at Tracking Points', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        for xp in track_positions:
            axes[1].plot(time_arr, A_track[xp], label=f'z={xp}m', linewidth=2, alpha=0.7)
        axes[1].set_ylabel('Area A', fontweight='bold')
        axes[1].set_xlabel('Time (s)', fontweight='bold')
        axes[1].set_title('Area Evolution at Tracking Points', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("📊 Test Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Grid Spacing (dz)", f"{dz:.6f} m")
            st.metric("Time Step (dt)", f"{dt:.8f} s")
        with col2:
            st.metric("CFL Number", f"{CFL:.3f}")
            st.metric("Total Time Steps", Nt)

# ============================================================
# TEST 1-Z: SPATIAL TEST
# ============================================================
elif page == "🧪 Test 1-Z (Spatial)":
    st.title("🧪 Test 1-Z: Spatial Distribution Test")
    st.markdown("Shows spatial evolution of test fields over multiple time snapshots.")
    
    # Static parameters (fixed, no sliders)
    domain_length = 1.0  # Domain length [m]
    
    # Display parameter values
    st.metric("Domain Length", f"{domain_length:.2f} m")
    
    run_test = st.button("🚀 Run Test 1-Z", key="run_test_z", use_container_width=True)
    
    if run_test:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        L = domain_length
        Nx = 200
        dz = L / Nx
        z = np.linspace(0, L, Nx+1)
        
        c = 1.0
        CFL = 0.4
        dt = CFL * dz / c
        Tfinal = 1.0
        Nt = int(Tfinal / dt)
        
        epsilon = 0.05
        
        def bump(z_vals, center, eps, amp=1.0):
            s = (z_vals - center) / eps
            out = np.zeros_like(z_vals)
            mask = np.abs(s) < 1
            out[mask] = amp * 0.5 * (1 + np.cos(np.pi * s[mask]))
            return out
        
        Q = bump(z, 0.4, epsilon, amp=1.0)
        A = bump(z, 0.7, epsilon, amp=1.0)
        
        snapshots_Q, snapshots_A, snap_times = [], [], []
        save_freq = max(1, Nt // 10)
        
        t = 0.0
        for n in range(Nt + 1):
            if n % max(1, Nt//10) == 0:
                progress_bar.progress(min(n / (Nt+1), 1.0))
                status_text.text(f"⏳ Processing... {n/(Nt+1)*100:.1f}%")
            
            if n % save_freq == 0:
                snapshots_Q.append(Q.copy())
                snapshots_A.append(A.copy())
                snap_times.append(t)
            
            Q[0], Q[-1] = Q[1], Q[-2]
            A[0], A[-1] = A[1], A[-2]
            
            Q_new = Q.copy()
            A_new = A.copy()
            for i in range(1, Nx):
                Q_new[i] = Q[i] - 0.5*CFL*(Q[i+1] - Q[i-1])
                A_new[i] = A[i] - 0.5*CFL*(A[i+1] - A[i-1])
            
            Q = Q_new
            A = A_new
            t += dt
        
        progress_bar.progress(1.0)
        status_text.text("✅ Test complete!")
        
        st.success("Test 1-Z completed!")
        
        snapshots_Q = np.array(snapshots_Q)
        snapshots_A = np.array(snapshots_A)
        snap_times = np.array(snap_times)
        
        # Display snapshots at saved times (like original code)
        st.subheader("📊 Snapshots at Different Times")
        
        for idx, t_snap in enumerate(snap_times):
            col1, col2 = st.columns(2)
            
            with col1:
                fig_q, ax_q = plt.subplots(figsize=(10, 4))
                ax_q.plot(z, snapshots_Q[idx], label="Flow Q(z,t)", linewidth=2, color='blue')
                ax_q.plot(z, snapshots_A[idx], label="Area A(z,t)", linewidth=2, color='orange')
                ax_q.set_xlabel('z (m)')
                ax_q.set_ylabel('Value')
                ax_q.set_title(f'Test problem at t = {t_snap:.2f}s')
                ax_q.grid(True, alpha=0.3)
                ax_q.legend()
                plt.tight_layout()
                st.pyplot(fig_q, use_container_width=True)
            
            with col2:
                st.metric(f"Time snapshot {idx}", f"{t_snap:.2f} s", delta=f"{t_snap:.4f}")
        
        # Summary stats
        st.subheader("📈 Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total snapshots", len(snap_times))
        with col2:
            st.metric("Final time", f"{snap_times[-1]:.2f} s")
        with col3:
            st.metric("Spatial points", len(z))

# ============================================================
# TEST DIMENSIONLESS MODEL
# ============================================================
elif page == "🧪 Dimensionless Model":
    st.title("🧪 Dimensionless Model: K1=0 Damping Test")
    st.markdown("Tests dimensionless formulation with MacCormack predictor-corrector scheme (original code).")
    
    @st.cache_data
    def run_dim_simulation():
        K3 = 0.0002
        N = 400
        dx = 1.0 / N
        x = (np.arange(N) + 0.5) * dx
        tau_final = 2.6
        CFL = 0.4
        dt = CFL * dx
        Nt = int(tau_final / dt)
        dt = tau_final / Nt
        a = np.zeros(N + 2)
        q = np.zeros(N + 2)
        def bump_dim(x_vals, center, width, amp):
            s = (x_vals - center) / width
            out = np.zeros_like(x_vals)
            mask = np.abs(s) < 1
            out[mask] = amp * 0.5 * (1 + np.cos(np.pi * s[mask]))
            return out
        eps = 0.020
        a[1:-1] = bump_dim(x, center=0.7, width=eps, amp=0.10)
        q[1:-1] = bump_dim(x, center=0.4, width=eps, amp=0.02)
        def apply_bc_dim(a, q):
            a[0] = a[1]
            q[0] = q[1]
            a[-1] = a[-2]
            q[-1] = q[-2]
        def mac_cormack_dim(a, q):
            a_p = a.copy()
            q_p = q.copy()
            for i in range(1, N + 1):
                a_p[i] = a[i] - (dt/dx)*(q[i+1] - q[i])
                q_p[i] = q[i] - (dt/dx)*(a[i+1] - a[i]) - dt*K3*q[i]
            apply_bc_dim(a_p, q_p)
            a_new = a.copy()
            q_new = q.copy()
            for i in range(1, N + 1):
                a_new[i] = 0.5*(a[i] + a_p[i] - (dt/dx)*(q_p[i] - q_p[i-1]))
                q_new[i] = 0.5*(q[i] + q_p[i]
                                - (dt/dx)*(a_p[i] - a_p[i-1])
                                - dt*K3*q_p[i])
            return a_new, q_new
        apply_bc_dim(a, q)
        a_hist = []
        q_hist = []
        t_hist = []
        save_every = 50
        tau = 0.0
        for n in range(Nt + 1):
            if n % save_every == 0:
                a_hist.append(a[1:-1].copy())
                q_hist.append(q[1:-1].copy())
                t_hist.append(tau)
            apply_bc_dim(a, q)
            a, q = mac_cormack_dim(a, q)
            tau += dt
        a_hist = np.array(a_hist)
        q_hist = np.array(q_hist)
        t_hist = np.array(t_hist)
        return a_hist, q_hist, t_hist, x

    a_hist, q_hist, t_hist, x = run_dim_simulation()

    st.subheader("🎚️ Interactive Time Slider")
    time_idx = st.slider("Time Index", 0, len(t_hist)-1, len(t_hist)//2)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, q_hist[time_idx], label="q(x, τ)", color='tab:blue')
    ax.plot(x, a_hist[time_idx], label="a(x, τ)", color='tab:orange')
    ax.set_xlabel("x (dimensionless)")
    ax.set_ylabel("Dimensionless Value")
    ax.set_title(f"Snapshot at τ = {t_hist[time_idx]:.4f}")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    st.subheader("📈 Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total snapshots", len(t_hist))
    with col2:
        st.metric("Final time", f"{t_hist[-1]:.2f}")
    with col3:
        st.metric("Spatial points", len(x))
