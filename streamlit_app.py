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
        "🧪 Test Simulations"
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
    
    with st.sidebar.expander("⚙️ Parameters"):
        L = st.slider("Artery Length (m)", 0.05, 1.0, 0.15, 0.01, key="t_length")
        D_ref = st.slider("Reference Diameter (mm)", 1.0, 20.0, 3.0, 0.5, key="t_diameter") / 1000.0
        E = st.slider("Young's Modulus (Pa)", 0.5e6, 5.0e6, 1.5e6, 0.1e6, key="t_modulus")
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
    st.warning("⚠️ This simulation is optimized for local execution with matplotlib interactive slider features.")
    st.info("For cloud-based deployment, use **Simulation T** instead.")
    st.markdown("""
    To run this locally, execute:
    ```bash
    python simulation_z.py
    ```
    This will show an interactive plot with a time slider to visualize spatial wave propagation.
    """)

# ============================================================
# HEALTHY ARTERY MODEL
# ============================================================
elif page == "💚 Healthy Artery Model":
    st.title("💚 Healthy Artery Model with Windkessel Outlet")
    st.warning("⚠️ This model is optimized for local execution with advanced Windkessel outlet conditions.")
    st.info("For cloud-based deployment, use **Simulation T** instead.")
    st.markdown("""
    To run this locally, execute:
    ```bash
    python healthy_wk/healthy_wk.py
    ```
    Features:
    - 3-element Windkessel outlet boundary conditions
    - Variable wall thickness
    - Realistic downstream compliance effects
    """)

# ============================================================
# TEST SIMULATIONS
# ============================================================
elif page == "🧪 Test Simulations":
    st.title("🧪 Test Simulations & Validation Cases")
    st.warning("⚠️ Test cases are optimized for local execution.")
    
    test_option = st.selectbox(
        "Select a test case:",
        ["Test 1-T", "Test 1-Z", "Dimensionless Model"]
    )
    
    if test_option == "Test 1-T":
        st.subheader("Test 1-T: Basic Validation Test")
        st.markdown("""
        A simple test case with smooth cosine bump initial conditions for validation.
        
        Run locally:
        ```bash
        python test_c.5/test1_t.py
        ```
        """)
    
    elif test_option == "Test 1-Z":
        st.subheader("Test 1-Z: Space-Based Test")
        st.markdown("""
        Space-focused test case for verification.
        
        Run locally:
        ```bash
        python test_c.5/test1_z.py
        ```
        """)
    
    elif test_option == "Dimensionless Model":
        st.subheader("Dimensionless Model Test")
        st.markdown("""
        Dimensionless formulation for theoretical studies.
        
        Run locally:
        ```bash
        python test_c.5/dimensionless/test-t_slider.py
        ```
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<p>Blood Flow Simulation Suite • Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)
