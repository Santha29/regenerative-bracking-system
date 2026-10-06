import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)
from xai.shap_explainer import SHAPExplainer
from dashboard.vehicle_simulator import EVBrakeSimulator
from dashboard.brake_pedal import brake_pedal

st.set_page_config(
    page_title="EV Regenerative Braking Simulator",
    page_icon="🚗",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD SIMULATOR
# ---------------------------------------------------------

@st.cache_resource
def load_simulator():
    return EVBrakeSimulator()


simulator = load_simulator()
@st.cache_resource
def load_shap_explainer():
    return SHAPExplainer()

shap_explainer = load_shap_explainer()

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "running" not in st.session_state:
    st.session_state.running = False

if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = simulator.step(
        brake_pedal=0.0,
        road_friction=0.80
    )


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🚗 EV AI Regenerative Braking Simulator")

st.markdown(
    """
    ### Real-Time Brake Pedal Simulation

    **Brake Pedal → Digital Twin → TD3 Controller → Safety Controller
    → Regenerative / Mechanical Braking → Battery**
    """
)
# ---------------------------------------------------------
# VEHICLE VISUALIZATION
# ---------------------------------------------------------

st.markdown("""
<style>
.ev-car {
    position: relative;
    width: 100%;
    height: 180px;
    background: linear-gradient(180deg, #1f2937, #111827);
    border-radius: 20px;
    overflow: hidden;
    margin: 10px 0 20px 0;
}

.car-body {
    position: absolute;
    left: 15%;
    top: 65px;
    width: 70%;
    height: 55px;
    background: #374151;
    border-radius: 25px 35px 15px 15px;
}

.car-window {
    position: absolute;
    left: 35%;
    top: 35px;
    width: 30%;
    height: 40px;
    background: #111827;
    border-radius: 20px 20px 5px 5px;
}

.wheel {
    position: absolute;
    bottom: 25px;
    width: 38px;
    height: 38px;
    background: #050505;
    border-radius: 50%;
    border: 5px solid #6b7280;
}

.wheel-left {
    left: 25%;
}

.wheel-right {
    right: 25%;
}

.flow {
    position: absolute;
    top: 125px;
    width: 100%;
    text-align: center;
    font-size: 16px;
    font-weight: 700;
}
</style>

<div class="ev-car">

    <div class="car-window"></div>

    <div class="car-body"></div>

    <div class="wheel wheel-left"></div>
    <div class="wheel wheel-right"></div>

    <div class="flow">
        🦶 BRAKE → 🤖 TD3 → ♻️ REGEN → 🔋 BATTERY
    </div>

</div>
""", unsafe_allow_html=True)

st.divider()
# ---------------------------------------------------------
# LIVE VEHICLE STATE
# ---------------------------------------------------------

if "result" in st.session_state:

    current_result = st.session_state.result

    current_brake = current_result.get(
        "brake_pedal", 0.0
    )

    if current_brake <= 0:
        st.info("🟢 VEHICLE STATE: CRUISING")
    elif current_brake < 30:
        st.warning(
            f"🟡 VEHICLE STATE: LIGHT BRAKING — "
            f"{current_brake:.0f}%"
        )
    elif current_brake < 70:
        st.warning(
            f"🟠 VEHICLE STATE: MEDIUM BRAKING — "
            f"{current_brake:.0f}%"
        )
    else:
        st.error(
            f"🔴 VEHICLE STATE: HARD BRAKING — "
            f"{current_brake:.0f}%"
        )

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("⚙️ Vehicle Settings")

initial_speed = st.sidebar.slider(
    "Initial Vehicle Speed (m/s)",
    min_value=5.0,
    max_value=40.0,
    value=20.0,
    step=0.5
)

soc = st.sidebar.slider(
    "Battery SOC (%)",
    min_value=10.0,
    max_value=95.0,
    value=60.0,
    step=1.0
)

soh = st.sidebar.slider(
    "Battery SOH (%)",
    min_value=50.0,
    max_value=100.0,
    value=100.0,
    step=1.0
)

temperature = st.sidebar.slider(
    "Battery Temperature (°C)",
    min_value=20.0,
    max_value=60.0,
    value=25.0,
    step=0.5
)

road_friction = st.sidebar.slider(
    "Road Friction (μ)",
    min_value=0.30,
    max_value=1.00,
    value=0.80,
    step=0.01
)


# ---------------------------------------------------------
# RESET
# ---------------------------------------------------------

if st.sidebar.button("🔄 Reset Simulation", use_container_width=True):

    simulator.reset(
        velocity=initial_speed,
        soc=soc,
        soh=soh,
        temperature=temperature
    )

    st.session_state.result = simulator.run_steps(
        brake_pedal=0.0,
        road_friction=road_friction
    )

    st.session_state.running = False
    st.session_state.history = []

    st.rerun()
@st.fragment(run_every="100ms")
def live_simulation():

    brake_pressure = brake_pedal(
        default=0.0,
        key="real_brake_pedal"
    )

    if brake_pressure is None:
        brake_pressure = 0.0

    brake_pressure = float(brake_pressure)

    result = simulator.run_steps(
        brake_pedal=brake_pressure,
        road_friction=road_friction,
        steps=10
    )

    st.session_state.result = result

    st.session_state.history.append({
        "Time": len(st.session_state.history) * 0.10,
        "Speed": result["speed"],
        "Brake": result["brake_pedal"],
        "k_reg": result["final_k_reg"],
        "Regen Power": result["regen_power"] / 1000.0,
        "SOC": result["soc"],
        "Temperature": result["temperature"],
        "Recovered Energy": result["recovered_energy"]
    })

    if brake_pressure == 0:
        st.success("🟢 BRAKE RELEASED")

    elif brake_pressure < 30:
        st.warning(
            f"🟡 LIGHT BRAKING — {brake_pressure:.0f}%"
        )

    elif brake_pressure < 70:
        st.warning(
            f"🟠 MEDIUM BRAKING — {brake_pressure:.0f}%"
        )

    else:
        st.error(
            f"🔴 HARD BRAKING — {brake_pressure:.0f}%"
        )
        # ---------------------------------------------------------
    # LIVE BRAKE CONTROL PANEL
    # ---------------------------------------------------------

    st.divider()
    st.subheader("🎛️ Live Brake Control")

    control_col1, control_col2, control_col3, control_col4 = st.columns(4)

    with control_col1:
        st.metric(
            "🦶 Brake Pressure",
            f"{brake_pressure:.1f}%"
        )

    with control_col2:
        st.metric(
            "🤖 TD3 k_reg",
            f"{result['final_k_reg']:.4f}"
        )

    with control_col3:
        st.metric(
            "♻️ Regen",
            f"{result['regen_percentage']:.1f}%"
        )
    st.caption(
        f"TD3 Requested: {result['final_k_reg'] * 100:.2f}% | "
        f"Actual Regen: {result['regen_percentage']:.2f}% | "
        f"Motor Limit: 310 Nm"
    )
    with control_col4:
        st.metric(
            "🛞 Mechanical",
            f"{result['mechanical_percentage']:.1f}%"
        )

    st.markdown("### 🦶 Brake Intensity")

    st.progress(
        min(max(int(brake_pressure), 0), 100),
        text=f"Brake Pressure — {brake_pressure:.1f}%"
    )

    if brake_pressure <= 0:
        st.info("🟢 CRUISING — Brake pedal released")

    elif result["safety_status"] != "NORMAL":
        st.warning(
            f"🛡️ SAFETY OVERRIDE — "
            f"{result['safety_status']}"
        )

    elif brake_pressure < 30:
        st.success(
            "🟡 LIGHT BRAKING — Regenerative braking active"
        )

    elif brake_pressure < 70:
        st.warning(
            "🟠 MEDIUM BRAKING — Torque split active"
        )

    else:
        st.error(
            "🔴 HARD BRAKING — Maximum braking demand"
        )

    st.subheader("📡 Live EV Simulation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🚗 Vehicle Speed",
            f"{result['speed']:.2f} m/s"
        )

        st.metric(
            "🦶 Brake Pressure",
            f"{result['brake_pedal']:.0f}%"
        )

        st.metric(
            "📉 Deceleration",
            f"{result['acceleration']:.2f} m/s²"
        )

    with control_col2:
        st.metric(
            "🤖 TD3 Requested k_reg",
            f"{result['requested_k_reg']:.4f}"
        )

        st.metric(
            "♻️ Regenerative",
            f"{result['regen_percentage']:.2f}%"
        )

        st.metric(
            "🛞 Mechanical",
            f"{result['mechanical_percentage']:.2f}%"
        )

    with col3:
        st.metric(
            "⚡ Regen Power",
            f"{result['regen_power'] / 1000:.2f} kW"
        )

        st.metric(
            "🛞 Mechanical Power",
            f"{result['mechanical_power'] / 1000:.2f} kW"
        )

        st.metric(
            "🔋 Battery Power",
            f"{result['battery_charge_power'] / 1000:.2f} kW"
        )

        st.metric(
            "♻️ Recovered Energy",
            f"{result['recovered_energy']:.6f} kWh"
        )

    st.subheader("🔋 Live Battery")

    b1, b2, b3, b4 = st.columns(4)

    b1.metric(
        "SOC",
        f"{result['soc']:.4f}%"
    )

    b2.metric(
        "SOH",
        f"{result['soh']:.4f}%"
    )

    b3.metric(
        "Temperature",
        f"{result['temperature']:.2f} °C"
    )

    b4.metric(
        "Current",
        f"{result['battery_current']:.2f} A"
    )
        # ---------------------------------------------------------
    # REAL-TIME BRAKING DISTRIBUTION
    # ---------------------------------------------------------

    st.divider()

    st.subheader("⚙️ Real-Time Braking Distribution")

    if brake_pressure > 0:
        regen = result["regen_percentage"]
        mechanical = result["mechanical_percentage"]
    else:
        regen = 0.0
        mechanical = 0.0

        st.markdown("### ⚡ Live Torque Distribution")

    split_col1, split_col2 = st.columns(2)

    with split_col1:
        st.metric(
            "♻️ Regenerative Braking",
            f"{regen:.2f}%"
        )

        st.progress(
            min(max(int(regen), 0), 100)
        )

    with split_col2:
        st.metric(
            "🛞 Mechanical Braking",
            f"{mechanical:.2f}%"
        )

        st.progress(
            min(max(int(mechanical), 0), 100)
        )

    st.caption(
        f"Total Torque: "
        f"{result['total_brake_torque']:.2f} Nm  |  "
        f"Regen Torque: "
        f"{result['regen_torque']:.2f} Nm  |  "
        f"Mechanical Torque: "
        f"{result['mechanical_torque']:.2f} Nm"
    )

    dist_col1, dist_col2, dist_col3 = st.columns(3)

    with dist_col1:
        st.metric(
            "♻️ Regenerative Torque",
            f"{result['regen_torque']:.2f} Nm"
        )

    with dist_col2:
        st.metric(
            "🛞 Mechanical Torque",
            f"{result['mechanical_torque']:.2f} Nm"
        )

    with dist_col3:
        st.metric(
            "⚡ Total Brake Torque",
            f"{result['total_brake_torque']:.2f} Nm"
        )

    # ---------------------------------------------------------
    # BATTERY GAUGES
    # ---------------------------------------------------------

    st.divider()

    st.subheader("🔋 Real-Time Battery Monitoring")

    gauge_col1, gauge_col2 = st.columns(2)

    with gauge_col1:
        st.markdown("### 🔋 Battery SOC")

        soc_value = result["soc"]

        st.progress(
            int(soc_value),
            text=f"SOC — {soc_value:.2f}%"
        )

        if soc_value >= 95:
            st.error("⚠️ SOC LIMIT — Regenerative braking restricted")
        elif soc_value >= 80:
            st.warning("🟡 High SOC")
        else:
            st.success("🟢 SOC Normal")

    with gauge_col2:
        st.markdown("### 🌡️ Battery Temperature")

        temperature_value = result["temperature"]

        temperature_percent = int(
            min(max((temperature_value - 20) / 25 * 100, 0), 100)
        )

        st.progress(
            temperature_percent,
            text=f"Temperature — {temperature_value:.2f} °C"
        )

        if temperature_value >= 45:
            st.error("🔴 TEMPERATURE LIMIT")
        elif temperature_value >= 35:
            st.warning("🟡 High Temperature")
        else:
            st.success("🟢 Temperature Normal")
       # ---------------------------------------------------------
    # SHAP EXPLAINABLE AI
    # ---------------------------------------------------------

    st.divider()

    st.subheader("🧠 Explainable AI — SHAP")

    shap_col1, shap_col2 = st.columns([1, 2])

    with shap_col1:

        st.markdown("### 🤖 TD3 Decision")

        st.metric(
            "Selected k_reg",
            f"{result['final_k_reg']:.4f}"
        )

        st.caption(
            "SHAP identifies which input features influenced "
            "the TD3 regenerative braking decision."
        )

        st.metric(
            "Safety Status",
            result["safety_status"]
        )

    with shap_col2:

        st.markdown("### 📊 Feature Contributions")

        shap_state = [
            result["speed"],
            result["braking_power"],
            result["soc"] / 100.0,
            result["soh"] / 100.0,
            result["temperature"],
            road_friction
        ]

        try:

            shap_values = shap_explainer.explain(
                shap_state
            )

            feature_names = [
                "Vehicle Speed",
                "Braking Power",
                "SOC",
                "SOH",
                "Battery Temperature",
                "Road Friction"
            ]

                        # Handle SHAP dictionary output
            if isinstance(shap_values, dict):

                feature_names = [
                    "Vehicle Speed",
                    "Braking Power",
                    "SOC",
                    "SOH",
                    "Battery Temperature",
                    "Road Friction"
                ]

                values = []

                for feature in feature_names:

                    value = shap_values.get(feature, 0.0)

                    if isinstance(value, (list, tuple)):
                        value = value[0]

                    if isinstance(value, np.ndarray):
                        value = value.reshape(-1)[0]

                    values.append(float(value))

                shap_values = np.array(
                    values,
                    dtype=float
                )

            else:

                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                shap_values = np.asarray(
                    shap_values,
                    dtype=float
                ).reshape(-1)

            if len(shap_values) != len(feature_names):
                raise ValueError(
                    f"Expected 6 SHAP values, "
                    f"got {len(shap_values)}"
                )

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": shap_values
            })

            shap_df["Absolute Impact"] = (
                shap_df["SHAP Value"].abs()
            )

            shap_df = shap_df.sort_values(
                "Absolute Impact",
                ascending=False
            )

            st.dataframe(
                shap_df[
                    [
                        "Feature",
                        "SHAP Value",
                        "Absolute Impact"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### 📊 SHAP Impact Chart")

            chart_df = shap_df[
                ["Feature", "SHAP Value"]
            ].set_index("Feature")

            st.bar_chart(
                chart_df
            )

            top_feature = shap_df.iloc[0]

            st.info(
                f"🔍 Top contributing feature: "
                f"**{top_feature['Feature']}** "
                f"(SHAP = "
                f"{top_feature['SHAP Value']:+.4f})"
            )

        except Exception as e:

            st.warning(
                f"SHAP calculation unavailable: {e}"
            )
        # ---------------------------------------------------------
    # SAFETY CONSTRAINT OVERRIDE
    # ---------------------------------------------------------

    st.divider()

    st.subheader("🛡️ Safety Constraint Override")

    safety_col1, safety_col2, safety_col3 = st.columns(3)

    with safety_col1:
        st.metric(
            "🤖 TD3 Requested k_reg",
            f"{result['requested_k_reg']:.4f}"
        )

    with safety_col2:
        st.metric(
            "🛡️ Final Safe k_reg",
            f"{result['final_k_reg']:.4f}"
        )

    with safety_col3:

        if result["safety_status"] == "NORMAL":
            st.success(
                "🟢 NORMAL\n\nSafety limits OK"
            )

        elif result["safety_status"] == "SOC_LIMIT":
            st.warning(
                "🟡 SOC LIMIT\n\n"
                "Regenerative braking restricted"
            )

        elif result["safety_status"] == "TEMPERATURE_LIMIT":
            st.error(
                "🔴 TEMPERATURE LIMIT\n\n"
                "Regenerative braking restricted"
            )

    # ---------------------------------------------------------
    # VEHICLE STATUS
    # ---------------------------------------------------------

    st.divider()

    st.subheader("🚗 Live Vehicle Status")

    vehicle_col1, vehicle_col2, vehicle_col3, vehicle_col4 = st.columns(4)

    with vehicle_col1:
        st.metric(
            "🚗 Vehicle Speed",
            f"{result['speed']:.2f} m/s"
        )

    with vehicle_col2:
        st.metric(
            "🦶 Brake Pressure",
            f"{result['brake_pedal']:.0f}%"
        )

    with vehicle_col3:
        st.metric(
            "📉 Deceleration",
            f"{result['acceleration']:.2f} m/s²"
        )

    with vehicle_col4:
        st.metric(
            "⚙️ Total Brake Torque",
            f"{result['total_brake_torque']:.2f} Nm"
        )
        # ---------------------------------------------------------
    # LIVE ENERGY FLOW
    # ---------------------------------------------------------

    st.divider()
    st.subheader("⚡ Live Energy Flow")

    if brake_pressure > 0:

        braking_power = result["braking_power"] / 1000
        regen_power = result["regen_power"] / 1000
        battery_power = result["battery_charge_power"] / 1000

        flow_col1, flow_col2, flow_col3 = st.columns(3)

        with flow_col1:
            st.metric(
                "🚗 Braking Power",
                f"{braking_power:.2f} kW"
            )

        with flow_col2:
            st.metric(
                "♻️ Regenerative Power",
                f"{regen_power:.2f} kW"
            )

        with flow_col3:
            st.metric(
                "🔋 Battery Charging",
                f"{battery_power:.2f} kW"
            )

        st.markdown("### 🔄 Energy Conversion Flow")

        st.markdown(
            """
            <div style="
                display:flex;
                justify-content:center;
                align-items:center;
                gap:20px;
                font-size:24px;
                padding:20px;
            ">
                <div>🚗<br><b>Vehicle</b></div>
                <div>➡️</div>
                <div>⚙️<br><b>Braking</b></div>
                <div>➡️</div>
                <div>♻️<br><b>Regeneration</b></div>
                <div>➡️</div>
                <div>🔋<br><b>Battery</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            f"♻️ {regen_power:.2f} kW regenerative power "
            f"is being recovered and sent to the battery."
        )

        st.info(
            f"🛞 Mechanical braking torque: "
            f"{result['mechanical_torque']:.2f} Nm"
        )

    else:
        st.info(
            "⏸️ No braking — energy recovery is inactive."
        )
        # ---------------------------------------------------------
    # LIVE SIMULATION GRAPHS
    # ---------------------------------------------------------

    st.divider()
    st.subheader("📈 Live Simulation Graphs")

    if len(st.session_state.history) > 0:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        # Row 1
        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            st.markdown("### 🚗 Vehicle Speed")

            st.line_chart(
                history_df.set_index("Time")["Speed"],
                height=250
            )

        with graph_col2:
            st.markdown("### 🦶 Brake Pressure")

            st.line_chart(
                history_df.set_index("Time")["Brake"],
                height=250
            )

        # Row 2
        graph_col3, graph_col4 = st.columns(2)

        with graph_col3:
            st.markdown("### 🤖 TD3 Regenerative Control")

            st.line_chart(
                history_df.set_index("Time")["k_reg"],
                height=250
            )

        with graph_col4:
            st.markdown("### ⚡ Regenerative Power")

            st.line_chart(
                history_df.set_index("Time")["Regen Power"],
                height=250
            )

        # Row 3
        graph_col5, graph_col6 = st.columns(2)

        with graph_col5:
            st.markdown("### 🔋 Battery SOC")

            st.line_chart(
                history_df.set_index("Time")["SOC"],
                height=250
            )

        with graph_col6:
            st.markdown("### 🌡️ Battery Temperature")

            st.line_chart(
                history_df.set_index("Time")["Temperature"],
                height=250
            )

        # Row 4
        st.markdown("### ♻️ Recovered Energy")

        st.line_chart(
            history_df.set_index("Time")["Recovered Energy"],
            height=250
        )
        # ---------------------------------------------------------
    # PERFORMANCE SUMMARY
    # ---------------------------------------------------------

    st.divider()
    st.subheader("📊 Performance Summary")

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.metric(
            "TD3 k_reg",
            f"{final_k_reg:.3f}"
        )

    with p2:
        st.metric(
            "Regen Power",
            f"{regen_power/1000:.2f} kW"
        )

    with p3:
        st.metric(
            "Battery SOC",
            f"{soc*100:.2f}%"
        )

    with p4:
        st.metric(
            "Recovered Energy",
            f"{recovered_energy:.4f} kWh"
        )
  
live_simulation()
# ---------------------------------------------------------
# SYSTEM PIPELINE
# ---------------------------------------------------------

st.divider()

st.subheader("🔄 AI Regenerative Braking Pipeline")

st.markdown(
    """
    **🦶 Brake Pedal**
    
    ↓
    
    **🚗 Vehicle Digital Twin**
    
    ↓
    
    **🤖 TD3 Controller**
    
    ↓
    
    **🛡️ Safety Controller**
    
    ↓
    
    **⚡ Torque Split**
    
    ↙️　　　　　　　　　↘️
    
    **♻️ Regenerative**　　 **🛞 Mechanical**
    
    ↓
    
    **🔋 Battery Model**
    
    ↓
    
    **SOC / SOH / Temperature / Current**
    """
)


