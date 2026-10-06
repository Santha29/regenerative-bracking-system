from simulation.vehicle_model import EVDigitalTwin
from battery.battery_model import BatteryModel


# Create models
ev = EVDigitalTwin()
battery = BatteryModel()


# Simulation inputs
velocity = 20.0          # m/s
acceleration = -2.0      # braking
k_reg = 0.70             # 70% regenerative braking request
dt = 0.01                # 100 Hz


# -----------------------------
# VEHICLE DIGITAL TWIN
# -----------------------------

total_torque = ev.total_braking_torque(
    velocity,
    acceleration
)

regen_torque = ev.regenerative_torque(
    total_torque,
    k_reg
)

mechanical_torque = ev.mechanical_torque(
    total_torque,
    k_reg
)

regen_power = ev.regenerative_power(
    regen_torque,
    velocity
)


# -----------------------------
# BATTERY CHARGING LIMIT
# -----------------------------

# Simplified maximum battery regenerative charging power
max_battery_charge_power = 80000.0   # 80 kW

battery_power = min(
    regen_power,
    max_battery_charge_power
)


# -----------------------------
# BATTERY UPDATE
# -----------------------------

battery_state = battery.step(
    battery_power,
    dt
)


# -----------------------------
# RESULTS
# -----------------------------

print("===== EV REGENERATIVE BRAKING SIMULATION =====")

print(f"Vehicle Speed        : {velocity:.2f} m/s")
print(f"Acceleration         : {acceleration:.2f} m/s²")

print()

print(f"Total Brake Torque   : {total_torque:.2f} Nm")
print(f"Requested Regen      : {k_reg * 100:.2f}%")
print(f"Regen Torque         : {regen_torque:.2f} Nm")
print(f"Mechanical Torque    : {mechanical_torque:.2f} Nm")

print()

print(f"Motor Regen Power    : {regen_power / 1000:.2f} kW")
print(f"Battery Charge Power : {battery_power / 1000:.2f} kW")

print()

print(f"Battery SOC          : {battery_state['soc'] * 100:.4f}%")
print(f"Battery SOH          : {battery_state['soh'] * 100:.4f}%")
print(f"Battery Temperature  : {battery_state['temperature']:.4f} °C")
print(f"Battery Current      : {battery_state['current']:.2f} A")