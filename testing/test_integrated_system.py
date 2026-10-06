import numpy as np

from simulation.vehicle_model import EVDigitalTwin
from battery.battery_model import BatteryModel
from controllers.integrated_controller import IntegratedController


def run_integrated_test():

    vehicle = EVDigitalTwin()
    battery = BatteryModel()
    controller = IntegratedController()

    velocity = 20.0
    acceleration = -2.0
    road_friction = 0.80

    dt = 0.01
    steps = 100

    total_energy = 0.0
    regen_energy = 0.0

    print("===== FULL EV REGENERATIVE BRAKING TEST =====")
    print()

    for step in range(steps):

        # Total braking torque
        total_torque = vehicle.total_braking_torque(
            velocity,
            acceleration
        )

        # Motor speed
        motor_speed = vehicle.motor_speed(velocity)

        # Braking power
        braking_power = total_torque * motor_speed

        braking_power = np.clip(
            braking_power,
            0.0,
            500000.0
        )

        # Current battery state
        state = np.array([
            velocity,
            braking_power,
            battery.soc,
            battery.soh,
            battery.temperature,
            road_friction
        ], dtype=np.float32)

        # TD3 + Safety Controller
        control = controller.get_control_action(state)

        requested_k_reg = control["requested_k_reg"]
        final_k_reg = control["final_k_reg"]

        # Torque split
        regen_torque = vehicle.regenerative_torque(
            total_torque,
            final_k_reg
        )

        mechanical_torque = vehicle.mechanical_torque(
            total_torque,
            final_k_reg
        )

        # Regenerative power
        regen_power = vehicle.regenerative_power(
            regen_torque,
            velocity
        )

        # Battery charging limit
        battery_power = min(
            regen_power,
            80000.0
        )

        # Battery update
        battery_state = battery.step(
            battery_power,
            dt
        )

        # Energy calculation
        total_energy += braking_power * dt
        regen_energy += battery_power * dt

        # Vehicle speed update
        velocity += acceleration * dt
        velocity = max(0.0, velocity)

    print("===== FINAL RESULTS =====")
    print()
    print(f"Final Vehicle Speed     : {velocity:.2f} m/s")
    print(f"Requested k_reg         : {requested_k_reg:.4f}")
    print(f"Final k_reg             : {final_k_reg:.4f}")
    print(f"Safety Status           : {control['safety_status']}")
    print()
    print(f"Regenerative Torque     : {regen_torque:.2f} Nm")
    print(f"Mechanical Torque       : {mechanical_torque:.2f} Nm")
    print(f"Regenerative Power      : {regen_power / 1000:.2f} kW")
    print()
    print(f"Battery SOC             : {battery.soc * 100:.4f}%")
    print(f"Battery SOH             : {battery.soh * 100:.4f}%")
    print(f"Battery Temperature     : {battery.temperature:.4f} °C")
    print(f"Battery Current         : {battery_state['current']:.2f} A")
    print()
    print(
        f"Recovered Energy        : "
        f"{regen_energy / 3600000:.6f} kWh"
    )
    print(
        f"Braking Energy          : "
        f"{total_energy / 3600000:.6f} kWh"
    )
    print()
    print("===== INTEGRATED TEST COMPLETE =====")


if __name__ == "__main__":
    run_integrated_test()