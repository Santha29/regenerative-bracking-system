import os
import sys
import numpy as np

# Project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from simulation.vehicle_model import EVDigitalTwin
from battery.battery_model import BatteryModel
from controllers.integrated_controller import IntegratedController


class EVBrakeSimulator:

    def __init__(self):

        self.vehicle = EVDigitalTwin()
        self.battery = BatteryModel()
        self.controller = IntegratedController()

        self.dt = 0.01

        self.velocity = 20.0
        
        # Total recovered energy during the simulation
        self.total_recovered_energy = 0.0
    def reset(
        self,
        velocity=20.0,
        soc=60.0,
        soh=100.0,
        temperature=25.0
    ):

        self.velocity = float(velocity)

        self.total_recovered_energy = 0.0

        self.battery.soc = np.clip(
            soc / 100.0,
            0.10,
            0.95
        )

        self.battery.soh = np.clip(
            soh / 100.0,
            0.0,
            1.0
        )

        self.battery.temperature = float(
            temperature
        )

    def step(
        self,
        brake_pedal,
        road_friction=0.80
    ):

        brake_pedal = np.clip(
            brake_pedal,
            0.0,
            100.0
        )

        # --------------------------------------------------
        # Brake pedal → automatic braking acceleration
        # --------------------------------------------------

        if brake_pedal <= 0.0:

            acceleration = 0.0

        else:

            acceleration = -(
                1.0
                + 5.0 * brake_pedal / 100.0
            )

        # --------------------------------------------------
        # Digital Twin
        # --------------------------------------------------

        total_torque = (
            self.vehicle.total_braking_torque(
                self.velocity,
                acceleration
            )
            if brake_pedal > 0
            else 0.0
        )

        motor_speed = self.vehicle.motor_speed(
            self.velocity
        )

        braking_power = (
            total_torque * motor_speed
        )

        braking_power = np.clip(
            braking_power,
            0.0,
            500000.0
        )

        # --------------------------------------------------
        # State for TD3
        # --------------------------------------------------

        state = np.array([
            self.velocity,
            braking_power,
            self.battery.soc,
            self.battery.soh,
            self.battery.temperature,
            road_friction
        ], dtype=np.float32)

        # --------------------------------------------------
        # TD3 + Safety Controller
        # --------------------------------------------------

        control = self.controller.get_control_action(
            state
        )

        requested_k_reg = control[
            "requested_k_reg"
        ]

        final_k_reg = control[
            "final_k_reg"
        ]

        safety_status = control[
            "safety_status"
        ]

        # --------------------------------------------------
        # Braking split
        # --------------------------------------------------

        if brake_pedal > 0 and self.velocity > 0:

            regen_torque = self.vehicle.regenerative_torque(
                total_torque,
                final_k_reg
            )

            mechanical_torque = self.vehicle.mechanical_torque(
                total_torque,
                final_k_reg
            )

            regen_power = self.vehicle.regenerative_power(
                regen_torque,
                self.velocity
            )

            mechanical_power = (
                mechanical_torque
                * self.vehicle.motor_speed(self.velocity)
            )

        else:

            regen_torque = 0.0
            mechanical_torque = 0.0
            
            regen_power = 0.0
            mechanical_power = 0.0
        # --------------------------------------------------
        # Battery charging
        # --------------------------------------------------

        battery_charge_power = min(
            regen_power,
            80000.0
        )

        battery_state = self.battery.step(
            battery_charge_power,
            self.dt
        )

        # --------------------------------------------------
        # Recovered energy
        # --------------------------------------------------

        recovered_energy = (
            battery_charge_power
            * self.dt
            / 3600000.0
        )

        # --------------------------------------------------
        # Vehicle speed update
        # --------------------------------------------------

        self.velocity += (
            acceleration * self.dt
        )

        self.velocity = max(
            0.0,
            self.velocity
        )

        # --------------------------------------------------
        # Brake percentages
        # --------------------------------------------------

        if total_torque > 0:
            regen_percentage = (
                regen_torque / total_torque * 100.0
            )

            mechanical_percentage = (
                mechanical_torque / total_torque * 100.0
            )

            # Make sure the displayed percentages represent
            # the actual torque split.
            total_percentage = (
                regen_percentage + mechanical_percentage
            )

            if total_percentage > 0:
                regen_percentage = (
                    regen_percentage / total_percentage * 100.0
                )

                mechanical_percentage = (
                    mechanical_percentage / total_percentage * 100.0
                )

        else:
            regen_percentage = 0.0
            mechanical_percentage = 0.0

        return {    

            "speed": self.velocity,

            "brake_pedal": brake_pedal,

            "braking_power": braking_power,

            "acceleration": acceleration,

            "total_brake_torque":
                total_torque,

            "requested_k_reg":
                requested_k_reg,

            "final_k_reg":
                final_k_reg,

            "regen_percentage":
                regen_percentage,

            "mechanical_percentage":
                mechanical_percentage,

            "regen_torque":
                regen_torque,

            "mechanical_torque":
                mechanical_torque,

            "regen_power":
                regen_power,

            "mechanical_power":
                mechanical_power,

            "battery_charge_power":
                battery_charge_power,

            "recovered_energy":
                recovered_energy,

            "soc":
                battery_state["soc"] * 100.0,

            "soh":
                battery_state["soh"] * 100.0,

            "temperature":
                battery_state["temperature"],

            "battery_current":
                battery_state["current"],

            "safety_status":
                safety_status
        }
    # --------------------------------------------------
    # Run multiple simulation steps
    # --------------------------------------------------

    def run_steps(
        self,
        brake_pedal,
        road_friction=0.80,
        steps=10
    ):
        """
        Run multiple 100 Hz simulation steps.

        Brake pedal controls braking demand.
        Recovered energy is accumulated across
        the complete simulation.
        """

        result = None

        for _ in range(steps):

            result = self.step(
                brake_pedal=brake_pedal,
                road_friction=road_friction
            )

            self.total_recovered_energy += (
                result["recovered_energy"]
            )

            if self.velocity <= 0.0:
                break

        if result is not None:

            result["recovered_energy"] = (
                self.total_recovered_energy
            )

        return result

if __name__ == "__main__":

    print(
        "===== EV BRAKE PEDAL SIMULATOR TEST ====="
    )

    simulator = EVBrakeSimulator()

    simulator.reset(
        velocity=20.0,
        soc=60.0,
        soh=100.0,
        temperature=25.0
    )

    print()
    print("Before Brake")
    print(
        f"Speed : {simulator.velocity:.2f} m/s"
    )

    result = simulator.run_steps(
        brake_pedal=70.0,
        road_friction=0.80,
        steps=100
    )

    print()
    print("After Brake")
    print(
        f"Brake Pedal          : "
        f"{result['brake_pedal']:.1f}%"
    )

    print(
        f"Speed                : "
        f"{result['speed']:.2f} m/s"
    )

    print(
        f"Total Brake Torque   : "
        f"{result['total_brake_torque']:.2f} Nm"
    )

    print(
        f"TD3 k_reg            : "
        f"{result['final_k_reg']:.4f}"
    )

    print(
        f"Regenerative Brake   : "
        f"{result['regen_percentage']:.2f}%"
    )

    print(
        f"Mechanical Brake     : "
        f"{result['mechanical_percentage']:.2f}%"
    )

    print(
        f"Regen Torque         : "
        f"{result['regen_torque']:.2f} Nm"
    )

    print(
        f"Mechanical Torque    : "
        f"{result['mechanical_torque']:.2f} Nm"
    )

    print(
        f"Regen Power          : "
        f"{result['regen_power'] / 1000:.2f} kW"
    )

    print(
        f"Battery Charge Power : "
        f"{result['battery_charge_power'] / 1000:.2f} kW"
    )

    print(
        f"Recovered Energy     : "
        f"{result['recovered_energy']:.6f} kWh"
    )

    print(
        f"Battery SOC          : "
        f"{result['soc']:.4f}%"
    )

    print(
        f"Battery SOH          : "
        f"{result['soh']:.4f}%"
    )

    print(
        f"Battery Temperature  : "
        f"{result['temperature']:.4f} °C"
    )

    print(
        f"Battery Current      : "
        f"{result['battery_current']:.2f} A"
    )

    print(
        f"Safety Status        : "
        f"{result['safety_status']}"
    )

    print()
    print("===== TEST COMPLETE =====")