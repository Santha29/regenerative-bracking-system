import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulation.vehicle_model import EVDigitalTwin
from battery.battery_model import BatteryModel


class BrakingEnvironment(gym.Env):

    def __init__(self):

        super().__init__()

        self.vehicle = EVDigitalTwin()
        self.dt = 0.01

        # State:
        # [velocity, braking_power, SOC, SOH, temperature, road_friction]

        self.observation_space = spaces.Box(
            low=np.array([
                0.0,
                0.0,
                0.10,
                0.0,
                20.0,
                0.30
            ], dtype=np.float32),

            high=np.array([
                40.0,
                500000.0,
                0.95,
                1.0,
                60.0,
                1.0
            ], dtype=np.float32)
        )

        # Action:
        # regenerative braking coefficient

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32)
        )

        self.state = None
        self.step_count = 0
        self.max_steps = 500

        self.reset()

    # =====================================================
    # RESET
    # =====================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.battery = BatteryModel()

        self.step_count = 0

        # Random initial driving condition
        velocity = self.np_random.uniform(
            10.0,
            30.0
        )

        # Random braking intensity
        acceleration = -self.np_random.uniform(
            1.0,
            3.5
        )

        # Random road friction
        road_friction = self.np_random.uniform(
            0.40,
            0.95
        )

        total_torque = self.vehicle.total_braking_torque(
            velocity,
            acceleration
        )

        motor_speed = self.vehicle.motor_speed(
            velocity
        )

        braking_power = (
            total_torque
            * motor_speed
        )

        # Keep braking power inside observation range
        braking_power = np.clip(
            braking_power,
            0.0,
            500000.0
        )

        self.state = np.array([
            velocity,
            braking_power,
            self.battery.soc,
            self.battery.soh,
            self.battery.temperature,
            road_friction
        ], dtype=np.float32)

        return self.state, {}

    # =====================================================
    # STEP
    # =====================================================

    def step(self, action):

        self.step_count += 1

        # -------------------------------------------------
        # Action
        # -------------------------------------------------

        k_reg = float(
            np.clip(
                action[0],
                0.0,
                1.0
            )
        )

        # -------------------------------------------------
        # Current state
        # -------------------------------------------------

        velocity = float(self.state[0])
        road_friction = float(self.state[5])

        # -------------------------------------------------
        # Variable braking condition
        # -------------------------------------------------

        acceleration = -self.np_random.uniform(
            1.0,
            3.5
        )

        # -------------------------------------------------
        # Total braking torque
        # -------------------------------------------------

        total_torque = self.vehicle.total_braking_torque(
            velocity,
            acceleration
        )

        # -------------------------------------------------
        # Regenerative torque
        # -------------------------------------------------

        regen_torque = self.vehicle.regenerative_torque(
            total_torque,
            k_reg
        )

        # -------------------------------------------------
        # Mechanical torque
        # -------------------------------------------------

        mechanical_torque = self.vehicle.mechanical_torque(
            total_torque,
            k_reg
        )

        # -------------------------------------------------
        # Regenerative power
        # -------------------------------------------------

        regen_power = self.vehicle.regenerative_power(
            regen_torque,
            velocity
        )

        # Battery charging limit
        battery_power = min(
            regen_power,
            80000.0
        )

        # -------------------------------------------------
        # Battery update
        # -------------------------------------------------

        battery_state = self.battery.step(
            battery_power,
            self.dt
        )

        soc = battery_state["soc"]
        soh = battery_state["soh"]
        temperature = battery_state["temperature"]

        # -------------------------------------------------
        # REWARD
        # -------------------------------------------------

               	#=================================================
        # ADAPTIVE REWARD FUNCTION
        # =================================================

        # 1. Reward recovered regenerative energy
        energy_reward = battery_power / 80000.0

        # 2. Penalize high SOC
        soc_penalty = 0.0

        if soc > 0.80:
            soc_penalty = (
                (soc - 0.80) / 0.15
            ) * k_reg * 2.0

        # 3. Penalize high battery temperature
        temperature_penalty = 0.0

        if temperature > 30.0:
            temperature_penalty = (
                (temperature - 30.0) / 15.0
            ) * k_reg * 2.0

        # 4. Low road-friction penalty
        friction_penalty = 0.0

        if road_friction < 0.50 and k_reg > 0.70:
            friction_penalty = (
                (k_reg - 0.70) / 0.30
            ) * 2.0

        # 5. Penalize excessively aggressive regeneration
        aggressive_penalty = 0.0

        if k_reg > 0.85:
            aggressive_penalty = (
                k_reg - 0.85
            ) * 2.0

        # Final adaptive reward
        reward = (
            2.0 * energy_reward
            - soc_penalty
            - temperature_penalty
            - friction_penalty
            - aggressive_penalty
        )
	        	# -------------------------------------------------
        # Update velocity
        # -------------------------------------------------

        velocity = velocity + acceleration * self.dt

        velocity = max(
            0.0,
            velocity
        )

        # -------------------------------------------------
        # New braking power
        # -------------------------------------------------

        next_total_torque = self.vehicle.total_braking_torque(
            velocity,
            acceleration
        )

        next_motor_speed = self.vehicle.motor_speed(
            velocity
        )

        next_braking_power = (
            next_total_torque
            * next_motor_speed
        )

        next_braking_power = np.clip(
            next_braking_power,
            0.0,
            500000.0
        )
        # -------------------------------------------------
        # Next state
        # -------------------------------------------------

        self.state = np.array([
            velocity,
            next_braking_power,
            soc,
            soh,
            temperature,
            road_friction
        ], dtype=np.float32)

        # -------------------------------------------------
        # Termination
        # -------------------------------------------------

        terminated = (
            velocity <= 0.0
            or soc >= 0.95
            or temperature >= 45.0
        )

        truncated = (
            self.step_count >= self.max_steps
        )

        info = {
            "k_reg": k_reg,
            "acceleration": acceleration,
            "regen_torque": regen_torque,
            "mechanical_torque": mechanical_torque,
            "regen_power": regen_power,
            "battery_power": battery_power,
            "soc": soc,
            "soh": soh,
            "temperature": temperature,
            "road_friction": road_friction
        }

        return (
            self.state,
            float(reward),
            terminated,
            truncated,
            info
        )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    env = BrakingEnvironment()

    print("===== BRAKING ENVIRONMENT TEST =====")

    for test in range(3):

        state, _ = env.reset()

        print()
        print(f"Test Episode {test + 1}")

        print(
            f"Velocity      : {state[0]:.2f} m/s"
        )

        print(
            f"Braking Power : {state[1]:.2f} W"
        )

        print(
            f"SOC           : {state[2]:.4f}"
        )

        print(
            f"SOH           : {state[3]:.4f}"
        )

        print(
            f"Temperature   : {state[4]:.2f} °C"
        )

        print(
            f"Road Friction : {state[5]:.2f}"
        )

        # Test action
        action = np.array(
            [0.70],
            dtype=np.float32
        )

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        print(
            f"k_reg         : {info['k_reg']:.2f}"
        )

        print(
            f"Acceleration   : {info['acceleration']:.2f} m/s²"
        )

        print(
            f"Regen Power    : {info['regen_power'] / 1000:.2f} kW"
        )

        print(
            f"Reward         : {reward:.4f}"
        )

