import numpy as np


class EVDigitalTwin:
    def __init__(self):
        # Vehicle parameters
        self.mass = 1650.0          # kg
        self.tire_radius = 0.32     # m
        self.cd = 0.28              # aerodynamic drag coefficient
        self.frontal_area = 2.22    # m^2

        # Environment
        self.air_density = 1.225    # kg/m^3
        self.gravity = 9.81         # m/s^2
        self.rolling_coeff = 0.012

        # Drivetrain
        self.gear_ratio = 8.55
        self.motor_max_torque = 310.0  # Nm

    def resistance_force(self, velocity, road_angle=0.0):
        """
        Calculate rolling, aerodynamic and grade resistance.
        velocity: m/s
        road_angle: radians
        """

        rolling = (
            self.rolling_coeff
            * self.mass
            * self.gravity
            * np.cos(road_angle)
        )

        aerodynamic = (
            0.5
            * self.air_density
            * self.cd
            * self.frontal_area
            * velocity ** 2
        )

        grade = (
            self.mass
            * self.gravity
            * np.sin(road_angle)
        )

        total_resistance = rolling + aerodynamic + grade

        return total_resistance

    def braking_force(self, acceleration):
        """
        Calculate braking force magnitude.
        Negative acceleration represents braking.
        """

        return self.mass * abs(acceleration)

    def total_braking_torque(
        self,
        velocity,
        acceleration,
        road_angle=0.0
    ):
        """
        Calculate total wheel braking torque.
        """

        f_total = self.braking_force(acceleration)

        f_res = self.resistance_force(
            velocity,
            road_angle
        )

        braking_force = f_total + f_res

        torque = (
            braking_force
            * self.tire_radius
        )

        return max(0.0, torque)

    def motor_speed(self, velocity):
        """
        Calculate motor angular speed.
        """

        wheel_angular_speed = (
            velocity / self.tire_radius
        )

        motor_angular_speed = (
            wheel_angular_speed
            * self.gear_ratio
        )

        return motor_angular_speed

    def regenerative_torque(
        self,
        total_torque,
        k_reg
    ):
        """
        Calculate regenerative braking torque.
        k_reg is regeneration torque split ratio [0, 1].
        """

        k_reg = np.clip(k_reg, 0.0, 1.0)

        regenerative = k_reg * total_torque

        regenerative = min(
            regenerative,
            self.motor_max_torque
        )

        return regenerative

    def mechanical_torque(
        self,
        total_torque,
        k_reg
    ):
        """
        Calculate mechanical braking torque.
        """

        k_reg = np.clip(k_reg, 0.0, 1.0)

        regenerative = self.regenerative_torque(
            total_torque,
            k_reg
        )

        mechanical = max(
            0.0,
            total_torque - regenerative
        )

        return mechanical

    def regenerative_power(
        self,
        regenerative_torque,
        velocity
    ):
        """
        Calculate recovered electrical power.
        """

        motor_speed = self.motor_speed(velocity)

        # Simplified generator efficiency
        efficiency = 0.90

        power = (
            regenerative_torque
            * motor_speed
            * efficiency
        )

        return max(0.0, power)


if __name__ == "__main__":

    ev = EVDigitalTwin()

    velocity = 20.0       # m/s
    acceleration = -2.0   # m/s^2
    k_reg = 0.70

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

    print("===== EV DIGITAL TWIN TEST =====")
    print(f"Vehicle Speed       : {velocity:.2f} m/s")
    print(f"Acceleration        : {acceleration:.2f} m/s²")
    print(f"Total Brake Torque  : {total_torque:.2f} Nm")
    print(f"Regen Torque        : {regen_torque:.2f} Nm")
    print(f"Mechanical Torque   : {mechanical_torque:.2f} Nm")
    print(f"Recovered Power     : {regen_power / 1000:.2f} kW")