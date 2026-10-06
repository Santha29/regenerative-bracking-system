import numpy as np


class BatteryModel:

    def __init__(self):
        # Battery parameters
        self.nominal_voltage = 380.0
        self.capacity_kwh = 60.0
        self.capacity_ah = (self.capacity_kwh * 1000) / self.nominal_voltage

        # Initial conditions
        self.soc = 0.60
        self.soh = 1.00
        self.temperature = 25.0

        # Limits
        self.min_soc = 0.10
        self.max_soc = 0.95
        self.min_temperature = 20.0
        self.max_temperature = 45.0

        # Battery resistance
        self.internal_resistance = 0.05

    def update_soc(self, power_w, dt):
        """
        Update battery SOC.

        Positive power_w = charging
        Negative power_w = discharging
        """

        energy_kwh = (power_w * dt) / 3600000.0

        delta_soc = energy_kwh / self.capacity_kwh

        self.soc += delta_soc

        self.soc = np.clip(
            self.soc,
            self.min_soc,
            self.max_soc
        )

        return self.soc

    def update_temperature(self, current_a, dt):
        # Battery heating due to charging current
        heat_power = (current_a ** 2) * self.internal_resistance

        # Stronger thermal response for simulation
        temperature_rise = heat_power * dt * 0.0005

        # Natural cooling
        cooling = (self.temperature - 25.0) * 0.005 * dt

        self.temperature += temperature_rise - cooling

        # Keep temperature physically reasonable
        self.temperature = np.clip(
            self.temperature,
            20.0,
            60.0
        )

        return self.temperature

    def update_soh(self, current_a, dt):
        """
        Simplified battery degradation model.
        """

        degradation_rate = (
            1e-8
            * abs(current_a)
            * np.exp(
                0.02 * (self.temperature - 25.0)
            )
        )

        self.soh -= degradation_rate * dt

        self.soh = np.clip(
            self.soh,
            0.0,
            1.0
        )

        return self.soh

    def charging_current(self, charging_power_w):
        """
        Calculate approximate battery charging current.
        """

        if charging_power_w <= 0:
            return 0.0

        return charging_power_w / self.nominal_voltage

    def step(self, charging_power_w, dt):
        """
        Perform one battery simulation step.
        """

        current = self.charging_current(charging_power_w)

        soc = self.update_soc(
            charging_power_w,
            dt
        )

        temperature = self.update_temperature(
            current,
            dt
        )

        soh = self.update_soh(
            current,
            dt
        )

        return {
            "soc": soc,
            "soh": soh,
            "temperature": temperature,
            "current": current
        }


if __name__ == "__main__":

    battery = BatteryModel()

    charging_power = 50000.0   # 50 kW
    dt = 0.01                  # 100 Hz = 0.01 s

    result = battery.step(
        charging_power,
        dt
    )

    print("===== BATTERY MODEL TEST =====")
    print(f"SOC         : {result['soc'] * 100:.4f}%")
    print(f"SOH         : {result['soh'] * 100:.4f}%")
    print(f"Temperature : {result['temperature']:.4f} °C")
    print(f"Current     : {result['current']:.2f} A")