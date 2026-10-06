import numpy as np


class SafetyController:

    def __init__(self):
        # Safety limits
        self.max_soc = 0.95
        self.max_temperature = 45.0

        # Safe fallback regenerative braking
        self.safe_k_reg = 0.10

    def apply_safety(self, requested_k_reg, soc, temperature):
        """
        Apply safety constraint override.

        If SOC or temperature exceeds the limit,
        regenerative braking is reduced to safe_k_reg.
        """

        requested_k_reg = np.clip(
            requested_k_reg,
            0.0,
            1.0
        )

        # SOC safety constraint
        if soc >= self.max_soc:
            return self.safe_k_reg, "SOC_LIMIT"

        # Temperature safety constraint
        if temperature >= self.max_temperature:
            return self.safe_k_reg, "TEMPERATURE_LIMIT"

        # Normal operation
        return requested_k_reg, "NORMAL"


if __name__ == "__main__":

    safety = SafetyController()

    # Test 1: Normal condition
    k_reg, status = safety.apply_safety(
        requested_k_reg=0.70,
        soc=0.60,
        temperature=25.0
    )

    print("===== SAFETY CONTROLLER TEST =====")
    print(f"Requested k_reg : 0.70")
    print(f"Actual k_reg    : {k_reg:.2f}")
    print(f"Status          : {status}")

    print()

    # Test 2: High SOC
    k_reg, status = safety.apply_safety(
        requested_k_reg=0.70,
        soc=0.96,
        temperature=25.0
    )

    print("High SOC Test")
    print(f"Actual k_reg    : {k_reg:.2f}")
    print(f"Status          : {status}")

    print()

    # Test 3: High temperature
    k_reg, status = safety.apply_safety(
        requested_k_reg=0.70,
        soc=0.60,
        temperature=46.0
    )

    print("High Temperature Test")
    print(f"Actual k_reg    : {k_reg:.2f}")
    print(f"Status          : {status}")