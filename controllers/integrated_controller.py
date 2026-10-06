import os
import numpy as np
import torch

from controllers.td3_agent import TD3Agent
from controllers.safety_controller import SafetyController


class IntegratedController:
    def __init__(self):
        self.td3_agent = TD3Agent()
        self.safety_controller = SafetyController()

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "td3_actor.pth"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained TD3 model not found: {model_path}"
            )

        self.td3_agent.actor.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )

        self.td3_agent.actor.eval()

        self.model_path = model_path

    def get_control_action(self, state):
        """
        State:
        [velocity, braking_power, SOC, SOH, temperature, road_friction]
        """

        state = np.asarray(state, dtype=np.float32)

        if state.shape != (6,):
            raise ValueError(
                f"Expected 6-state input, got shape {state.shape}"
            )
        normalized_state = np.array([
            state[0] / 40.0,
            state[1] / 500000.0,
            state[2],
            state[3],
            (state[4] - 20.0) / 40.0,
            (state[5] - 0.30) / 0.70
        ], dtype=np.float32)

        requested_k_reg = self.td3_agent.select_action(
            normalized_state,
        )

        # Safety override
        final_k_reg, safety_status = (
            self.safety_controller.apply_safety(
                requested_k_reg,
                float(state[2]),   # SOC
                float(state[4])    # Temperature
            )
        )

        return {
            "requested_k_reg": requested_k_reg,
            "final_k_reg": final_k_reg,
            "safety_status": safety_status
        }


if __name__ == "__main__":

    controller = IntegratedController()

    print("===== INTEGRATED CONTROLLER TEST =====")
    print(f"TD3 Model: {controller.model_path}")

    # Normal operating condition
    state = np.array(
        [20.0, 100000.0, 0.60, 1.00, 25.0, 0.80],
        dtype=np.float32
    )

    result = controller.get_control_action(state)

    print("\nNormal Condition")
    print(f"Requested k_reg : {result['requested_k_reg']:.4f}")
    print(f"Final k_reg     : {result['final_k_reg']:.4f}")
    print(f"Safety Status   : {result['safety_status']}")

    # High SOC safety test
    high_soc_state = np.array(
        [20.0, 100000.0, 0.96, 1.00, 25.0, 0.80],
        dtype=np.float32
    )

    result = controller.get_control_action(high_soc_state)

    print("\nHigh SOC Condition")
    print(f"Requested k_reg : {result['requested_k_reg']:.4f}")
    print(f"Final k_reg     : {result['final_k_reg']:.4f}")
    print(f"Safety Status   : {result['safety_status']}")

    # High temperature safety test
    high_temp_state = np.array(
        [20.0, 100000.0, 0.60, 1.00, 46.0, 0.80],
        dtype=np.float32
    )

    result = controller.get_control_action(high_temp_state)

    print("\nHigh Temperature Condition")
    print(f"Requested k_reg : {result['requested_k_reg']:.4f}")
    print(f"Final k_reg     : {result['final_k_reg']:.4f}")
    print(f"Safety Status   : {result['safety_status']}")