import os
import sys
import numpy as np
import torch
import shap

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from controllers.td3_agent import TD3Agent


FEATURE_NAMES = [
    "Vehicle Speed",
    "Braking Power",
    "SOC",
    "SOH",
    "Battery Temperature",
    "Road Friction"
]


def normalize_state(state):
    state = np.asarray(state, dtype=np.float32)

    return np.array([
        state[0] / 40.0,
        state[1] / 500000.0,
        state[2],
        state[3],
        (state[4] - 20.0) / 40.0,
        (state[5] - 0.30) / 0.70
    ], dtype=np.float32)


class TD3ModelWrapper:
    def __init__(self):
        self.agent = TD3Agent()

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "td3_actor.pth"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found: {model_path}"
            )

        self.agent.actor.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )

        self.agent.actor.eval()

    def predict(self, states):
        states = np.asarray(states, dtype=np.float32)

        with torch.no_grad():
            state_tensor = torch.tensor(
                states,
                dtype=torch.float32
            )

            actions = self.agent.actor(state_tensor)

        return actions.numpy()


class SHAPExplainer:
    def __init__(self):
        self.model = TD3ModelWrapper()

        # Small normalized background dataset
        self.background = np.array([
            [0.25, 0.20, 0.60, 1.00, 0.125, 0.71],
            [0.40, 0.30, 0.70, 0.98, 0.25, 0.80],
            [0.55, 0.50, 0.75, 0.95, 0.375, 0.60],
            [0.70, 0.65, 0.80, 0.92, 0.50, 0.50],
            [0.80, 0.80, 0.85, 0.90, 0.625, 0.40],
        ], dtype=np.float32)

        self.explainer = shap.KernelExplainer(
            self.model.predict,
            self.background
        )

    def explain(self, state):
        normalized_state = normalize_state(state)

        shap_values = self.explainer.shap_values(
            normalized_state.reshape(1, -1),
            nsamples=100
        )

        prediction = self.model.predict(
            normalized_state.reshape(1, -1)
        )[0][0]

        # Handle SHAP output format
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[0]).flatten()
        else:
            values = np.asarray(shap_values).flatten()

        return {
            "prediction": float(prediction),
            "shap_values": values,
            "feature_names": FEATURE_NAMES
        }



if __name__ == "__main__":

    print("===== SHAP XAI TEST =====")

    explainer = SHAPExplainer()

    # Example EV state
    state = np.array([
        20.0,       # Vehicle Speed
        150000.0,   # Braking Power
        0.60,       # SOC
        1.00,       # SOH
        25.0,       # Battery Temperature
        0.80        # Road Friction
    ], dtype=np.float32)

    result = explainer.explain(state)

    print()
    print(f"Predicted k_reg : {result['prediction']:.4f}")

    print()
    print("===== SHAP FEATURE CONTRIBUTIONS =====")

    for name, value in zip(
        result["feature_names"],
        result["shap_values"]
    ):
        print(f"{name:25s}: {value:+.6f}")

    print()
    print("===== TOP CONTRIBUTING FEATURE =====")

    values = np.abs(result["shap_values"])
    top_index = np.argmax(values)

    print(
        f"{result['feature_names'][top_index]} "
        f"-> SHAP = {result['shap_values'][top_index]:+.6f}"
    )

    print()
    print("===== SHAP TEST COMPLETE =====")