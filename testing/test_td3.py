import os
import numpy as np
import torch

from simulation.braking_environment import BrakingEnvironment
from controllers.td3_agent import Actor


def load_trained_actor():

    device = torch.device("cpu")

    actor = Actor(
        state_dim=6,
        action_dim=1
    ).to(device)

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "td3_actor.pth"
    )

    actor.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    actor.eval()

    return actor, device
def normalize_state(state):

    state = np.asarray(
        state,
        dtype=np.float32
    )

    return np.array([
        state[0] / 40.0,
        state[1] / 500000.0,
        state[2],
        state[3],
        (state[4] - 20.0) / 40.0,
        (state[5] - 0.30) / 0.70
    ], dtype=np.float32)


def test_td3():

    env = BrakingEnvironment()

    actor, device = load_trained_actor()

    print("===== TRAINED TD3 MODEL TEST =====")
    print()

    total_reward = 0.0

    for episode in range(1, 6):

        state, _ = env.reset()
        initial_speed = state[0]

        done = False
        episode_reward = 0.0
        steps = 0

        k_reg_values = []
        regen_power_values = []

        while not done and steps < 500:

            normalized_state = normalize_state(state)

            state_tensor = torch.tensor(
                normalized_state,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            with torch.no_grad():

                action = actor(
                    state_tensor
                ).cpu().numpy()[0]

            action = np.clip(
                action,
                0.0,
                1.0
            )

            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            episode_reward += reward

            k_reg_values.append(
                info["k_reg"]
            )

            regen_power_values.append(
                info["regen_power"]
            )

            state = next_state

            done = terminated or truncated

            steps += 1

        total_reward += episode_reward

        print(f"Episode {episode}")
        print(f"Initial Speed     : {initial_speed:.2f} m/s")
        print(f"Average k_reg     : {np.mean(k_reg_values):.4f}")
        print(
            f"Average Regen Power: "
            f"{np.mean(regen_power_values) / 1000:.2f} kW"
        )
        print(f"Episode Reward    : {episode_reward:.3f}")
        print(f"Steps             : {steps}")
        print()

    print("===== TD3 TEST COMPLETE =====")

    print(
        f"Average Test Reward: "
        f"{total_reward / 5:.3f}"
    )


if __name__ == "__main__":
    test_td3()