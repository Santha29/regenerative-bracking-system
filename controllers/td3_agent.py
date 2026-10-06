import numpy as np
import torch
import torch.nn as nn


# =========================================================
# ACTOR NETWORK
# State → k_reg
# =========================================================

class Actor(nn.Module):

    def __init__(self, state_dim=6, action_dim=1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, action_dim),
            nn.Sigmoid()
        )

    def forward(self, state):
        return self.network(state)


# =========================================================
# CRITIC NETWORK
# State + Action → Q Value
# =========================================================

class Critic(nn.Module):

    def __init__(self, state_dim=6, action_dim=1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)


# =========================================================
# TD3 AGENT
# =========================================================

class TD3Agent:

    def __init__(self):

        self.state_dim = 6
        self.action_dim = 1

        self.actor = Actor(
            self.state_dim,
            self.action_dim
        )

        self.critic1 = Critic(
            self.state_dim,
            self.action_dim
        )

        self.critic2 = Critic(
            self.state_dim,
            self.action_dim
        )

    def select_action(self, state):

        state = np.asarray(
            state,
            dtype=np.float32
        )

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            action = self.actor(
                state_tensor
            ).cpu().numpy()[0]

        return float(action[0])


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    agent = TD3Agent()

    # Paper state:
    # [v, Pbrk, SOC, SOH, Tbatt, μroad]

    state = np.array([
        20.0,       # Vehicle speed (m/s)
        100000.0,   # Braking power (W)
        0.60,       # SOC
        1.00,       # SOH
        25.0,       # Battery temperature (°C)
        0.80        # Road friction coefficient
    ])

    action = agent.select_action(state)

    print("===== TD3 AGENT TEST =====")

    print(f"State Dimension : {agent.state_dim}")
    print(f"Action Dimension: {agent.action_dim}")

    print()

    print("State:")
    print(f"Speed           : {state[0]:.2f} m/s")
    print(f"Braking Power   : {state[1]:.2f} W")
    print(f"SOC             : {state[2]:.2f}")
    print(f"SOH             : {state[3]:.2f}")
    print(f"Temperature     : {state[4]:.2f} °C")
    print(f"Road Friction   : {state[5]:.2f}")

    print()

    print(f"TD3 k_reg       : {action:.4f}")

    if 0.0 <= action <= 1.0:
        print("Action Range    : VALID")
    else:
        print("Action Range    : INVALID")