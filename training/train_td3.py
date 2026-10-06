import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from simulation.braking_environment import BrakingEnvironment
from controllers.td3_agent import Actor, Critic


# =========================================================
# REPLAY BUFFER
# =========================================================

class ReplayBuffer:

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# =========================================================
# STATE NORMALIZATION
# =========================================================

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

# =========================================================
# TD3 AGENT
# =========================================================

class TD3Trainer:

    def __init__(self, state_dim=6, action_dim=1):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Training Device: {self.device}")

        self.state_dim = state_dim
        self.action_dim = action_dim

        # -----------------------------
        # Actor
        # -----------------------------

        self.actor = Actor(
            state_dim,
            action_dim
        ).to(self.device)

        self.actor_target = Actor(
            state_dim,
            action_dim
        ).to(self.device)

        # -----------------------------
        # Twin Critics
        # -----------------------------

        self.critic1 = Critic(
            state_dim,
            action_dim
        ).to(self.device)

        self.critic2 = Critic(
            state_dim,
            action_dim
        ).to(self.device)

        self.critic1_target = Critic(
            state_dim,
            action_dim
        ).to(self.device)

        self.critic2_target = Critic(
            state_dim,
            action_dim
        ).to(self.device)

        # Copy initial weights
        self.actor_target.load_state_dict(
            self.actor.state_dict()
        )

        self.critic1_target.load_state_dict(
            self.critic1.state_dict()
        )

        self.critic2_target.load_state_dict(
            self.critic2.state_dict()
        )

        # -----------------------------
        # Optimizers
        # -----------------------------

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=1e-4
        )

        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(),
            lr=1e-3
        )

        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(),
            lr=1e-3
        )

        # -----------------------------
        # TD3 parameters
        # -----------------------------

        self.gamma = 0.99
        self.tau = 0.005

        self.policy_noise = 0.2
        self.noise_clip = 0.5

        self.policy_delay = 2

        self.total_iterations = 0

    # =====================================================
    # SELECT ACTION
    # =====================================================

    def select_action(self, state, noise=0.1):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            action = self.actor(
                state_tensor
            ).cpu().numpy()[0]

        if noise > 0:

            action += np.random.normal(
                0,
                noise,
                size=self.action_dim
            )

        return np.clip(
            action,
            0.0,
            1.0
        )

    # =====================================================
    # TRAIN
    # =====================================================

    def train(
        self,
        replay_buffer,
        batch_size=64
    ):

        if len(replay_buffer) < batch_size:
            return None

        self.total_iterations += 1

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = replay_buffer.sample(batch_size)

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.float32,
            device=self.device
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        # -------------------------------------------------
        # Target policy smoothing
        # -------------------------------------------------

        noise = torch.randn_like(actions) * self.policy_noise

        noise = torch.clamp(
            noise,
            -self.noise_clip,
            self.noise_clip
        )

        next_actions = (
            self.actor_target(next_states)
            + noise
        )

        next_actions = torch.clamp(
            next_actions,
            0.0,
            1.0
        )

        # -------------------------------------------------
        # Target Q values
        # -------------------------------------------------

        target_q1 = self.critic1_target(
            next_states,
            next_actions
        )

        target_q2 = self.critic2_target(
            next_states,
            next_actions
        )

        target_q = torch.min(
            target_q1,
            target_q2
        )

        target_q = rewards + (
            1.0 - dones
        ) * self.gamma * target_q.detach()

        # -------------------------------------------------
        # Critic 1 update
        # -------------------------------------------------

        current_q1 = self.critic1(
            states,
            actions
        )

        critic1_loss = nn.MSELoss()(
            current_q1,
            target_q
        )

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # -------------------------------------------------
        # Critic 2 update
        # -------------------------------------------------

        current_q2 = self.critic2(
            states,
            actions
        )

        critic2_loss = nn.MSELoss()(
            current_q2,
            target_q
        )

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # -------------------------------------------------
        # Delayed Actor update
        # -------------------------------------------------

        actor_loss = None

        if self.total_iterations % self.policy_delay == 0:

            actor_actions = self.actor(states)

            actor_loss = -self.critic1(
                states,
                actor_actions
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft target update
            for target_param, param in zip(
                self.actor_target.parameters(),
                self.actor.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data
                    + (1.0 - self.tau) * target_param.data
                )

            for target_param, param in zip(
                self.critic1_target.parameters(),
                self.critic1.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data
                    + (1.0 - self.tau) * target_param.data
                )

            for target_param, param in zip(
                self.critic2_target.parameters(),
                self.critic2.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data
                    + (1.0 - self.tau) * target_param.data
                )

        return {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": (
                None
                if actor_loss is None
                else float(actor_loss.item())
            )
        }


# =========================================================
# TRAINING LOOP
# =========================================================

def train_td3():

    env = BrakingEnvironment()

    agent = TD3Trainer(
        state_dim=6,
        action_dim=1
    )

    replay_buffer = ReplayBuffer(
        capacity=100000
    )

    episodes = 50

    batch_size = 64

    print()
    print("===== TD3 TRAINING STARTED =====")
    print(f"Episodes    : {episodes}")
    print(f"Batch Size  : {batch_size}")
    print()

    rewards_history = []

    for episode in range(1, episodes + 1):

        state, _ = env.reset()

        episode_reward = 0.0

        done = False

        steps = 0

        while not done and steps < 500:
            normalized_state = normalize_state(state)
            action = agent.select_action(
                normalized_state,
                noise=0.10
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            done = terminated or truncated

            normalized_next_state = normalize_state(next_state)

            replay_buffer.add(
                normalized_state,
                action,
                reward,
                normalized_next_state,
                float(done)
            )

            agent.train(
                replay_buffer,
                batch_size
            )

            state = next_state

            episode_reward += reward

            steps += 1

        rewards_history.append(
            episode_reward
        )

        if episode % 5 == 0:

            average_reward = np.mean(
                rewards_history[-5:]
            )

            print(
                f"Episode {episode:3d} | "
                f"Reward: {episode_reward:8.3f} | "
                f"Avg(5): {average_reward:8.3f} | "
                f"Buffer: {len(replay_buffer)}"
            )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models"
    )

    os.makedirs(model_dir, exist_ok=True)

    torch.save(
        agent.actor.state_dict(),
        os.path.join(model_dir,"td3_actor.pth")
    )

    torch.save(
        agent.critic1.state_dict(),
        os.path.join(model_dir, "td3_critic1.pth")
    )

    torch.save(
        agent.critic2.state_dict(),
        os.path.join(model_dir, "td3_critic2.pth")
    )

    np.save(
        os.path.join(model_dir, "training_rewards.npy"),
        np.array(rewards_history)
    )

    print()
    print("===== TRAINING COMPLETE =====")
    print("Actor saved   :", os.path.join(model_dir, "td3_actor.pth"))
    print("Critic 1 saved:", os.path.join(model_dir, "td3_critic1.pth"))
    print("Critic 2 saved:", os.path.join(model_dir, "td3_critic2.pth"))
    print("Rewards saved :", os.path.join(model_dir, "training_rewards.npy"))


if __name__ == "__main__":
    train_td3()