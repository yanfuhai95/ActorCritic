from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return F.softmax(self.layers(x), dim=1)


class Critic(nn.Module):

    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.layers(x)


class ActorCritic:
    def __init__(self,
                 state_dim, action_dim,
                 actor_hidden_dim=128, critic_hidden_dim=64,
                 actor_lr=1e-3, critic_lr=1e-2,
                 actor_optimizer=optim.AdamW, critic_optimizer=optim.AdamW,
                 gamma=0.98,
                 device="cpu"):
        self.actor = Actor(state_dim, action_dim,
                           hidden_dim=actor_hidden_dim).to(device)
        self.critic = Critic(
            state_dim, hidden_dim=critic_hidden_dim).to(device)

        self.actor_optimizer = actor_optimizer(
            self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = critic_optimizer(
            self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.device = device

    def select_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        prob_dist = self.actor(state)
        action_dist = torch.distributions.Categorical(prob_dist)
        action = action_dist.sample()
        return action.item()

    def optimize(self, transitions):
        batch = Transition(*zip(*transitions))

        states = torch.tensor(
            batch.state, dtype=torch.float).to(self.device)
        actions = torch.tensor(
            batch.action).view(-1, 1).to(self.device)
        rewards = torch.tensor(
            batch.reward, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(
            batch.next_state, dtype=torch.float).to(self.device)
        dones = torch.tensor(
            batch.done, dtype=torch.float).view(-1, 1).to(self.device)

        # Target of temporal difference
        td_target = rewards + self.gamma * \
            self.critic(next_states) * (1 - dones)

        # Delta of temporal difference
        td_delta = td_target - self.critic(states)

        log_probs = torch.log(self.actor(
            states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def save(self, path: str):
        torch.save(self.actor.state_dict(), path)

    def load(self, path: str):
        self.actor.load_state_dict(torch.load(path))
