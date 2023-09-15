import os
import time
from itertools import count

from tqdm import tqdm
import numpy as np
import torch
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt

from actorcritic.model import ActorCritic, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

if torch.cuda.is_available():
    num_episodes = 5000
else:
    num_episodes = 500

converged_threshold = 495

episode_durations = []
episode_rewards = []

fig, axs = plt.subplots(ncols=2, figsize=(24, 10))


def plot_result(show_result=False):
    # Duration
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        axs[0].set_title('Result')
    else:
        axs[0].cla()
        axs[0].set_title('Training...')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Duration')
    axs[0].plot(durations_t.numpy())

    # Take 10 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(90), means))
        axs[0].plot(means.numpy())

    # Reward
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        axs[1].set_title('Result')
    else:
        axs[1].cla()
        axs[1].set_title('Training...')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Reward')
    axs[1].plot(rewards_t.numpy())
    # Take 10 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        axs[1].plot(means.numpy())

    plt.pause(0.001)   # pause a bit so that plots are updated

    if show_result:
        plt.tight_layout()
        plt.show()


def make_env():
    env = gym.make("CartPole-v1")
    return env


def is_converged():
    if len(episode_rewards) >= 100:
        return np.mean(episode_rewards[-100:]) >= converged_threshold
    return False


if __name__ == "__main__":
    env = make_env()
    state, _ = env.reset()

    state_dim = len(state)
    action_dim = env.action_space.n

    actor_hidden_dim = 128
    actor_lr = 1e-4

    critic_hidden_dim = 64
    critic_lr = 1e-3

    agent = ActorCritic(
        state_dim, action_dim,
        actor_hidden_dim=actor_hidden_dim, critic_hidden_dim=critic_hidden_dim,
        actor_lr=actor_lr, critic_lr=critic_lr, device=device)

    converged = False
    for i in range(10):
        if converged:
            break

        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                state, _ = env.reset()

                episode_reward = 0.
                transitions = []

                for t in count():
                    action = agent.select_action(state)
                    observation, reward, terminated, truncated, _ = env.step(
                        action)
                    episode_reward += reward

                    done = terminated or truncated
                    transitions.append(Transition(
                        state, action, reward, observation, done))

                    if done:
                        episode_durations.append(t + 1)
                        break
                    
                    # Move to next state
                    state = observation

                agent.optimize(transitions)

                episode_rewards.append(episode_reward)
                plot_result()

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'reward':
                        '%.3f' % np.mean(episode_rewards[-10:])
                    })
                pbar.update(1)

                if is_converged():
                    print("Converged")
                    converged = True
                    break

    if not os.path.exists('model'):
        os.makedirs('model')

    agent.save('model/cart_pole.pth')

    print('Complete')
    plot_result(show_result=True)
    plt.ioff()
    plt.show()
