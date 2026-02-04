import torch
import numpy as np
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def train_agent(agent, env, num_steps=10000, batch_size=128, gamma=0.99):
    memory = ReplayMemory(100000)
    print(f"Starting training on {agent.device} with {env.n_envs} vectorized environments...")

    states = env.reset()

    for step in range(num_steps):
        states_tensor = torch.tensor(states, dtype=torch.float32, device=agent.device)
        epsilon = max(0.05, 0.9 - step / (num_steps * 0.5)) 

        with torch.no_grad():
            q_values = agent.policy_net(states_tensor)

        actions = q_values.max(1)[1].cpu().numpy()
        mask = np.random.rand(env.n_envs) < epsilon
        random_actions = np.random.randint(0, agent.action_dim, size=env.n_envs)
        actions[mask] = random_actions[mask]

        next_states, rewards, dones, _ = env.step(actions)

        for i in range(env.n_envs):
            memory.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

        states = next_states

        if len(memory) > batch_size:
            transitions = memory.sample(batch_size)
            batch = Transition(*zip(*transitions))

            state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=agent.device)
            action_batch = torch.tensor(np.array(batch.action), dtype=torch.long, device=agent.device).view(-1, 1)
            reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32, device=agent.device).view(-1, 1)
            next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=agent.device)
            done_batch = torch.tensor(np.array(batch.done), dtype=torch.float32, device=agent.device).view(-1, 1)

            loss = agent.optimize_model(state_batch, action_batch, reward_batch, next_state_batch, done_batch, gamma)

            if step % 100 == 0:
                agent.update_target_network()
                if step % 1000 == 0:
                    print(f"Step {step}/{num_steps}, Loss: {loss:.4f}, Epsilon: {epsilon:.2f}, Avg Reward: {np.mean(rewards):.2f}")

    print("Training Complete.")