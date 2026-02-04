import matplotlib.pyplot as plt
import numpy as np
import torch
from .simulation import VectorizedGridEnvironment

def plot_optimization_results(agent, df, start_idx=0, steps=48, output_file='optimization_results.png'):
    print(f"Generating visualization for {steps} steps...")

    env = VectorizedGridEnvironment(df, n_envs=1, battery_capacity=100.0)

    start_idx = min(start_idx, env.max_steps - steps - 1)
    env.current_steps[0] = start_idx
    env.battery_levels[0] = 50.0 

    states = env._get_states()

    history_battery = []
    history_actions = []
    history_prices = []
    history_winds = []

    for _ in range(steps):
        state_tensor = torch.tensor(states, dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            action = agent.policy_net(state_tensor).max(1)[1].cpu().numpy()

        history_battery.append(states[0][0])
        history_prices.append(states[0][1])

        current_step_idx = env.current_steps[0]
        actual_wind = env.winds[current_step_idx]
        history_winds.append(actual_wind)

        history_actions.append(action[0])

        next_states, _, _, _ = env.step(action)
        states = next_states

    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.set_xlabel('Time (Hours)')
    ax1.set_ylabel('Price ($) / Wind (kW)', color='black')

    l1, = ax1.plot(history_prices, color='blue', label='Grid Price', alpha=0.7)
    l2, = ax1.plot(history_winds, color='cyan', linestyle='--', label='Wind Gen', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Battery Level (kWh)', color='green')
    l3, = ax2.plot(history_battery, color='green', linewidth=2, label='Battery Level')
    ax2.tick_params(axis='y', labelcolor='green')

    for i, act in enumerate(history_actions):
        if act == 0: # Charge
            ax2.scatter(i, history_battery[i], color='red', marker='^', s=100, zorder=5, label='Charge' if i == 0 else "")
        elif act == 1: # Discharge
            ax2.scatter(i, history_battery[i], color='orange', marker='v', s=100, zorder=5, label='Discharge' if i == 0 else "")

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title('Optimization Results: Agent Strategy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")