import numpy as np
import torch

class VectorizedGridEnvironment:
    def __init__(self, df, n_envs=1, battery_capacity=100.0):
        self.df = df
        self.n_envs = n_envs
        self.battery_capacity = battery_capacity
        self.prices = df['price'].values
        self.winds = df['generation'].values
        self.current_steps = np.zeros(n_envs, dtype=int)
        self.battery_levels = np.zeros(n_envs)

    def _get_states(self):
        states = []
        for i in range(self.n_envs):
            idx = self.current_steps[i]
            # Basit state: [Batarya%, Fiyat, Rüzgar, Saat]
            state = [
                self.battery_levels[i] / self.battery_capacity,
                self.prices[idx] / 100.0, # Normalize price
                self.winds[idx] / 100.0,
                (idx % 96) / 96.0 # Time of day
            ]
            states.append(state)
        return np.array(states)

    def step(self, actions):
        rewards = []
        next_states = []
        
        for i, action in enumerate(actions):
            idx = self.current_steps[i]
            current_price = self.prices[idx]
            
            # Action: 0=Charge, 1=Discharge, 2=Hold
            reward = 0
            if action == 0: # Charge
                if self.battery_levels[i] < self.battery_capacity:
                    self.battery_levels[i] += 5 # 5kWh charge speed
                    reward = -current_price * 0.1 # Cost
            elif action == 1: # Discharge
                if self.battery_levels[i] > 0:
                    self.battery_levels[i] -= 5
                    reward = current_price * 0.1 # Profit
            
            self.battery_levels[i] = np.clip(self.battery_levels[i], 0, self.battery_capacity)
            self.current_steps[i] += 1
            rewards.append(reward)

        return self._get_states(), rewards, False, {}