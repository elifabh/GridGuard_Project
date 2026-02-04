import numpy as np

class VectorizedGridEnvironment:
    """
    Simulates multiple grid environments in parallel for HPC training.
    """
    def __init__(self, df, n_envs=32, battery_capacity=100.0):
        self.df = df
        self.n_envs = n_envs
        self.capacity = battery_capacity

        # Ensure necessary columns exist
        if 'generation' in df.columns:
             self.winds = df['generation'].values
        elif 'wind_generation' in df.columns:
             self.winds = df['wind_generation'].values
        else:
             raise ValueError("Dataframe must contain 'generation'")

        self.demands = df['demand'].values

        if 'price' in df.columns:
            self.prices = df['price'].values
        else:
            self.prices = (self.demands / self.demands.max()) * 0.20

        self.max_steps = len(self.df) - 25

        self.battery_levels = np.zeros(n_envs)
        self.current_steps = np.random.randint(0, self.max_steps, size=n_envs) 

    def reset(self):
        self.battery_levels = np.zeros(self.n_envs)
        self.current_steps = np.random.randint(0, self.max_steps, size=self.n_envs)
        return self._get_states()

    def step(self, actions):
        rewards = np.zeros(self.n_envs)
        dones = np.zeros(self.n_envs, dtype=bool)

        current_prices = self.prices[self.current_steps]
        current_demands = self.demands[self.current_steps]
        current_winds = self.winds[self.current_steps]

        charge_rate = 10.0
        discharge_rate = 10.0

        for i in range(self.n_envs):
            action = actions[i]
            price = current_prices[i]
            demand = current_demands[i]
            wind = current_winds[i]

            if action == 0: # Charge
                amount = min(self.capacity - self.battery_levels[i], charge_rate)
                self.battery_levels[i] += amount
                energy_flow = -amount 
            elif action == 1: # Discharge
                amount = min(self.battery_levels[i], discharge_rate)
                self.battery_levels[i] -= amount
                energy_flow = amount 
            else: # Hold
                energy_flow = 0

            available_power = wind + energy_flow
            surplus = available_power - demand

            if surplus >= 0:
                profit = surplus * price 
                reward = profit 
                reward += 1.0 
            else:
                deficit = abs(surplus)
                cost = deficit * price * 2.0 
                reward = -cost
                reward -= 5.0

            rewards[i] = reward

            self.current_steps[i] += 1
            if self.current_steps[i] >= self.max_steps:
                dones[i] = True
                self.current_steps[i] = np.random.randint(0, self.max_steps)
                self.battery_levels[i] = 0.0

        next_states = self._get_states()
        return next_states, rewards, dones, {}

    def _get_states(self):
        states = np.zeros((self.n_envs, 4), dtype=np.float32)
        for i in range(self.n_envs):
            step = self.current_steps[i]
            wind_forecast = np.mean(self.winds[step : step+24])
            states[i] = [
                self.battery_levels[i],
                self.prices[step],
                wind_forecast,
                self.demands[step]
            ]
        return states