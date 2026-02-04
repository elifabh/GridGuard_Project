import pandas as pd
import numpy as np

class GridDataset:
    def __init__(self, csv_path=None):
        self.df = self.generate_irish_grid_data()

    def generate_irish_grid_data(self):
        print("⚡ Loading EirGrid Synthetic Profiles...")
        
        hours = 24 * 365 
        intervals = hours * 4 
        
        time_index = pd.date_range(start="2024-01-01", periods=intervals, freq="15min")
        
        # Talep Eğrisi (Numpy ile hesaplama)
        hour_of_day = time_index.hour.values + time_index.minute.values / 60.0
        daily_pattern = -np.cos((hour_of_day - 4) * 2 * np.pi / 24) 
        base_demand = 4000 + (daily_pattern * 1500) 
        demand = base_demand + np.random.normal(0, 200, intervals)
        
        # Rüzgar
        wind_signal = np.zeros(intervals)
        wind_val = 1500 
        for i in range(1, intervals):
            change = np.random.normal(0, 50) 
            wind_val = np.clip(wind_val + change, 0, 4500)
            wind_signal[i] = wind_val
            
        # Fiyat
        base_price = 150 
        scarcity_premium = np.clip(demand - 5000, 0, None) * 0.2
        wind_discount = (wind_signal * 0.04) 
        
        price = base_price + scarcity_premium - wind_discount + np.random.normal(0, 10, intervals)
        price = np.clip(price, -50, 500) 
        
        df = pd.DataFrame({
            'timestamp': time_index,
            'generation': wind_signal / 4500.0 * 100,
            'price': price / 100.0,
            'demand': demand
        })
        
        print(f"✅ EirGrid Data Ready: {len(df)} points.")
        return df