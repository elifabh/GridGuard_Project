import pandas as pd
import numpy as np

class GridDataset:
    def __init__(self, csv_path=None):
        self.df = self.generate_irish_grid_data()

    def generate_irish_grid_data(self):
        """
        İrlanda (EirGrid) Profili:
        - Rüzgar: Atlantik cephelerinden gelen değişken yapı (Weibull dağılımı).
        - Talep: Sabah artar, akşam 17:00-19:00 (Tea Time) zirve yapar, gece düşer.
        - Fiyat: Gaz fiyatlarına endeksli, rüzgar artınca düşer (Negative pricing).
        """
        print("⚡ Loading EirGrid Synthetic Profiles...")
        
        hours = 24 * 365 # 1 Yıl
        intervals = hours * 4 # 15dk çözünürlük
        
        # 1. Zaman İndeksi
        time_index = pd.date_range(start="2024-01-01", periods=intervals, freq="15min")
        
        # 2. İrlanda Talep Eğrisi (Akşam Zirvesi)
        # Günlük döngü (Sinüs dalgası + gürültü)
        hour_of_day = time_index.hour + time_index.minute / 60.0
        daily_pattern = -np.cos((hour_of_day - 4) * 2 * np.pi / 24) # Sabah 4'te dip, akşam 16'da zirve
        base_demand = 4000 + (daily_pattern * 1500) # MW cinsinden (İrlanda ortalaması)
        demand = base_demand + np.random.normal(0, 200, intervals)
        
        # 3. Rüzgar Üretimi (Stokastik Süreç)
        # Rüzgar bir anda değişmez, yavaş değişir (Autocorrelation)
        wind_signal = np.zeros(intervals)
        wind_val = 1500 # Başlangıç MW
        for i in range(1, intervals):
            change = np.random.normal(0, 50) # Değişim hızı
            wind_val = np.clip(wind_val + change, 0, 4500) # İrlanda kurulu gücü ~4.5GW
            wind_signal[i] = wind_val
            
        # 4. Piyasa Fiyatı (Merit Order Effect)
        # Talep yüksekse fiyat artar, Rüzgar yüksekse fiyat ÇÖKER.
        base_price = 150 # €/MWh (Gaz maliyeti)
        scarcity_premium = (demand - 5000).clip(0, None) * 0.2 # Kıtlık primi
        wind_discount = (wind_signal * 0.04) # Rüzgar indirimi
        
        price = base_price + scarcity_premium - wind_discount + np.random.normal(0, 10, intervals)
        price = np.clip(price, -50, 500) # Negatif fiyatlar mümkün!
        
        # Dashboard için veriyi normalize et (0-1 arası değil, gerçek değerler kalsın, dashboardda böleriz)
        df = pd.DataFrame({
            'timestamp': time_index,
            'generation': wind_signal / 4500.0 * 100, # % Kapasite Faktörü (Grafik için)
            'price': price / 100.0, # Cent/kWh cinsine çevir (Daha anlaşılır)
            'demand': demand
        })
        
        print(f"✅ EirGrid Data Ready: {len(df)} points. Avg Price: {df['price'].mean():.2f}¢")
        return df