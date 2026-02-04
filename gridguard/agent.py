import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DuelingDQN(nn.Module):
    """
    HPC Seviyesi: Dueling Deep Q-Network.
    Durum değerini (Value) ve Aksiyon avantajını (Advantage) ayrı hesaplar.
    Daha stabil ve akıllı kararlar verir.
    """
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        # Özellik Çıkarıcı (Feature Extractor) - Derin Ağ
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2), # Overfitting engelleme
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Value Stream (Durumun değeri ne?)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream (Hangi aksiyon daha iyi?)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Q = V + (A - mean(A))
        return values + (advantages - advantages.mean())

class GridAgent:
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # HPC Optimization: Modeli GPU'ya taşı
        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Hedef ağ sadece tahmin yapar
        
        # Optimizer: AdamW (Daha modern ve stabil)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-4)
        self.loss_fn = nn.SmoothL1Loss() # Huber Loss (Hatalara karşı daha dayanıklı)
        
        # Experience Replay Buffer (Hafıza)
        self.memory = deque(maxlen=50000) # HPC belleği geniştir
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state):
        # Epsilon-Greedy Stratejisi
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
            
    def train_step(self, batch_size=64):
        # (HPC Notu: Burada toplu öğrenme (batch learning) yapılır)
        pass # Dashboard modunda eğitim yapmıyoruz, sadece inference (tahmin)