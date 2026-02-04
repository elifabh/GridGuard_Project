import torch
import pandas as pd
import os
from gridguard.data import GridDataset
from gridguard.forecaster import WindForecaster
from gridguard.agent import GridAgent
from gridguard.simulation import VectorizedGridEnvironment
from gridguard.training import train_agent
from gridguard.visualization import plot_optimization_results

def main():
    # 1. Detect Device (Supercomputer GPU check)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 GridGuard Éire System Initializing...")
    print(f"🖥️  Compute Device: {device}")
    if device.type == 'cuda':
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    
    # 2. Load Data (Synthetic Generation)
    # The 'csv_path' is ignored because we are using the synthetic generator in data.py
    dataset = GridDataset(csv_path='dummy_path.csv')
    
    # 3. Initialize Agent (DQN)
    # State: [Battery, Price, Wind Forecast, Demand] -> Dim: 4
    # Action: [Charge, Discharge, Hold] -> Dim: 3
    state_dim = 4 
    action_dim = 3 
    agent = GridAgent(state_dim, action_dim, device)
    print("🤖 AI Agent Initialized (DQN Architecture).")
    
    # 4. Initialize Simulation Environment
    # We use the dataframe from the dataset to drive the simulation.
    n_envs = 32 # Parallel environments for HPC efficiency
    env = VectorizedGridEnvironment(dataset.df, n_envs=n_envs)
    print(f"🌍 Simulation Environment ready with {n_envs} parallel instances.")
    
    # 5. Training Loop
    print("\n--- ⚡ STARTING HPC TRAINING SIMULATION ⚡ ---")
    # Training for 5000 steps to show convergence quickly
    train_agent(agent, env, num_steps=5000, batch_size=128)
    
    # 6. Visualization & Metrics
    print("\n--- 📊 Generating Optimization Results ---")
    # We visualize a slice of the data where there is interesting activity
    plot_optimization_results(agent, dataset.df, start_idx=100, steps=48, output_file='optimization_results.png')
    print("\n✅ PROJECT EXECUTION COMPLETE.")
    print("Please check 'optimization_results.png' for the performance graph.")

if __name__ == "__main__":
    main()