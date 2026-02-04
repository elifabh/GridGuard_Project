


# ⚡ GridGuard Enterprise: Sovereign AI for Grid Stability

![Project Status](https://img.shields.io/badge/Status-Live_Simulation-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![AI Framework](https://img.shields.io/badge/PyTorch-Dueling_DQN-orange)
![Infrastructure](https://img.shields.io/badge/HPC-CloudCIX_Sovereign-green)

> **Solving Ireland's "Duck Curve" crisis with autonomous Agentic AI.**


## 📖 Project Overview

**GridGuard Enterprise** is an autonomous AI system designed to optimize energy grid stability. It addresses the critical issue of **Wind Curtailment**—where excess renewable energy is wasted due to grid inflexibility.

By deploying **Reinforcement Learning (Dueling DQN)** agents on **Sovereign HPC Infrastructure (CloudCIX)**, GridGuard predicts energy spikes, manages Battery Energy Storage Systems (BESS) in real-time, and ensures 100% data residency within Ireland.

### 🏆 Key Innovation
Unlike traditional "Chatbots," GridGuard is an **"Actionbot"**. It doesn't just analyze data; it makes millisecond-level decisions to Buy, Sell, or Hold energy, turning grid volatility into profit and carbon savings.

---

## 🚀 Mission Control Dashboard

The project features a state-of-the-art **Mission Control Interface** built with Plotly Dash.

| Feature | Description |
| :--- | :--- |
| **🤖 Autonomous Agent** | A Dueling-DQN agent that trades energy against live market data. |
| **📊 Real-time Analytics** | Live visualization of the "Duck Curve," market prices, and BESS state-of-charge (SoC). |
| **🧠 Local LLM Analyst** | An embedded AI analyst that explains market conditions in plain English (No API calls). |
| **📄 Executive Reporting** | One-click PDF generation for stakeholder reporting (Profit & CO2 metrics). |
| **🇮🇪 Data Sovereignty** | Designed to run on Bare Metal GPUs, ensuring no data leaves Irish jurisdiction. |

---

## 🛠️ Technical Architecture

The system operates on a three-layer architecture:

1.  **Ingestion Layer (MQTT/Rest):** Securely buffers telemetry from Met Éireann (Weather) and SEMO (Market Prices).
2.  **HPC AI Core (The Brain):**
    * **LSTM:** Forecasts wind generation.
    * **Dueling DQN:** Executes arbitrage strategies (Charge/Discharge).
3.  **Execution Layer (Edge):** Sends signals to BESS hardware with <100ms latency.

---

## 💻 Installation & Usage

### Prerequisites
* Python 3.8+
* CUDA (Optional, for GPU acceleration)

### 1. Clone the Repository
```bash
git clone [https://github.com/elifabh/GridGuard_Project.git](https://github.com/elifabh/GridGuard_Project.git)
cd GridGuard_Project

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

*(Key libraries: `dash`, `pandas`, `torch`, `plotly`, `fpdf`, `dash-bootstrap-components`)*

### 3. Launch Mission Control

```bash
python dashboard.py

```

### 4. Access the Dashboard

Open your browser and navigate to:
`http://127.0.0.1:8050/`

---

## 📂 Project Structure

```text
GridGuard_Project/
├── assets/                  # CSS, Images, and Fonts
│   ├── images/              # Chip icon, Duck Curve diagrams
│   └── style.css            # Neon UI styling
├── gridguard/               # Core AI Package
│   ├── agent.py             # Dueling DQN Model Definition
│   ├── data.py              # Data Ingestion & Preprocessing
│   └── simulation.py        # OpenAI Gym-style Environment
├── dashboard.py             # Main Application Entry Point
├── requirements.txt         # Python Dependencies
└── README.md                # Project Documentation

```

---

## 🌍 Impact Analysis

* **Economic:** Unlocks millions in lost revenue by arbitraging price differentials.
* **Environmental:** Reduces reliance on fossil-fuel peaker plants by utilizing stored wind energy.
* **Sovereign:** Demonstrates that critical infrastructure AI can be built and hosted entirely within Ireland.

---

## 👤 Author & Producer

**Elif Gul Abdul Halim**

* *AI Engineer & Project Lead*
* *Specialization:* Sovereign AI & HPC Systems

---

> *"The grid of the future doesn't just transmit power; it thinks."* — **GridGuard**


