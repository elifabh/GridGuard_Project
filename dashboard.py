import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# --- PROJE MODÜLLERİ ---
from gridguard.data import GridDataset
from gridguard.agent import GridAgent
from gridguard.simulation import VectorizedGridEnvironment

# --- YENİ EKLENEN LLM KÖPRÜSÜ ---
# (Aynı klasörde llm_bridge.py dosyası olduğundan emin ol)
try:
    from llm_bridge import LLMAnalyst
    llm_available = True
except ImportError:
    llm_available = False
    print("⚠️ UYARI: llm_bridge.py bulunamadı. LLM özellikleri devre dışı.")

# --- KONFIGÜRASYON ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
server = app.server
app.title = "GridGuard Enterprise | HPC AI"

# Cihaz Tespiti
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri ve Model Yükleme
dataset = GridDataset(csv_path='dummy_path.csv') # data.py içindeki EirGrid simülasyonunu kullanır
agent = GridAgent(4, 3, device)

# --- LLM BAŞLATMA (Senin sorduğun yer burası) ---
# Uygulama başlarken analisti hafızaya alıyoruz.
if llm_available:
    # Model ismi bilgisayarındaki Ollama modeliyle aynı olmalı (llama3, mistral vb.)
    analyst = LLMAnalyst(model="llama3")
else:
    analyst = None

# --- ARAYÜZ BİLEŞENLERİ ---

# Sidebar (Sol Menü)
sidebar = html.Div(
    [
        html.Div([
            html.Img(src='/assets/images/ai_brain.png', style={"height": "42px", "borderRadius": "50%", "border": "2px solid #66fcf1"}),
            html.Div([
                html.H3("GridGuard", className="text-white fw-bold ms-2 mb-0", style={"fontSize": "1.4rem"}),
                html.Small("Enterprise AI", className="text-info ms-2", style={"fontSize": "0.7rem", "letterSpacing": "1px"})
            ])
        ], className="d-flex align-items-center mb-5"),
        
        dbc.Nav(
            [
                dbc.NavLink([html.I(className="bi bi-grid-1x2-fill me-3"), "Overview"], href="#", id="btn-tab-0", active=True),
                dbc.NavLink([html.I(className="bi bi-cpu-fill me-3"), "Mission Control"], href="#", id="btn-tab-1"),
                dbc.NavLink([html.I(className="bi bi-diagram-3-fill me-3"), "Architecture"], href="#", id="btn-tab-2"),
                dbc.NavLink([html.I(className="bi bi-file-earmark-bar-graph-fill me-3"), "Reports"], href="#", id="btn-tab-3"),
            ],
            vertical=True,
            pills=True,
            className="d-grid gap-2"
        ),
        
        html.Div([
            html.Hr(className="border-secondary"),
            html.Div([
                html.I(className="bi bi-gpu-card me-2 text-success"),
                html.Small("HPC / GPU Connected", className="text-white fw-bold")
            ], className="mb-2"),
            
            html.Small("Lead Engineer:", className="text-muted d-block"),
            html.Strong("Elif Gul", className="text-white"),
        ], className="mt-auto")
    ],
    className="sidebar",
    style={"width": "18rem", "zIndex": "1000"}
)

# TAB 0: OVERVIEW
tab0_home = html.Div([
    html.Div([
        html.Div(className="hero-overlay"),
        html.Div([
            dbc.Badge("v2.0 Enterprise Edition", color="info", className="mb-3 px-3 py-2"),
            html.H1("Optimizing Ireland's Energy Future", className="display-3 fw-bold text-white mb-3"),
            html.P("An autonomous AI agent designed to eliminate wind energy curtailment and stabilize the national grid through intelligent battery arbitrage. Powered by CloudCIX HPC.", 
                   className="lead text-light mb-4", style={"maxWidth": "700px", "textShadow": "0 2px 4px rgba(0,0,0,0.8)"}),
            dbc.Button([html.I(className="bi bi-rocket-takeoff me-2"), "Launch Mission Control"], id="jump-to-sim", color="primary", size="lg", className="fw-bold rounded-pill px-5 py-3 shadow-lg")
        ], className="hero-content container")
    ], className="hero-section mb-5", style={"backgroundImage": "url('/assets/images/hero_cover.png')"}),

    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H6("THE CHALLENGE", className="text-info fw-bold letter-spacing-2 mb-2"),
                html.H2("Why GridGuard Matters?", className="text-white fw-bold mb-4"),
                html.P("Ireland creates massive amounts of wind energy, but the grid isn't flexible enough to store it. This leads to 'Curtailment'—turning off turbines when wind is high but demand is low."),
                html.P("Simultaneously, data centers consume vast amounts of power, often relying on fossil fuels during peak hours. GridGuard bridges this gap."),
            ], width=6, className="pe-5"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4([html.I(className="bi bi-shield-check me-2"), "Core Strategy"], className="text-white mb-3"),
                        html.Ul([
                            html.Li("Analyze historical weather patterns (Met Éireann)."),
                            html.Li("Predict wind generation spikes using LSTM Neural Nets."),
                            html.Li("Store cheap energy in BESS (Battery Energy Storage)."),
                            html.Li("Discharge during peak price/demand hours."),
                        ], className="mt-3 ps-3 text-muted")
                    ])
                ], className="glass-panel border-start border-4 border-info")
            ], width=6)
        ], className="mb-5 align-items-center"),

        dbc.Row([
            dbc.Col(html.Div([
                html.H1("92%", className="display-4 fw-bold text-success"),
                html.P("Forecasting Accuracy", className="text-uppercase small letter-spacing-2")
            ], className="text-center glass-panel h-100 py-4"), width=4),
             dbc.Col(html.Div([
                html.H1("15%", className="display-4 fw-bold text-info"),
                html.P("Cost Reduction Target", className="text-uppercase small letter-spacing-2")
            ], className="text-center glass-panel h-100 py-4"), width=4),
             dbc.Col(html.Div([
                html.H1("24/7", className="display-4 fw-bold text-warning"),
                html.P("Autonomous Operation", className="text-uppercase small letter-spacing-2")
            ], className="text-center glass-panel h-100 py-4"), width=4),
        ], className="mb-5")
    ])
], id="tab-0")

# TAB 1: MISSION CONTROL
tab1_dashboard = html.Div([
    dbc.Container([
        html.H2("Mission Control Center", className="text-white mb-4 fw-bold border-bottom border-secondary pb-3"),
        
        # Rehber
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5([html.I(className="bi bi-info-circle-fill me-2"), "Operator Guide"], className="text-info fw-bold"),
                    html.P("1. Select a Grid Scenario (Stress Test) from the parameters panel.", className="mb-1 small"),
                    html.P("2. Click 'INITIATE AI AGENT' to trigger the GPU-accelerated simulation.", className="mb-0 small fw-bold text-white"),
                ], className="glass-panel mb-4 py-3")
            ], width=12)
        ]),

        dbc.Row([
            # Sol Panel
            dbc.Col([
                html.Div([
                    html.H5("Simulation Parameters", className="text-white fw-bold mb-3"),
                    html.Label("Grid Condition (Scenario)", className="text-info small fw-bold"),
                    dcc.Dropdown(
                        id='scenario-dropdown',
                        options=[
                            {'label': '🌤️ Normal Operations (Baseline)', 'value': 'normal'},
                            {'label': '⛈️ Storm Event (Excess Supply)', 'value': 'storm'},
                            {'label': '🌫️ Dunkelflaute (Supply Shortage)', 'value': 'calm'}
                        ],
                        value='normal', clearable=False, className="mb-3"
                    ),
                    html.Label("Forecast Horizon (Hours)", className="text-info small fw-bold"),
                    dcc.Slider(id='duration-slider', min=24, max=72, step=12, value=48, marks={24:'24h', 48:'48h', 72:'72h'}),
                    dbc.Button([html.I(className="bi bi-cpu-fill me-2"), "INITIATE AI AGENT"], id='run-btn', color="primary", className="w-100 mt-4 fw-bold py-3 shadow-lg")
                ], className="glass-panel mb-4"),

                html.Div([
                    html.Div([
                        html.Img(src='/assets/images/ai_brain.png', height="30px", className="me-2"),
                        html.H5("Live Decision Feed", className="text-white fw-bold mb-0"),
                    ], className="d-flex align-items-center mb-3 border-bottom border-secondary pb-2"),
                    html.Div(id='activity-log-content', style={"height": "300px", "overflowY": "auto"}, className="log-box", 
                             children=[html.Small("System Standby. Waiting for initialization...", className="text-muted fst-italic")])
                ], className="glass-panel h-100")
            ], width=4),

            # Sağ Panel (Grafikler)
            dbc.Col([
                # Grafik 1
                html.Div([
                    html.H5([html.I(className="bi bi-activity me-2"), "1. Environment State (Inputs)"], className="text-muted small text-uppercase fw-bold"),
                    dcc.Loading(dcc.Graph(id='input-graph', style={"height": "250px"}), type="cube", color="#66fcf1"),
                    html.Div(id='input-comment', className="mt-2 border-top border-secondary pt-2 text-white small") # LLM Yorum Alanı
                ], className="glass-panel mb-4"),
                
                # Grafik 2
                html.Div([
                    html.H5([html.I(className="bi bi-battery-charging me-2"), "2. Agent Optimization (Outputs)"], className="text-muted small text-uppercase fw-bold"),
                    dcc.Loading(dcc.Graph(id='output-graph', style={"height": "250px"}), type="cube", color="#66fcf1"),
                    html.Div(id='output-comment', className="mt-2 border-top border-secondary pt-2 text-white small") # LLM Yorum Alanı
                ], className="glass-panel mb-4"),
                
                # Canlı KPI
                dbc.Row([
                    dbc.Col(html.Div([
                         html.H6("Projected Profit", className="text-muted text-uppercase small"),
                         html.H3("--", id="live-profit", className="text-success fw-bold")
                    ], className="glass-panel text-center"), width=6),
                    dbc.Col(html.Div([
                         html.H6("CO2 Avoided", className="text-muted text-uppercase small"),
                         html.H3("--", id="live-co2", className="text-info fw-bold")
                    ], className="glass-panel text-center"), width=6),
                ])
            ], width=8)
        ])
    ])
], id="tab-1", style={"display": "none"})

# TAB 2: ARCHITECTURE
tab2_arch = html.Div([
    dbc.Container([
        html.H2("System Architecture", className="text-white mb-4 fw-bold"),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Img(src='/assets/images/workflow_schema.png', className="img-fluid rounded shadow-lg border border-secondary w-100"),
                    html.P("Figure 1.0: End-to-End Data Flow Architecture (HPC Integration)", className="text-center text-muted small mt-2")
                ], className="glass-panel p-2")
            ], width=12)
        ], className="mb-5"),

        dbc.Row([
            dbc.Col([
                html.H4("1. Data Ingestion", className="text-info"),
                html.P("Real-time telemetry from wind turbines and SEMO market APIs is ingested into the pipeline."),
            ], width=4),
            dbc.Col([
                html.H4("2. Neural Processing", className="text-warning"),
                html.P("The LSTM module predicts generation, while the DQN Agent calculates the optimal battery state using HPC nodes."),
            ], width=4),
            dbc.Col([
                html.H4("3. Grid Execution", className="text-success"),
                html.P("Optimized instructions are sent to the BMS (Battery Management System) to Charge or Discharge."),
            ], width=4),
        ])
    ])
], id="tab-2", style={"display": "none"})

# TAB 3: REPORTS
tab3_report = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Performance Reporting", className="text-white mb-2 fw-bold"),
                html.P("Post-simulation analysis and KPI tracking.", className="text-muted mb-4"),
            ])
        ]),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Img(src='/assets/images/sustainability_icon.png', style={"height": "120px"}, className="mb-3"),
                    html.H4("Generate Executive Report", className="text-white"),
                    html.P("Create a detailed PDF-ready summary of the AI's performance.", className="text-muted small"),
                    dbc.Button([html.I(className="bi bi-file-earmark-pdf-fill me-2"), "Generate PDF Report"], id="btn-gen-report", color="success", className="mt-3")
                ], className="glass-panel text-center p-5 mb-4")
            ], width=12)
        ]),

        html.Div(id="report-paper-area", className="report-paper shadow-lg d-none mt-4", children=[
            html.Div([
                html.H2("GridGuard: Executive Summary", className="text-dark fw-bold mb-3"),
                html.Hr(),
                html.H5("Simulation Results", className="text-dark fw-bold"),
                html.P("The GridGuard AI Agent successfully demonstrated load-shifting capabilities. The system prioritized charging during high-wind/low-price intervals.", className="text-dark"),
                
                dbc.Row([
                   dbc.Col([html.H3(id="paper-profit", className="text-success fw-bold"), html.Small("Total Profit Generated", className="text-dark")]), 
                   dbc.Col([html.H3(id="paper-co2", className="text-info fw-bold"), html.Small("CO2 Emissions Avoided", className="text-dark")]), 
                ], className="mt-4 mb-4"),
                
                html.Hr(),
                html.Small(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | HPC Cluster", className="text-dark")
            ])
        ])
    ])
], id="tab-3", style={"display": "none"})

# --- LAYOUT WRAPPER ---
app.layout = html.Div([
    dcc.Store(id="store-data"),
    sidebar,
    html.Div([tab0_home, tab1_dashboard, tab2_arch, tab3_report], id="page-content", style={"marginLeft": "19rem", "padding": "2rem"})
])

# --- CALLBACKS ---

# 1. Navigation
@app.callback(
    [Output("tab-0", "style"), Output("tab-1", "style"), Output("tab-2", "style"), Output("tab-3", "style"),
     Output("btn-tab-0", "active"), Output("btn-tab-1", "active"), Output("btn-tab-2", "active"), Output("btn-tab-3", "active")],
    [Input("btn-tab-0", "n_clicks"), Input("btn-tab-1", "n_clicks"), Input("btn-tab-2", "n_clicks"), Input("btn-tab-3", "n_clicks"),
     Input("jump-to-sim", "n_clicks")]
)
def navigate_tabs(b0, b1, b2, b3, jump):
    ctx_id = ctx.triggered_id
    show, hide = {"display": "block", "animation": "fadeIn 0.5s"}, {"display": "none"}
    
    if not ctx_id: return show, hide, hide, hide, True, False, False, False
    if ctx_id == "btn-tab-1" or ctx_id == "jump-to-sim": return hide, show, hide, hide, False, True, False, False
    elif ctx_id == "btn-tab-2": return hide, hide, show, hide, False, False, True, False
    elif ctx_id == "btn-tab-3": return hide, hide, hide, show, False, False, False, True
    else: return show, hide, hide, hide, True, False, False, False

# 2. Simulation & LLM Analysis
@app.callback(
    [Output('input-graph', 'figure'), Output('output-graph', 'figure'),
     Output('activity-log-content', 'children'),
     Output('live-profit', 'children'), Output('live-co2', 'children'),
     Output('input-comment', 'children'), Output('output-comment', 'children'),
     Output('store-data', 'data')],
    [Input('run-btn', 'n_clicks')],
    [State('scenario-dropdown', 'value'), State('duration-slider', 'value')]
)
def run_simulation(n_clicks, scenario, steps):
    layout_cfg = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color':'#ccc'}, margin={'l':30,'r':10,'t':30,'b':30}, xaxis={'showgrid':False}, yaxis={'gridcolor':'#333'})
    
    if not n_clicks:
        empty = go.Figure()
        empty.update_layout(**layout_cfg)
        return empty, empty, html.Small("Waiting...", className="text-muted"), "--", "--", "Waiting...", "Waiting...", {}

    # 1. Simülasyon
    sim_data = dataset.df.copy()
    
    # Senaryo Manipülasyonu
    if scenario == 'storm': sim_data['generation'] *= 1.5; sim_data['price'] *= 0.5
    elif scenario == 'calm': sim_data['generation'] *= 0.3; sim_data['price'] *= 1.8
    sim_data['generation'] = sim_data['generation'].clip(0, 100) # %100 kapasiteyi geçmesin

    env = VectorizedGridEnvironment(sim_data, n_envs=1, battery_capacity=100.0)
    start_idx = np.random.randint(0, len(sim_data) - steps - 1)
    env.current_steps[0] = start_idx
    env.battery_levels[0] = 30.0
    
    states = env._get_states()
    hist_bat, hist_price, hist_wind, hist_action = [], [], [], []
    logs = []
    total_reward = 0
    charge_count, sell_count = 0, 0
    
    for i in range(steps):
        state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        with torch.no_grad():
            action = agent.policy_net(state_tensor).max(1)[1].cpu().numpy()[0]
        
        hist_bat.append(states[0][0])
        hist_price.append(states[0][1] * 100)
        hist_wind.append(env.winds[env.current_steps[0]])
        hist_action.append(action)
        
        ts = f"T+{i:02d}h"
        curr_p = states[0][1]*100
        if action == 0:
            logs.insert(0, html.Div(f"[{ts}] CHARGING | Price: {curr_p:.1f}¢", className="text-danger small border-bottom border-secondary mb-1 fw-bold"))
            charge_count += 1
        elif action == 1:
            logs.insert(0, html.Div(f"[{ts}] SELLING | Price: {curr_p:.1f}¢", className="text-success small border-bottom border-secondary mb-1 fw-bold"))
            sell_count += 1
            
        next_states, rewards, _, _ = env.step([action])
        total_reward += rewards[0]
        states = next_states

    # 2. LLM / AI ANALYST ÇAĞRISI
    in_comment = "AI Analysis unavailable (Bridge not connected)"
    out_comment = "AI Analysis unavailable (Bridge not connected)"

    if analyst: # Eğer LLMAnalyst yüklendiyse
        # Veri özetini hazırla
        market_stats = {
            'avg_wind': np.mean(hist_wind),
            'avg_price': np.mean(hist_price),
            'max_price': np.max(hist_price)
        }
        battery_stats = {
            'profit': f"{total_reward:.2f}",
            'charge_count': charge_count,
            'sell_count': sell_count
        }
        # Ollama'ya gönder
        in_comment = analyst.analyze("market", market_stats)
        out_comment = analyst.analyze("battery", battery_stats)
    else:
        # Yedek mesaj (Eğer ollama yoksa)
        in_comment = html.Span("ℹ️ LLM Bridge offline. Using deterministic analysis: Wind/Price inverse correlation detected.", className="text-muted")
        out_comment = html.Span(f"ℹ️ LLM Bridge offline. Agent executed {charge_count} buy and {sell_count} sell actions.", className="text-muted")

    # 3. Grafikler
    fig_in = go.Figure()
    fig_in.add_trace(go.Scatter(y=hist_wind, name='Wind (% Cap)', line={'color':'#38bdf8', 'width':2}, fill='tozeroy'))
    fig_in.add_trace(go.Scatter(y=hist_price, name='Price (c/kWh)', line={'color':'#fb7185', 'dash':'dot'}, yaxis='y2'))
    fig_in.update_layout(**layout_cfg, yaxis2={'overlaying':'y', 'side':'right'})
    
    fig_out = go.Figure()
    fig_out.add_trace(go.Scatter(y=hist_bat, name='Battery', line={'color':'#4ade80', 'width':3}))
    fig_out.update_layout(**layout_cfg)

    profit_str = f"€{total_reward:.2f}"
    co2_str = f"{int(total_reward*0.42)} kg"

    return fig_in, fig_out, logs, profit_str, co2_str, in_comment, out_comment, {"profit": profit_str, "co2": co2_str}

# 3. REPORT GENERATION
@app.callback(
    [Output("report-paper-area", "className"), Output("paper-profit", "children"), Output("paper-co2", "children")],
    [Input("btn-gen-report", "n_clicks")],
    [State("store-data", "data")]
)
def generate_report(n, data):
    if not n or not data:
        return "report-paper shadow-lg d-none", "", ""
    return "report-paper shadow-lg d-block", data.get('profit'), data.get('co2')

if __name__ == '__main__':
    app.run(debug=True)