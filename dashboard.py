import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# Modüller
from gridguard.data import GridDataset
from gridguard.agent import GridAgent
from gridguard.simulation import VectorizedGridEnvironment
try:
    from llm_bridge import LLMAnalyst
    llm_available = True
except:
    llm_available = False

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
app.title = "GridGuard Enterprise"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = GridDataset()
agent = GridAgent(4, 3, device)
analyst = LLMAnalyst(model="llama3") if llm_available else None

# --- ARAYÜZ (Kısaltılmış Sidebar) ---
sidebar = html.Div([
    html.H3("GridGuard", className="text-white fw-bold mb-4"),
    dbc.Nav([
        dbc.NavLink("Overview", href="#", id="btn-tab-0", active=True),
        dbc.NavLink("Mission Control", href="#", id="btn-tab-1"),
    ], vertical=True, pills=True),
    html.Hr(className="border-secondary"),
    html.Small(f"System: {device}", className="text-muted")
], className="sidebar")

tab1_dashboard = html.Div([
    dbc.Container([
        html.H2("Mission Control", className="text-white mb-4"),
        dbc.Row([
            dbc.Col([
                html.Label("Scenario", className="text-info"),
                dcc.Dropdown(id='scenario', options=[
                    {'label': 'Normal', 'value': 'normal'},
                    {'label': 'Storm', 'value': 'storm'}
                ], value='normal', className="mb-3"),
                dbc.Button("INITIATE AI", id='run-btn', color="primary", className="w-100")
            ], width=3),
            dbc.Col([
                dcc.Loading(dcc.Graph(id='main-graph'), type="cube"),
                html.Div(id='ai-comment', className="mt-2 text-white")
            ], width=9)
        ])
    ])
], id="tab-1", style={"display": "none"})

# Overview Tab (Hero Section)
tab0_home = html.Div([
    dbc.Container([
        html.H1("GridGuard Enterprise", className="display-3 text-white fw-bold"),
        html.P("Autonomous AI for National Grid Stabilization", className="lead text-info"),
        dbc.Button("Launch System", id="jump-btn", color="success", size="lg", className="mt-4")
    ], className="py-5")
], id="tab-0")

app.layout = html.Div([
    sidebar,
    html.Div([tab0_home, tab1_dashboard], id="page-content", style={"marginLeft": "19rem", "padding": "2rem"})
])

@app.callback(
    [Output("tab-0", "style"), Output("tab-1", "style")],
    [Input("btn-tab-0", "n_clicks"), Input("btn-tab-1", "n_clicks"), Input("jump-btn", "n_clicks")]
)
def render_content(b0, b1, jump):
    ctx_id = ctx.triggered_id
    if ctx_id in ["btn-tab-1", "jump-btn"]: return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}

@app.callback(
    [Output('main-graph', 'figure'), Output('ai-comment', 'children')],
    [Input('run-btn', 'n_clicks')], [State('scenario', 'value')]
)
def update_sim(n, scenario):
    if not n: return go.Figure(), "Waiting for initialization..."
    
    sim_data = dataset.df.copy()
    if scenario == 'storm': sim_data['generation'] *= 1.5
    
    env = VectorizedGridEnvironment(sim_data)
    env.current_steps[0] = np.random.randint(0, 1000)
    states = env._get_states()
    
    hist_bat, hist_price = [], []
    steps = 48
    
    for _ in range(steps):
        state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        action = agent.select_action(state_tensor)
        
        hist_bat.append(states[0][0])
        hist_price.append(states[0][1]*100)
        
        next_states, _, _, _ = env.step([action])
        states = next_states

    # LLM Analiz
    comment = "AI Analysis Offline"
    if analyst:
        stats = {'avg_wind': 50, 'avg_price': np.mean(hist_price), 'max_price': np.max(hist_price)}
        comment = analyst.analyze("market", stats)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=hist_bat, name='Battery', line=dict(color='#39ff14', width=3)))
    fig.add_trace(go.Scatter(y=hist_price, name='Price', yaxis='y2', line=dict(color='#f96d00', dash='dot')))
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis2=dict(overlaying='y', side='right')
    )
    
    return fig, comment

if __name__ == '__main__':
    app.run_server(debug=True)