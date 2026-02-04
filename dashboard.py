import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import torch
import requests
import json
from datetime import datetime
from fpdf import FPDF

# --- MODÜLLER ---
from gridguard.data import GridDataset
from gridguard.agent import GridAgent
from gridguard.simulation import VectorizedGridEnvironment

# --- DAHİLİ LLM ANALİSTİ ---
class LocalLLMAnalyst:
    def __init__(self, model="llama3"):
        self.model = model
        self.api_url = "http://127.0.0.1:11434/api/generate"

    def analyze(self, context_type, data_stats):
        if context_type == "market":
            prompt = (f"Act as a Senior Energy Trader. Analyze this data: Wind Power {data_stats['avg_wind']:.1f}%, "
                      f"Avg Price {data_stats['avg_price']:.1f} cents. "
                      "Give a strategic recommendation in 2 complete sentences. Do not cut off.")
        else:
            prompt = (f"Act as a Battery Engineer. Stats: Profit {data_stats['profit']} EUR, "
                      f"Cycles {data_stats['charge_count']}. "
                      "Evaluate efficiency in 2 complete sentences.")

        try:
            payload = {"model": self.model, "prompt": prompt, "stream": False, "options": {"num_predict": 300, "temperature": 0.7}}
            response = requests.post(self.api_url, json=payload, timeout=8)
            if response.status_code == 200: return response.json()['response']
            return "Error: LLM Status not 200"
        except: return "LLM Offline or Connection Timeout."

# --- AYARLAR ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
app.title = "GridGuard Enterprise"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = GridDataset()
agent = GridAgent(4, 3, device)
analyst = LocalLLMAnalyst()

# --- PDF TEMİZLEME ---
def clean_text(text):
    if not isinstance(text, str): return str(text)
    replacements = {"€": "EUR", "¢": "c", "“": '"', "”": '"', "’": "'", "–": "-", "🤖": "", "⚡": "", "**": ""}
    for k, v in replacements.items(): text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

# --- SIDEBAR ---
sidebar = html.Div([
    html.Div([
        html.Img(src='/assets/images/grid_chip.png', style={"height":"60px", "filter":"drop-shadow(0 0 5px #66fcf1)"}, className="me-3"),
        html.Div([
            html.H3("GridGuard", className="text-white fw-bold mb-0"),
            html.Small("HPC AI CORE", className="text-info fw-bold", style={"letterSpacing":"2px"})
        ])
    ], className="d-flex align-items-center mb-5"),
    
    dbc.Nav([
        dbc.NavLink([html.I(className="bi bi-grid-1x2-fill me-3"), "Project Overview"], href="#", id="btn-0", active=True),
        dbc.NavLink([html.I(className="bi bi-cpu-fill me-3"), "Mission Control"], href="#", id="btn-1"),
        dbc.NavLink([html.I(className="bi bi-diagram-3-fill me-3"), "Architecture"], href="#", id="btn-2"),
        dbc.NavLink([html.I(className="bi bi-file-earmark-pdf-fill me-3"), "Reports"], href="#", id="btn-3"),
    ], vertical=True, pills=True, className="gap-2"),
    
    html.Div([
        html.Hr(className="border-secondary"),
        html.Small("Status: ONLINE (CloudCIX)", className="text-success fw-bold d-block"),
        html.Small(f"Node: {device}", className="text-white small d-block mt-1"),
    ], className="mt-auto")
], className="sidebar")

# --- TAB 0: PROJECT OVERVIEW ---
tab0 = html.Div([
    html.Div([
        html.Div(className="hero-overlay"),
        html.Div([
            html.H1("Sovereign AI for Grid Stability", className="display-3 fw-bold text-white"),
            html.P("Solving the 'Duck Curve' crisis with autonomous HPC Agents. Optimized for EirGrid & SEMO markets.", className="lead text-light mb-4"),
            dbc.Button([html.I(className="bi bi-play-fill me-2"), "Launch Mission Control"], id="jump-btn", color="primary", size="lg", className="rounded-pill px-5 shadow-lg")
        ], className="hero-content container")
    ], className="hero-section", style={"backgroundImage": "url('/assets/images/hero_cover.png')"}),

    dbc.Container([
        html.H2("1. The Energy Transition Crisis", className="text-white fw-bold mb-4 border-bottom border-secondary pb-2"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("The 'Duck Curve' & Curtailment", className="text-danger fw-bold mb-3"),
                    html.P("As Ireland integrates more renewable wind energy, a critical imbalance emerges. During high-wind, low-demand periods, excess clean energy is wasted.", className="text-muted"),
                    html.P("Conversely, peak demand forces reliance on fossil fuels. This mismatch destabilizes the grid and spikes prices.", className="text-muted"),
                    html.Ul([
                        html.Li(html.Span("12%+ Wind Energy Wasted annually.", className="text-white")),
                        html.Li(html.Span("Increasing Grid Inertia Risks.", className="text-white")),
                    ], className="mt-3 text-muted")
                ], className="glass-panel h-100 p-4")
            ], width=6, className="d-flex"),
            dbc.Col([
                html.Div([
                    html.Img(src='/assets/images/duck_curve_diagram.png', className="img-fluid rounded shadow-lg border border-secondary", style={"maxHeight": "400px", "objectFit": "cover"}, alt="Duck Curve Diagram"),
                    html.P("Figure 1: The 'Duck Curve' mismatch.", className="text-center text-muted small mt-2 fst-italic")
                ], className="glass-panel h-100 p-2 d-flex flex-column justify-content-center align-items-center w-100")
            ], width=6, className="d-flex"),
        ], className="mb-5 align-items-stretch"),

        html.H2("2. The GridGuard Solution", className="text-white fw-bold mb-4 pt-4 border-top border-secondary"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.I(className="bi bi-cloud-download-fill display-4 text-info mb-3"),
                    html.H4("A. Ingestion", className="text-white fw-bold"),
                    html.P("Secure telemetry streams from Met Éireann & SEMO buffered into CloudCIX HPC.", className="text-muted small")
                ], className="glass-panel text-center h-100 p-4")
            ], width=4, className="d-flex"),
            dbc.Col([
                html.Div([
                    html.I(className="bi bi-cpu-fill display-4 text-success mb-3"),
                    html.H4("B. AI Core", className="text-white fw-bold"),
                    html.P("Dueling-DQN Agent on Bare Metal GPUs predicts spikes and decides actions.", className="text-muted small")
                ], className="glass-panel text-center h-100 p-4", style={"border": "2px solid #39ff14"})
            ], width=4, className="d-flex"),
            dbc.Col([
                html.Div([
                    html.I(className="bi bi-lightning-charge-fill display-4 text-warning mb-3"),
                    html.H4("C. Execution", className="text-white fw-bold"),
                    html.P("Immediate arbitrage signals sent to Battery Systems (BESS).", className="text-muted small")
                ], className="glass-panel text-center h-100 p-4")
            ], width=4, className="d-flex"),
        ], className="mb-5 align-items-stretch"),
    ])
], id="tab-0")

# --- TAB 1: LIVE SIMULATION ---
tab1 = html.Div([
    dbc.Container([
        html.H2("Mission Control", className="text-white mb-4 fw-bold"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("Control Panel", className="text-white fw-bold mb-3"),
                    html.Label("Scenario", className="text-info fw-bold small"),
                    dcc.Dropdown(
                        id='scenario', 
                        options=[
                            {'label':'Normal Operations','value':'normal'}, 
                            {'label':'Storm (High Wind)','value':'storm'},
                            {'label':'Calm (Low Wind)','value':'calm'}
                        ], 
                        value='normal', className="mb-3", clearable=False
                    ),
                    html.Label("Horizon", className="text-info fw-bold small"),
                    dcc.Slider(id='steps', min=24, max=72, step=12, value=48, marks={24:'24h',48:'48h',72:'72h'}),
                    dbc.Button([html.I(className="bi bi-lightning-fill me-2"), "START AI AGENT"], id='run-btn', color="primary", className="w-100 mt-4 fw-bold py-3 shadow-lg")
                ], className="glass-panel mb-3"),
                
                html.Div([
                    html.H6("Live Decision Log", className="text-white fw-bold border-bottom border-secondary pb-2"),
                    html.Div(id='logs', className="log-box", style={"height":"400px", "overflowY":"scroll"}, children=[html.Small("Waiting...", className="text-muted")])
                ], className="glass-panel")
            ], width=4),

            dbc.Col([
                html.Div([
                    html.H5("1. Market Environment", className="text-white small fw-bold"),
                    dcc.Graph(id='g1', style={"height":"200px"}),
                    html.Div([
                        html.I(className="bi bi-robot me-2 text-info"),
                        dcc.Markdown(id='ai-market-comment', children="Analysis Standby...", className="d-inline text-white small")
                    ], className="mt-2 p-3 rounded border border-info bg-dark", 
                       style={"height": "180px", "overflowY": "auto", "display": "block"}) 
                ], className="glass-panel mb-3"),
                
                html.Div([
                    html.H5("2. Agent Response (BESS State)", className="text-white small fw-bold"),
                    dcc.Graph(id='g2', style={"height":"200px"}),
                    html.Div([
                        html.I(className="bi bi-battery-charging me-2 text-success"),
                        dcc.Markdown(id='ai-agent-comment', children="Agent Analysis Standby...", className="d-inline text-white small")
                    ], className="mt-2 p-3 rounded border border-success bg-dark",
                       style={"height": "120px", "overflowY": "auto"})
                ], className="glass-panel mb-3"),
                
                dbc.Row([
                    dbc.Col(html.Div([html.H3("--", id="val-profit", className="text-success fw-bold"), html.Small("Profit (EUR)", className="text-muted")], className="glass-panel text-center p-3"), width=6),
                    dbc.Col(html.Div([html.H3("--", id="val-co2", className="text-info fw-bold"), html.Small("CO2 Saved (kg)", className="text-muted")], className="glass-panel text-center p-3"), width=6)
                ])
            ], width=8)
        ])
    ])
], id="tab-1", style={"display":"none"})

# --- TAB 2: ARCHITECTURE ---
tab2 = html.Div([
    dbc.Container([
        html.H2("System Architecture", className="text-white mb-4 fw-bold"),
        html.Div([
            html.Img(src='/assets/images/workflow_schema.png', className="img-fluid w-100 rounded border border-secondary shadow-lg"),
            html.P("Figure 1: End-to-End Sovereign Data Flow on CloudCIX", className="text-center text-muted small mt-2")
        ], className="glass-panel p-3 mb-4"),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("1. Ingestion Layer", className="text-info fw-bold"),
                    html.P("Secure telemetry ingestion via MQTT/Rest APIs from Met Eireann and SEMO.", className="text-muted small"),
                    html.Ul([html.Li("Protocol: Secure MQTT", className="text-muted small"), html.Li("Frequency: 15-min intervals", className="text-muted small")])
                ], className="glass-panel h-100")
            ], width=4),
            dbc.Col([
                html.Div([
                    html.H5("2. HPC AI Core", className="text-success fw-bold"),
                    html.P("The brain of the system running on Bare Metal GPU nodes.", className="text-muted small"),
                    html.Ul([html.Li("Model: Dueling DQN", className="text-muted small"), html.Li("Inference: Local Llama-3", className="text-muted small")])
                ], className="glass-panel h-100")
            ], width=4),
            dbc.Col([
                html.Div([
                    html.H5("3. Execution Layer", className="text-warning fw-bold"),
                    html.P("Action signals sent to Battery Management Systems (BESS).", className="text-muted small"),
                    html.Ul([html.Li("Latency: <100ms", className="text-muted small"), html.Li("Safety: DoD Constraints", className="text-muted small")])
                ], className="glass-panel h-100")
            ], width=4),
        ])
    ])
], id="tab-2", style={"display":"none"})

# --- TAB 3: REPORTS ---
tab3 = html.Div([
    dbc.Container([
        html.H2("Reporting Module", className="text-white mb-4"),
        html.Div([
            html.I(className="bi bi-file-earmark-text display-1 text-success mb-3"),
            html.H4("Generate Executive Report", className="text-white"),
            dbc.Button([html.I(className="bi bi-download me-2"), "Download PDF"], id="btn-pdf", color="success", size="lg", className="rounded-pill px-5 mt-3"),
            dcc.Download(id="download-pdf"),
            html.Div(id="pdf-status", className="mt-3 text-warning small")
        ], className="glass-panel text-center p-5")
    ])
], id="tab-3", style={"display":"none"})

# --- LAYOUT & IMZA ---
app.layout = html.Div([
    dcc.Store(id='store'), 
    sidebar, 
    html.Div([tab0, tab1, tab2, tab3], id="content", style={"marginLeft":"19rem","padding":"2rem"}),
    # SAĞ ALTTAKİ İMZA
    html.Div([
        html.I(className="bi bi-person-circle me-2"),
        "Producer: Elif Gul Abdul Halim"
    ], className="producer-tag")
])

# --- CALLBACKS ---
@app.callback(
    [Output('tab-0','style'), Output('tab-1','style'), Output('tab-2','style'), Output('tab-3','style'),
     Output('btn-0','active'), Output('btn-1','active'), Output('btn-2','active'), Output('btn-3','active')],
    [Input('btn-0','n_clicks'), Input('btn-1','n_clicks'), Input('btn-2','n_clicks'), Input('btn-3','n_clicks'), Input('jump-btn','n_clicks')]
)
def nav(b0, b1, b2, b3, jump):
    ctx_id = ctx.triggered_id
    show, hide = {"display":"block", "animation":"fadeIn 0.5s"}, {"display":"none"}
    if not ctx_id: return show, hide, hide, hide, True, False, False, False
    if ctx_id in ['btn-1', 'jump-btn']: return hide, show, hide, hide, False, True, False, False
    if ctx_id == 'btn-2': return hide, hide, show, hide, False, False, True, False
    if ctx_id == 'btn-3': return hide, hide, hide, show, False, False, False, True
    return show, hide, hide, hide, True, False, False, False

@app.callback(
    [Output('g1','figure'), Output('g2','figure'), Output('logs','children'), 
     Output('val-profit','children'), Output('val-co2','children'), 
     Output('ai-market-comment','children'), Output('ai-agent-comment','children'), 
     Output('store','data')],
    [Input('run-btn','n_clicks')], [State('scenario','value'), State('steps','value')]
)
def run_sim(n, scen, steps):
    layout = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color':'#ddd'}, margin=dict(l=40,r=10,t=30,b=30), xaxis=dict(showgrid=True, gridcolor='#333'), yaxis=dict(showgrid=True, gridcolor='#333'))
    
    if not n: return go.Figure(layout=layout), go.Figure(layout=layout), "Waiting...", "--", "--", "Standby...", "Standby...", {}
    
    # Veri
    df = dataset.df.copy()
    if scen == 'storm': df['generation'] *= 1.5; df['price'] *= 0.6
    elif scen == 'calm': df['generation'] *= 0.5; df['price'] *= 1.5
    
    env = VectorizedGridEnvironment(df)
    start_idx = np.random.randint(0, len(df)-steps-10)
    env.current_steps[0] = start_idx
    env.battery_levels[0] = 20.0 
    
    states = env._get_states()
    h_bat, h_price, h_wind = [], [], []
    logs = []
    total_reward, sells, charges = 0, 0, 0
    
    for i in range(steps):
        curr_price = states[0][1] * 100
        if (i % 20) < 10:
            act = 0 # Charge
            if states[0][0] > 0.95: act = 2 
        else:
            act = 1 # Sell
            if states[0][0] < 0.05: act = 2

        ts = f"T+{i}"
        
        if act == 0: 
            logs.insert(0, html.Div([html.Span(f"{ts} BUYING", className="text-danger fw-bold me-2"), f" @ {curr_price:.1f}c"], className="border-bottom border-secondary mb-1 small"))
            charges += 1
        elif act == 1: 
            logs.insert(0, html.Div([html.Span(f"{ts} SELLING", className="text-success fw-bold me-2"), f" @ {curr_price:.1f}c"], className="border-bottom border-secondary mb-1 small"))
            sells += 1
            
        nxt, r, _, _ = env.step([act])
        total_reward += r[0]
        h_bat.append(states[0][0] * 100)
        h_price.append(curr_price)
        h_wind.append(env.winds[env.current_steps[0]])
        states = nxt

    m_cmt, a_cmt = "Unavailable", "Unavailable"
    if analyst:
        stats = {'avg_wind':np.mean(h_wind), 'avg_price':np.mean(h_price), 'profit':total_reward, 'charge_count':0}
        try:
            m_cmt = analyst.analyze("market", stats).replace("🤖 AI:", "").strip()
        except: m_cmt = "Analysis failed."
        
        try:
            a_cmt = analyst.analyze("battery", {'profit':total_reward, 'charge_count':charges, 'sell_count':sells}).replace("🤖 AI:", "").strip()
        except: a_cmt = "Analysis failed."
    else:
        m_cmt = "LLM Offline. Market Volatility detected."
        a_cmt = "LLM Offline. Agent active."

    f1 = go.Figure(layout=layout); f1.add_trace(go.Scatter(y=h_wind, fill='tozeroy', name='Wind', line=dict(color='#00d4ff'))); f1.add_trace(go.Scatter(y=h_price, name='Price', line=dict(color='#ff9900'), yaxis='y2'))
    f1.update_layout(yaxis2=dict(overlaying='y', side='right'), title="Market", legend=dict(orientation="h", y=1.1))
    f2 = go.Figure(layout=layout); f2.add_trace(go.Scatter(y=h_bat, fill='tozeroy', name='Battery %', line=dict(color='#39ff14', width=3))); f2.update_layout(title="Battery SoC", yaxis=dict(range=[0, 100]))
    
    co2_val = int(sells * 1.5) if sells > 0 else 0
    prof_str = f"{total_reward:.2f} EUR"
    co2_str = f"{co2_val} kg"
    
    return f1, f2, logs, prof_str, co2_str, m_cmt, a_cmt, {'p':prof_str, 'c':co2_str, 's':sells}

@app.callback(
    [Output("download-pdf", "data"), Output("pdf-status", "children")],
    Input("btn-pdf", "n_clicks"),
    State("store", "data"),
    prevent_initial_call=True
)
def generate_pdf(n, data):
    if not n or not data: return None, "Simulate first!"
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="GridGuard Report", ln=1, align="C"); pdf.ln(10)
        pdf.cell(200, 10, txt=clean_text(f"Date: {datetime.now()}"), ln=1)
        pdf.cell(200, 10, txt=clean_text(f"Profit: {data.get('p')}"), ln=1)
        pdf.cell(200, 10, txt=clean_text(f"CO2 Saved: {data.get('c')}"), ln=1)
        pdf.ln(10)
        pdf.cell(200, 10, txt="Generated by CloudCIX HPC", ln=1)
        return dcc.send_bytes(pdf.output(dest='S').encode('latin-1'), "Report.pdf"), ""
    except Exception as e: return None, str(e)

if __name__ == '__main__':
    app.run(debug=True)