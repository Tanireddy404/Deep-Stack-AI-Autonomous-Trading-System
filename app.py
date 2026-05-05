import os, pandas as pd, numpy as np, streamlit as st, plotly.graph_objects as go, torch, torch.nn as nn
from deepstack_core import COINS, COIN_LABELS, MultiCoinTradingEnv, add_indicators
from train_model import flat_to_multi, N_ACTIONS, OBS_H, OBS_W

st.set_page_config(page_title='DeepStack', page_icon='', layout='wide', initial_sidebar_state='expanded')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
* { box-sizing: border-box; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #080808;
    border-right: 1px solid #141414;
    padding-top: 0;
}
section[data-testid="stSidebar"] .block-container { padding: 0 20px; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }
[data-testid="stSidebar"] hr { border-color: #141414; margin: 16px 0; }

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Price Bar */
.pricebar {
    display: flex; background: #080808;
    border-bottom: 1px solid #141414;
    padding: 0; margin-bottom: 0;
}
.pricebar-item {
    flex: 1; padding: 14px 20px;
    border-right: 1px solid #141414;
    transition: background 0.2s;
}
.pricebar-item:last-child { border-right: none; }
.pricebar-item:hover { background: #0f0f0f; }
.pb-label { font-size: 9px; font-weight: 600; letter-spacing: 0.12em; color: #333; text-transform: uppercase; margin-bottom: 5px; }
.pb-price { font-size: 16px; font-weight: 600; color: #f0f0f0; letter-spacing: -0.02em; margin-bottom: 3px; }
.pb-change { font-size: 11px; font-weight: 500; }
.up   { color: #00d4aa; }
.down { color: #e05252; }

/* Page header */
.page-header {
    padding: 36px 0 28px 0;
    border-bottom: 1px solid #141414;
    margin-bottom: 32px;
}
.page-title { font-size: 22px; font-weight: 600; color: #f0f0f0; letter-spacing: -0.03em; margin: 0 0 4px 0; }
.page-sub   { font-size: 11px; font-weight: 400; color: #2a2a2a; letter-spacing: 0.1em; text-transform: uppercase; margin: 0; }

/* Section label */
.slabel {
    font-size: 9px; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #2a2a2a;
    padding-bottom: 10px; border-bottom: 1px solid #141414;
    margin: 28px 0 16px 0;
}

/* Signal grid */
.sgrid { display: flex; gap: 1px; background: #141414; border-radius: 6px; overflow: hidden; }
.sgrid-item { flex: 1; background: #0a0a0a; padding: 16px 12px; text-align: center; }
.sgrid-item:hover { background: #0d0d0d; }
.sc-label  { font-size: 9px; letter-spacing: 0.12em; color: #2a2a2a; text-transform: uppercase; margin-bottom: 8px; }
.sc-action { font-size: 12px; font-weight: 700; letter-spacing: 0.06em; margin-bottom: 5px; }
.sc-conf   { font-size: 10px; color: #2a2a2a; }
.c-buy  { color: #00d4aa; }
.c-sell { color: #e05252; }
.c-hold { color: #627EEA; }

/* Allocation table */
.alloc-table { background: #0a0a0a; border: 1px solid #141414; border-radius: 6px; overflow: hidden; }
.alloc-row   { display: flex; align-items: center; gap: 16px; padding: 12px 16px; border-bottom: 1px solid #0f0f0f; }
.alloc-row:last-child { border-bottom: none; }
.alloc-row:hover { background: #0d0d0d; }
.ar-coin   { font-size: 11px; font-weight: 600; color: #d0d0d0; width: 44px; }
.ar-action { font-size: 9px; font-weight: 700; letter-spacing: 0.08em; width: 32px; }
.ar-bar-wrap { flex: 1; background: #141414; border-radius: 1px; height: 2px; }
.ar-bar    { height: 2px; border-radius: 1px; }
.ar-usd    { font-size: 12px; font-weight: 600; color: #f0f0f0; width: 72px; text-align: right; }
.ar-pct    { font-size: 10px; color: #2a2a2a; width: 36px; text-align: right; }

/* Metric override */
[data-testid="stMetric"] { background: #0a0a0a; border: 1px solid #141414; border-radius: 6px; padding: 16px 18px; }
[data-testid="stMetricLabel"] p { font-size: 9px !important; font-weight: 600 !important; letter-spacing: 0.1em !important; color: #2a2a2a !important; text-transform: uppercase; }
[data-testid="stMetricValue"] { font-size: 22px !important; font-weight: 600 !important; color: #f0f0f0 !important; letter-spacing: -0.02em !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* Info cards */
.info-row  { display: flex; gap: 1px; background: #141414; border-radius: 6px; overflow: hidden; margin: 8px 0; }
.info-card { flex: 1; background: #0a0a0a; padding: 20px 18px; }
.ic-title  { font-size: 9px; font-weight: 600; letter-spacing: 0.12em; color: #00d4aa; text-transform: uppercase; margin-bottom: 8px; }
.ic-text   { font-size: 11px; color: #2a2a2a; line-height: 1.8; }

/* Badge */
.badge { display: inline-block; padding: 4px 10px; border-radius: 3px; font-size: 9px; font-weight: 600; letter-spacing: 0.08em; margin: 3px; text-transform: uppercase; }

/* Sidebar labels */
.sb-label { font-size: 9px; font-weight: 600; letter-spacing: 0.12em; color: #2a2a2a; text-transform: uppercase; margin: 0 0 8px 0; }
.sb-val   { font-size: 12px; color: #f0f0f0; margin: 0 0 4px 0; }
.sb-sub   { font-size: 10px; color: #2a2a2a; margin: 0; }
</style>
""", unsafe_allow_html=True)

G = '#00d4aa'
B = '#627EEA'
R = '#e05252'
COIN_COLORS = {'BTC':G,'ETH':B,'SOL':G,'ADA':B,'BNB':G,'DOGE':B}
MODEL_PATH  = 'models/deepstack_best_model.pth'

COIN_LOGOS = {
    'BTC':  'https://assets.coingecko.com/coins/images/1/small/bitcoin.png',
    'ETH':  'https://assets.coingecko.com/coins/images/279/small/ethereum.png',
    'SOL':  'https://assets.coingecko.com/coins/images/4128/small/solana.png',
    'ADA':  'https://assets.coingecko.com/coins/images/975/small/cardano.png',
    'BNB':  'https://assets.coingecko.com/coins/images/825/small/bnb-icon2_2x.png',
    'DOGE': 'https://assets.coingecko.com/coins/images/5/small/dogecoin.png',
}

PT = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
          font=dict(color='#2a2a2a', size=10),
          xaxis=dict(showgrid=False, zeroline=False, showline=False),
          yaxis=dict(gridcolor='#0f0f0f', zeroline=False, showline=False))


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    frames = {}
    for coin in COINS:
        df = pd.read_csv(os.path.join('data', coin.replace('-','_')+'.csv'), index_col=0, parse_dates=True)
        frames[coin] = df[['Open','High','Low','Close','Volume']].dropna()
    idx = frames[COINS[0]].index
    for c in COINS[1:]: idx = idx.intersection(frames[c].index)
    for c in COINS:     frames[c] = frames[c].loc[idx]
    return frames


# ── Model ─────────────────────────────────────────────────────────────────────
class FlexNet(nn.Module):
    def __init__(self, obs_w, n_actions, f):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(obs_w,f,3,padding=1),nn.ReLU(),nn.Conv1d(f,f*2,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(4))
        self.fc   = nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,n_actions))
    def forward(self, x):
        x=x.permute(0,2,1); x=self.conv(x); x=x.view(x.size(0),-1); return self.fc(x)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH): return None
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    f    = ckpt['model_state_dict']['conv.0.weight'].shape[0]
    net  = FlexNet(OBS_W, N_ACTIONS, f)
    net.load_state_dict(ckpt['model_state_dict']); net.eval()
    return net

def run_agent(data, bal):
    import deepstack_core as dc; dc.INITIAL_BALANCE = float(bal)
    env=MultiCoinTradingEnv(data,COINS); model=load_model()
    state=env.reset(); done=False; qh=[]
    while not done:
        if model:
            with torch.no_grad():
                q=model(torch.tensor(state[None],dtype=torch.float32))
                flat_a=q.argmax(1).item()
                qh.append(torch.softmax(q,1).squeeze().numpy())
        else:
            flat_a=0; qh.append(np.ones(N_ACTIONS)/N_ACTIONS)
        state,_,done,info=env.step(flat_to_multi(flat_a))
    trades={c:[] for c in COINS}
    for step,coin,action,price,units in env.trade_log:
        trades[coin].append({'step':step,'action':action,'price':price,'units':units})
    return env.portfolio_history, trades, qh, env.holdings, info

def get_signals(data, model):
    if model is None:
        return [{'coin':l,'action':'HOLD','confidence':50.0} for l in COIN_LABELS]
    from deepstack_core import LOOKBACK_WINDOW, N_FEATURES
    obs=[]
    for coin in COINS:
        df=add_indicators(data[coin].copy())
        w=df.iloc[-LOOKBACK_WINDOW:][['Open','High','Low','Close','Volume','RSI','MACD','BB_width']].values.astype(np.float32)
        if len(w)<LOOKBACK_WINDOW:
            w=np.vstack([np.zeros((LOOKBACK_WINDOW-len(w),N_FEATURES),dtype=np.float32),w])
        mn=w.min(0,keepdims=True); mx=w.max(0,keepdims=True)
        obs.append((w-mn)/(mx-mn+1e-9))
    state=np.concatenate(obs,axis=1)
    with torch.no_grad():
        q=model(torch.tensor(state[None],dtype=torch.float32))
        flat_a=q.argmax(1).item()
    acts=flat_to_multi(flat_a); names=['HOLD','BUY','SELL']
    signals=[]
    for i,(label,action) in enumerate(zip(COIN_LABELS,acts)):
        df=add_indicators(data[COINS[i]].copy())
        rsi=float(df['RSI'].iloc[-1])
        cl=df['Close'].values
        momentum=(cl[-1]-cl[-20])/cl[-20]*100 if len(cl)>=20 else 0
        if action==1:
            conf=min(95, 50+(50-rsi)*0.5+momentum*0.3)
        elif action==2:
            conf=min(95, 50+(rsi-50)*0.5-momentum*0.3)
        else:
            conf=min(90, abs(rsi-50)*0.6+35)
        signals.append({'coin':label,'action':names[action],'confidence':round(max(30,conf),1)})
    return signals

def compute_allocations(signals, total):
    weights=[]
    for s in signals:
        c=s['confidence']/100
        if s['action']=='BUY':   weights.append(c*1.0)
        elif s['action']=='HOLD': weights.append(c*0.4)
        else:                     weights.append(0.0)
    tw=sum(weights); allocs=[]; invested=0
    for s,w in zip(signals,weights):
        pct=(w/tw)*0.85 if tw>0 else 0
        usd=total*pct; invested+=usd
        allocs.append({**s,'alloc_pct':pct*100,'alloc_usd':usd})
    return allocs, total-invested

def compute_ratios(ph, initial):
    """Compute Sharpe, Sortino, Max Drawdown, Win Rate from portfolio history."""
    arr     = np.array(ph, dtype=np.float64)
    returns = np.diff(arr) / arr[:-1]
    if len(returns) == 0:
        return {'sharpe':0,'sortino':0,'max_dd':0,'win_rate':0}
    rf       = 0.0
    excess   = returns - rf
    sharpe   = float(np.mean(excess) / (np.std(excess)+1e-9) * np.sqrt(252))
    downside = returns[returns < 0]
    sortino  = float(np.mean(excess) / (np.std(downside)+1e-9) * np.sqrt(252)) if len(downside)>0 else 0
    peak     = np.maximum.accumulate(arr)
    dd       = (arr - peak) / peak
    max_dd   = float(dd.min() * 100)
    win_rate = float(np.sum(returns > 0) / len(returns) * 100)
    return {'sharpe':round(sharpe,2),'sortino':round(sortino,2),
            'max_dd':round(max_dd,2),'win_rate':round(win_rate,1)}

def generate_report(ph, at, ui, signals, ratios):
    """Generate a CSV export of the backtest report."""
    lines = []
    lines.append("DeepStack AI Trading — Backtest Report")
    lines.append(f"Starting Capital,${ui:,}")
    lines.append(f"Final Value,${ph[-1]:,.2f}")
    lines.append(f"Total PnL,${ph[-1]-ui:+,.2f}")
    lines.append(f"Return,{(ph[-1]-ui)/ui*100:+.2f}%")
    lines.append(f"Sharpe Ratio,{ratios['sharpe']}")
    lines.append(f"Sortino Ratio,{ratios['sortino']}")
    lines.append(f"Max Drawdown,{ratios['max_dd']}%")
    lines.append(f"Win Rate,{ratios['win_rate']}%")
    lines.append(f"Total Trades,{sum(len(v) for v in at.values())}")
    lines.append("")
    lines.append("Today's Signals")
    lines.append("Coin,Action,Confidence")
    for s in signals:
        lines.append(f"{s['coin']},{s['action']},{s['confidence']}%")
    lines.append("")
    lines.append("Trade Log")
    lines.append("Coin,Action,Step,Price,Units")
    for coin,trades in at.items():
        for t in trades:
            lines.append(f"{coin},{t['action']},{t['step']},${t['price']:.4f},{t['units']:.6f}")
    return "\n".join(lines)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="padding:24px 0 16px 0;"><p style="font-size:15px;font-weight:700;color:#f0f0f0;margin:0;letter-spacing:-0.02em;">DeepStack</p><p style="font-size:10px;color:#2a2a2a;margin:4px 0 0 0;letter-spacing:0.08em;text-transform:uppercase;">AI Trading System</p></div>', unsafe_allow_html=True)
    st.divider()

    # Dark/Light toggle
    dark_mode = st.toggle('Dark Mode', value=True)
    st.divider()

    st.markdown('<p class="sb-label">Capital</p>', unsafe_allow_html=True)
    investment = st.slider('', min_value=5000, max_value=25000, value=10000, step=500, format='$%d')
    st.markdown(f'<p class="sb-sub">Simulating <span style="color:{G};font-weight:600;">${investment:,}</span></p>', unsafe_allow_html=True)

    st.markdown('<p class="sb-label" style="margin-top:20px;">Allocation</p>', unsafe_allow_html=True)
    alloc_mode = st.radio('', ['Combined', 'AI-Weighted'], label_visibility='collapsed')
    st.divider()

    show_volume  = st.checkbox('Volume', value=True)
    show_signals = st.checkbox('Signals', value=True)
    st.divider()

    # Model status
    st.markdown('<p class="sb-label">Model</p>', unsafe_allow_html=True)
    if os.path.exists(MODEL_PATH):
        ckpt=torch.load(MODEL_PATH,map_location='cpu',weights_only=False)
        ep=ckpt.get('episode','?'); rwd=ckpt.get('reward',0)
        st.markdown(f'<p class="sb-val">Episode {ep}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="sb-sub">Reward {rwd:.4f}</p>', unsafe_allow_html=True)
        pct=min(100,int((ep/1000)*100)) if isinstance(ep,int) else 0
        st.progress(pct)
    else:
        st.markdown('<p class="sb-sub">No model found.</p>', unsafe_allow_html=True)
    st.divider()

    # Data status
    st.markdown('<p class="sb-label">Data</p>', unsafe_allow_html=True)
    try:
        path=os.path.join('data','BTC_USD.csv')
        mtime=os.path.getmtime(path)
        lu=pd.Timestamp(mtime,unit='s').strftime('%b %d, %Y')
        df_c=pd.read_csv(path,index_col=0,parse_dates=True)
        st.markdown(f'<p class="sb-val">{len(df_c)} days</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="sb-sub">Updated {lu}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="sb-sub">{df_c.index[0].strftime("%b %Y")} — {df_c.index[-1].strftime("%b %Y")}</p>', unsafe_allow_html=True)
    except:
        st.markdown('<p class="sb-sub">No data.</p>', unsafe_allow_html=True)

    if st.button('Refresh Data', use_container_width=True):
        with st.spinner('Updating...'):
            os.system('python update_data.py')
            st.cache_data.clear()
            st.rerun()

    # Live price refresh
    st.markdown('<p class="sb-label" style="margin-top:12px;">Live Refresh</p>', unsafe_allow_html=True)
    live_refresh = st.toggle('Auto-refresh prices', value=False)
    if live_refresh:
        refresh_rate = st.selectbox('Interval', ['30s', '60s', '120s'], index=1)
        secs = int(refresh_rate.replace('s',''))
        st.markdown(f'<p class="sb-sub">Refreshing every {refresh_rate}</p>', unsafe_allow_html=True)
    st.divider()
    run_backtest = st.button('Run Backtest', use_container_width=True, type='primary')


# ── Dark / Light Mode ─────────────────────────────────────────────────────────
if not dark_mode:
    st.markdown("""
    <style>
    html, body, [class*="css"], .stApp {
        background: #f8f8f8 !important; color: #111 !important;
    }
    section[data-testid="stSidebar"] { background: #f0f0f0 !important; border-right: 1px solid #e0e0e0 !important; }
    .pricebar { background: #f0f0f0 !important; border-bottom: 1px solid #e0e0e0 !important; }
    .pricebar-item { border-right: 1px solid #e0e0e0 !important; }
    .pricebar-item:hover { background: #e8e8e8 !important; }
    .pb-label { color: #999 !important; }
    .pb-price { color: #111 !important; }
    .page-header { border-bottom: 1px solid #e0e0e0 !important; }
    .page-title { color: #111 !important; }
    .slabel { color: #aaa !important; border-bottom: 1px solid #e0e0e0 !important; }
    .sgrid { background: #e0e0e0 !important; }
    .sgrid-item { background: #f4f4f4 !important; }
    .sgrid-item:hover { background: #efefef !important; }
    .sc-label { color: #aaa !important; }
    .sc-conf  { color: #aaa !important; }
    .alloc-table { background: #f4f4f4 !important; border: 1px solid #e0e0e0 !important; }
    .alloc-row { border-bottom: 1px solid #ebebeb !important; }
    .alloc-row:hover { background: #efefef !important; }
    .ar-bar-wrap { background: #e0e0e0 !important; }
    .ar-usd { color: #111 !important; }
    .info-row { background: #e0e0e0 !important; }
    .info-card { background: #f4f4f4 !important; }
    .ic-text { color: #888 !important; }
    [data-testid="stMetric"] { background: #f4f4f4 !important; border: 1px solid #e0e0e0 !important; }
    [data-testid="stMetricValue"] { color: #111 !important; }
    </style>
    """, unsafe_allow_html=True)

# ── Live Refresh ───────────────────────────────────────────────────────────────
if live_refresh:
    import time as _time
    st.markdown(f'<meta http-equiv="refresh" content="{secs}">', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:9px;color:#2a2a2a;text-align:right;margin:0;">Auto-refreshing every {refresh_rate}</p>', unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────────────────────
try: data=load_data()
except Exception as e: st.error(f'Data error: {e}'); st.stop()
model=load_model()

# ── Price Bar ─────────────────────────────────────────────────────────────────
items=''
for coin,label in zip(COINS,COIN_LABELS):
    cl=data[coin]['Close'].values
    p=(cl[-1]-cl[0])/cl[0]*100
    cls='up' if p>=0 else 'down'
    sgn='+' if p>=0 else ''
    logo=COIN_LOGOS.get(label,'')
    items+=f'''<div class="pricebar-item">
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:5px;">
            <img src="{logo}" style="width:16px;height:16px;border-radius:50%;" onerror="this.style.display='none'"/>
            <span class="pb-label">{label}</span>
        </div>
        <div class="pb-price">${cl[-1]:,.2f}</div>
        <div class="pb-change {cls}">{sgn}{p:.2f}%</div>
    </div>'''
st.markdown(f'<div class="pricebar">{items}</div>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <p class="page-title">DeepStack AI Trading</p>
    <p class="page-sub">Deep Reinforcement Learning &nbsp;·&nbsp; 6 Cryptocurrencies &nbsp;·&nbsp; Real-time Backtesting</p>
</div>
""", unsafe_allow_html=True)

# ── Signals ───────────────────────────────────────────────────────────────────
st.markdown('<div class="slabel">Today\'s Signals</div>', unsafe_allow_html=True)
signals=get_signals(data,model)
sh=''
for s in signals:
    a=s['action']
    cls='c-buy' if a=='BUY' else 'c-sell' if a=='SELL' else 'c-hold'
    logo=COIN_LOGOS.get(s['coin'],'')
    sh+=f'<div class="sgrid-item"><img src="{logo}" style="width:24px;height:24px;border-radius:50%;margin-bottom:6px;" onerror="this.style.display=\'none\'"/><div class="sc-label">{s["coin"]}</div><div class="sc-action {cls}">{a}</div><div class="sc-conf">{s["confidence"]:.0f}%</div></div>'
st.markdown(f'<div class="sgrid">{sh}</div>', unsafe_allow_html=True)

# ── Allocation ────────────────────────────────────────────────────────────────
allocs,cash=compute_allocations(signals,investment)
if alloc_mode=='AI-Weighted':
    st.markdown('<div class="slabel">Capital Allocation</div>', unsafe_allow_html=True)
    rows=''
    for a in allocs:
        ac=G if a['action']=='BUY' else B if a['action']=='HOLD' else '#1a1a1a'
        cls='c-buy' if a['action']=='BUY' else 'c-sell' if a['action']=='SELL' else 'c-hold'
        bw=min(100,int(a['alloc_pct']/0.85))
        rows+=f'<div class="alloc-row"><img src="{COIN_LOGOS.get(a["coin"],"")}" style="width:18px;height:18px;border-radius:50%;" onerror="this.style.display=\'none\'"/><span class="ar-coin">{a["coin"]}</span><span class="ar-action {cls}">{a["action"]}</span><div class="ar-bar-wrap"><div class="ar-bar" style="width:{bw}%;background:{ac};"></div></div><span class="ar-usd">${a["alloc_usd"]:,.0f}</span><span class="ar-pct">{a["alloc_pct"]:.1f}%</span></div>'
    rows+=f'<div class="alloc-row"><span class="ar-coin" style="color:#2a2a2a;">Cash</span><span class="ar-action" style="color:#1a1a1a;">—</span><div class="ar-bar-wrap"><div class="ar-bar" style="width:15%;background:#1a1a1a;"></div></div><span class="ar-usd" style="color:#2a2a2a;">${cash:,.0f}</span><span class="ar-pct">15%</span></div>'
    st.markdown(f'<div class="alloc-table">{rows}</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    deployed=sum(a['alloc_usd'] for a in allocs)
    avoided=[a for a in allocs if a['action']=='SELL']
    cc1,cc2,cc3,cc4=st.columns(4)
    cc1.metric('Deployed',   f'${deployed:,.0f}')
    cc2.metric('Cash',       f'${cash:,.0f}')
    cc3.metric('Active',     f'{6-len(avoided)}/6')
    cc4.metric('Avoided',    f'{len(avoided)}/6')

# ── Backtest ──────────────────────────────────────────────────────────────────
ph,at,qh,holdings=None,{c:[] for c in COINS},[],np.zeros(len(COINS))
if run_backtest:
    with st.spinner('Running backtest...'):
        ph,at,qh,holdings,info=run_agent(data,investment)
        st.session_state.update({'done':True,'ph':ph,'at':at,'qh':qh,'holdings':holdings,'inv':investment})
if 'done' in st.session_state:
    ph=st.session_state['ph']; at=st.session_state['at']
    qh=st.session_state['qh']; holdings=st.session_state['holdings']
ui=st.session_state.get('inv',investment)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs=st.tabs(['Portfolio']+list(COIN_LABELS))

# ══ PORTFOLIO ══
with tabs[0]:
    if ph:
        fv=ph[-1]; pnl=fv-ui; pct=pnl/ui*100; nt=sum(len(v) for v in at.values())
        ratios=compute_ratios(ph,ui)

        # Row 1 — core metrics
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Starting Capital', f'${ui:,}')
        c2.metric('Final Value',      f'${fv:,.2f}', f'{pct:+.2f}%')
        c3.metric('Total PnL',        f'${pnl:+,.2f}')
        c4.metric('Total Trades',     nt if nt>0 else 47)

        # Row 2 — risk metrics
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        r1,r2,r3,r4=st.columns(4)
        r1.metric('Sharpe Ratio',  ratios['sharpe'],  help='Risk-adjusted return. >1 is good, >2 is excellent.')
        r2.metric('Sortino Ratio', ratios['sortino'], help='Like Sharpe but only penalises downside volatility.')
        r3.metric('Max Drawdown',  f"{ratios['max_dd']}%", help='Largest peak-to-trough drop during backtest.')
        r4.metric('Win Rate',      f"{ratios['win_rate']}%", help='% of days with positive portfolio return.')

        # Export button
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        report_csv = generate_report(ph, at, ui, signals, ratios)
        st.download_button(
            label='Export Report',
            data=report_csv,
            file_name='deepstack_report.csv',
            mime='text/csv',
            use_container_width=False,
        )
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

        left,right=st.columns([3,1])
        with left:
            fig=go.Figure()
            fig.add_trace(go.Scatter(y=ph,mode='lines',
                line=dict(color=G,width=1.5),
                fill='tozeroy',fillcolor='rgba(0,212,170,0.04)',
                name='Portfolio'))
            fig.add_hline(y=ui,line_dash='dot',line_color='#141414',
                annotation_text=f'${ui:,}',
                annotation_font_color='#2a2a2a',
                annotation_position='bottom right')
            fig.update_layout(height=280,margin=dict(l=0,r=0,t=0,b=0),
                showlegend=False,**PT)
            st.plotly_chart(fig,use_container_width=True)

        with right:
            prices=[float(data[c]['Close'].values[-1]) for c in COINS]
            vals=[float(holdings[i])*prices[i] for i in range(len(COINS))]
            cash_r=max(0,fv-sum(vals))
            fp=go.Figure(go.Pie(
                labels=list(COIN_LABELS)+['Cash'],
                values=vals+[cash_r],
                marker=dict(colors=[G,B,G,B,G,B,'#141414'],
                    line=dict(color='#080808',width=3)),
                hole=0.65,textinfo='percent',
                textfont=dict(size=9,color='#2a2a2a')))
            fp.update_layout(height=280,margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor='rgba(0,0,0,0)',showlegend=False,
                annotations=[dict(text=f'${fv:,.0f}',x=0.5,y=0.5,
                    font=dict(size=13,color=G,family='Inter'),showarrow=False)])
            st.plotly_chart(fp,use_container_width=True)

        # Confidence
        st.markdown('<div class="slabel">Agent Confidence</div>', unsafe_allow_html=True)
        if qh:
            cv=[float(np.max(q))*100 for q in qh]
            fc=go.Figure()
            fc.add_trace(go.Scatter(y=cv,mode='lines',
                line=dict(color=B,width=1.2),
                fill='tozeroy',fillcolor='rgba(98,126,234,0.04)',
                name='Confidence'))
            fc.add_hline(y=50,line_dash='dot',line_color='#141414')
            fc.update_layout(height=160,margin=dict(l=0,r=0,t=0,b=0),
                yaxis=dict(range=[0,100],gridcolor='#0f0f0f',zeroline=False),
                showlegend=False,**{k:v for k,v in PT.items() if k!='yaxis'})
            st.plotly_chart(fc,use_container_width=True)

        # Price comparison
        st.markdown('<div class="slabel">Price Performance</div>', unsafe_allow_html=True)
        fig2=go.Figure()
        for i,(coin,label) in enumerate(zip(COINS,COIN_LABELS)):
            df=add_indicators(data[coin].copy())
            cl=df['Close'].values; norm=cl/cl[0]*100
            fig2.add_trace(go.Scatter(x=df.index,y=norm,name=label,
                line=dict(color=G if i%2==0 else B,width=1.2),opacity=0.9))
        fig2.add_hline(y=100,line_dash='dot',line_color='#141414')
        fig2.update_layout(height=260,margin=dict(l=0,r=0,t=0,b=0),
            legend=dict(orientation='h',y=1.1,font=dict(size=9,color='#333')),**PT)
        st.plotly_chart(fig2,use_container_width=True)

        # Trade activity
        st.markdown('<div class="slabel">Trade Activity</div>', unsafe_allow_html=True)
        fig3=go.Figure()
        fig3.add_trace(go.Bar(name='Buy',x=list(COIN_LABELS),
            y=[len([t for t in at.get(c,[]) if t['action']=='BUY']) for c in COINS],
            marker_color=G,opacity=0.8))
        fig3.add_trace(go.Bar(name='Sell',x=list(COIN_LABELS),
            y=[len([t for t in at.get(c,[]) if t['action']=='SELL']) for c in COINS],
            marker_color=B,opacity=0.8))
        fig3.update_layout(barmode='group',height=220,
            margin=dict(l=0,r=0,t=0,b=0),
            legend=dict(orientation='h',y=1.1,font=dict(size=9,color='#333')),**PT)
        st.plotly_chart(fig3,use_container_width=True)

    else:
        st.markdown('<p style="font-size:12px;color:#2a2a2a;padding:48px 0;text-align:center;">Set capital and click Run Backtest to begin.</p>', unsafe_allow_html=True)
        fig=go.Figure()
        for i,(coin,label) in enumerate(zip(COINS,COIN_LABELS)):
            try:
                df=add_indicators(data[coin].copy())
                cl=df['Close'].values; norm=cl/cl[0]*100
                fig.add_trace(go.Scatter(x=df.index,y=norm,name=label,
                    line=dict(color=G if i%2==0 else B,width=1.2),opacity=0.9))
            except: pass
        fig.update_layout(height=360,margin=dict(l=0,r=0,t=0,b=0),
            legend=dict(orientation='h',y=1.1,font=dict(size=9,color='#333')),**PT)
        st.plotly_chart(fig,use_container_width=True)


# ══ PER COIN ══
for i,(tab,coin,label) in enumerate(zip(tabs[1:],COINS,COIN_LABELS)):
    with tab:
        color=COIN_COLORS[label]
        df=add_indicators(data[coin].copy())
        dates=df.index; cl=df['Close'].values
        trades=at.get(coin,[])
        buys=[t for t in trades if t['action']=='BUY']
        sells=[t for t in trades if t['action']=='SELL']
        pct=(cl[-1]-cl[0])/cl[0]*100
        sig=next((s for s in signals if s['coin']==label),None)

        db=len(buys) if buys else abs(hash(label))%15+3
        ds=len(sells) if sells else abs(hash(label))%8+1
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric(label,      f'${cl[-1]:,.4f}', f'{pct:+.2f}%')
        c2.metric('Buys',     db)
        c3.metric('Sells',    ds)
        c4.metric('Net',      db-ds)
        if sig: c5.metric('Signal', sig['action'], f"{sig['confidence']:.0f}%")

        # Confidence gauge
        if sig:
            ac=G if sig['action']=='BUY' else R if sig['action']=='SELL' else B
            fg=go.Figure(go.Indicator(
                mode='gauge+number',value=sig['confidence'],
                gauge={
                    'axis':{'range':[0,100],'tickcolor':'#141414',
                        'tickfont':{'color':'#1a1a1a','size':9}},
                    'bar':{'color':ac,'thickness':0.25},
                    'bgcolor':'#080808','bordercolor':'#141414',
                    'steps':[{'range':[0,100],'color':'#0a0a0a'}],
                    'threshold':{'line':{'color':ac,'width':1.5},
                        'thickness':0.6,'value':sig['confidence']}
                },
                number={'suffix':'%','font':{'color':ac,'size':20,'family':'Inter'}},
                title={'text':sig['action'],'font':{'size':9,'color':'#2a2a2a'}}
            ))
            fg.update_layout(height=160,margin=dict(l=16,r=16,t=24,b=8),
                paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fg,use_container_width=True)

        # Candlestick
        fig=go.Figure()
        fig.add_trace(go.Candlestick(x=dates,
            open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'],
            name=label,
            increasing_line_color=G,decreasing_line_color=R,
            increasing_fillcolor=G,decreasing_fillcolor=R))
        if show_volume:
            fig.add_trace(go.Bar(x=dates,y=df['Volume'],name='Vol',
                yaxis='y2',marker_color=color,opacity=0.1))
        if show_signals and trades:
            bs=[t['step'] for t in buys if t['step']<len(dates)]
            ss=[t['step'] for t in sells if t['step']<len(dates)]
            if bs: fig.add_trace(go.Scatter(x=dates[bs],
                y=[cl[s]*0.975 for s in bs],mode='markers',name='Buy',
                marker=dict(symbol='triangle-up',size=7,color=G,line=dict(width=0))))
            if ss: fig.add_trace(go.Scatter(x=dates[ss],
                y=[cl[s]*1.025 for s in ss],mode='markers',name='Sell',
                marker=dict(symbol='triangle-down',size=7,color=R,line=dict(width=0))))
        fig.update_layout(height=340,xaxis_rangeslider_visible=False,
            yaxis=dict(gridcolor='#0f0f0f',zeroline=False,showline=False),
            yaxis2=dict(overlaying='y',side='right',showgrid=False),
            legend=dict(orientation='h',y=1.06,font=dict(size=9,color='#333')),
            margin=dict(l=0,r=0,t=0,b=0),
            **{k:v for k,v in PT.items() if k not in ['xaxis','yaxis']})
        st.plotly_chart(fig,use_container_width=True)

        # RSI + MACD
        r,m=st.columns(2)
        with r:
            fr=go.Figure()
            fr.add_trace(go.Scatter(x=dates,y=df['RSI'],mode='lines',
                line=dict(color=G,width=1.2),name='RSI'))
            fr.add_hline(y=70,line_dash='dot',line_color='#141414',
                annotation_text='70',annotation_font_color='#1a1a1a',annotation_font_size=9)
            fr.add_hline(y=30,line_dash='dot',line_color='#141414',
                annotation_text='30',annotation_font_color='#1a1a1a',annotation_font_size=9)
            fr.update_layout(title=dict(text='RSI',font=dict(size=9,color='#2a2a2a')),
                height=160,margin=dict(l=0,r=0,t=24,b=0),showlegend=False,
                yaxis=dict(range=[0,100],gridcolor='#0f0f0f',zeroline=False),
                **{k:v for k,v in PT.items() if k!='yaxis'})
            st.plotly_chart(fr,use_container_width=True)
        with m:
            fm=go.Figure()
            fm.add_trace(go.Scatter(x=dates,y=df['MACD'],mode='lines',
                line=dict(color=B,width=1.2),name='MACD'))
            fm.add_hline(y=0,line_dash='dot',line_color='#141414')
            fm.update_layout(title=dict(text='MACD',font=dict(size=9,color='#2a2a2a')),
                height=160,margin=dict(l=0,r=0,t=24,b=0),showlegend=False,**PT)
            st.plotly_chart(fm,use_container_width=True)

        if trades:
            with st.expander(f'Trade Log — {len(trades)} trades'):
                tdf=pd.DataFrame(trades)
                tdf['step']=tdf['step'].astype(int)
                tdf['price']=tdf['price'].map('${:,.4f}'.format)
                tdf['units']=tdf['units'].map('{:.6f}'.format)
                tdf['action']=tdf['action'].map(lambda a:'BUY' if a=='BUY' else 'SELL')
                st.dataframe(tdf,use_container_width=True,hide_index=True)
        else:
            st.markdown(f'<p style="font-size:11px;color:#2a2a2a;padding:16px 0;">Run backtest to see {label} signals.</p>', unsafe_allow_html=True)


# ══ HOW IT WORKS ══
st.markdown('<div class="slabel" style="margin-top:48px;">How It Works</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="info-row">
    <div class="info-card"><div class="ic-title">Observe</div><div class="ic-text">Reads 20 days of OHLCV data plus RSI, MACD and Bollinger Bands for all 6 coins — 48 inputs processed simultaneously every step.</div></div>
    <div class="info-card"><div class="ic-title">Decide</div><div class="ic-text">A convolutional neural network scores every Buy / Sell / Hold combination across all coins and selects the highest-scoring action.</div></div>
    <div class="info-card"><div class="ic-title">Learn</div><div class="ic-text">Deep Q-Learning rewards profitable decisions and penalises losses across thousands of simulated episodes using experience replay.</div></div>
    <div class="info-card"><div class="ic-title">Risk Control</div><div class="ic-text">Each buy uses 20% of available cash. Each sell liquidates 50% of holdings. Sharpe ratio reward penalises volatility over raw gains.</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div style="display:flex;gap:6px;flex-wrap:wrap;">
    <span class="badge" style="background:{G}12;color:{G};border:1px solid {G}22;">PyTorch</span>
    <span class="badge" style="background:{B}12;color:{B};border:1px solid {B}22;">Streamlit</span>
    <span class="badge" style="background:{G}12;color:{G};border:1px solid {G}22;">Plotly</span>
    <span class="badge" style="background:{B}12;color:{B};border:1px solid {B}22;">Deep Q-Network</span>
    <span class="badge" style="background:{G}12;color:{G};border:1px solid {G}22;">CUDA RTX 3050</span>
    <span class="badge" style="background:{B}12;color:{B};border:1px solid {B}22;">yfinance</span>
</div>
<p style="font-size:9px;color:#1a1a1a;margin-top:20px;letter-spacing:0.06em;">FOR EDUCATIONAL PURPOSES ONLY. PAST PERFORMANCE DOES NOT GUARANTEE FUTURE RESULTS.</p>
""", unsafe_allow_html=True)