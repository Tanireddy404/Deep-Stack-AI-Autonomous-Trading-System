import pandas as pd, os
def fetch_data_csv(coins, data_dir='data'):
    frames={}
    for coin in coins:
        fname=coin.replace('-','_')+'.csv'
        df=pd.read_csv(os.path.join(data_dir,fname),index_col=0,parse_dates=True)
        frames[coin]=df[['Open','High','Low','Close','Volume']].dropna()
    idx=frames[coins[0]].index
    for c in coins[1:]: idx=idx.intersection(frames[c].index)
    for c in coins: frames[c]=frames[c].loc[idx]
    return frames
