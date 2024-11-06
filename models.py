import pandas as pd

TA_FEATURES = ['ROC', 'MOM', 'EMA', 'SMA', 'VAR', 'MACD', 'ADX', 'RSI']

if __name__ == "__main__":
    tokens_file='data/tokens.csv'

    tokens = pd.read_csv(tokens_file)
    for symbol, name in zip(tokens['CoinGeckoID'], tokens['Name']):
        for target in ['1D-close', '3D-close', '7D-close', '15D-close', '30D-close']:
            x, labels = build_features(symbol, cfg, ta_list=TA_FEATURES, ys=[target])