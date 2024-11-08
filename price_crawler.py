from copy import copy
from datetime import datetime, timedelta
from time import sleep
import pickle as pkl
from os import path
import numpy as np
import re

from pycoingecko import CoinGeckoAPI
from talib import abstract
import pandas as pd
import talib

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


cg = CoinGeckoAPI()

def get_data(address, start, end):

    '''
    Input (address <str> , start <timestamp>, end <timestamp>) -> DataFrame of price data
    This API wrapper wrap the CoinGecko API
    To get hourly data start and end must be within 90 days
    '''

    raw = cg.get_coin_market_chart_range_from_contract_address_by_id(id='ethereum', contract_address=address, vs_currency='usd', from_timestamp=start, to_timestamp=end)
    cleaned = {}
    cleaned['timestamp'] = [i[0] for i in raw['prices']]
    cleaned['prices'] = [i[1] for i in raw['prices']]
    cleaned['market_caps'] = [i[1] for i in raw['market_caps']]
    cleaned['total_volumes'] = [i[1] for i in raw['total_volumes']]
    return pd.DataFrame.from_dict(cleaned)


def get_data_by_id(ID, start, end):

    '''
    Input (ID <str> , start <timestamp>, end <timestamp>) -> DataFrame of price data
    This API wrapper wrap the CoinGecko API
    To get hourly data start and end must be within 90 days
    '''
    raw = cg.get_coin_market_chart_range_by_id(id=ID, vs_currency='usd', from_timestamp=start, to_timestamp=end)
    cleaned = {}
    cleaned['timestamp'] = [i[0] for i in raw['prices']]
    cleaned['prices'] = [i[1] for i in raw['prices']]
    cleaned['market_caps'] = [i[1] for i in raw['market_caps']]
    cleaned['total_volumes'] = [i[1] for i in raw['total_volumes']]
    return pd.DataFrame.from_dict(cleaned)


def update_token(symbol, start_date=datetime(2019, 1, 1), method='id'):

    '''
    Input(address <str>, start_date <datetime>, method <str>) -> None
    Where method can be address or id (coingecko ID)

    Update the dataframe of the specified address within price_data directory
    If dataframe not exist crawl from start date and create a new one
    '''

    get_data = get_data_by_id

    file_dir = 'price_data/' + symbol + '.pkl'
    now = datetime.now()

    if path.exists(file_dir):
        with open(file_dir, 'rb') as f:
            data, update_time = pkl.load(f)
            data = [data]
            if now - update_time < timedelta(hours=6):
                print(f'{symbol} Cached')
                return
        start = update_time
        end = update_time + timedelta(days=80)
    else:
        start = start_date
        end = start_date + timedelta(days=80)
        data = []
    
    while now > end:
        st = datetime.timestamp(start)
        et = datetime.timestamp(end)
        data.append(get_data(symbol, st, et)) # Require time to be in timestamp format
        start = start + timedelta(days=80)
        end = end + timedelta(days=80)
        sleep(1)                              # To avoid getting blocked
    st = datetime.timestamp(start)
    et = datetime.timestamp(now)
    data.append(get_data(symbol, st, et))
    df = pd.concat(data, ignore_index=True)
    print(f'{symbol} crawled with {len(df)} timestamps')
    with open(file_dir, 'wb') as f:
        pkl.dump((df, now), f)                # Save tuple (DataFrame, Update_Time) to pkl


def update_all(tokens_file='data/tokens.csv'):

    tokens = pd.read_csv(tokens_file)
    for symbol, name in zip(tokens['CoinGeckoID'], tokens['Name']):
        try:
            update_token(symbol, method='ID')
        except:
            print(f'Cannot get price of {name}')


def update_loop(tokens_file='data/tokens.csv'):
    update_all(tokens_file)
    sleep(60 * 60 * 12)


def get_period_and_frequency(string):
    '''
    Get period and frequency from a string. Eg: '3min' -> (3, 'min'); '7D' -> (7, 'D'), and then outputs are passed into the pd.DataFrame.shift().
    Input:
        string <str>: A string that contains information about the period and the frequency.
    Return:
        period <int>: An integer indicating the period length specified in string.
        freq <str>: A string indicating the frequency specified in string.
    '''
    period = None
    freq = None
    try:
        period = int(re.search('^[0-9]+', string).group())
        freq = re.search('[A-z]+$', string).group()
    except:
        raise ValueError(f'Error in parsing periods and freq! Received periods = {period}, freq = {freq}. The string should be in the format of [a number][an English word representing a interval]. For more details, please visit https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.')
    return period, freq


def build_ta_features(df, freq='1D', ta_list='all'):
    '''
    To build OHLCV (Open High Low Close Volume) features, and advanced features provided by TA-Lib.
    Input:
        df <pd.DataFrame>: A DataFrame object that contains columns ['timestamp', 'prices', 'market_caps', 'total_volumes']. It should be retreived from CoinGecko.
        freq <str>: A string that specifies the frequency of sampling. For different frequency notations, check https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
                    If set to None, it will not perform resampling.
        ta_list <list>: A list of techical analysis features to be built. For the whole list of feature names, check TA-Lib website: https://mrjbq7.github.io/ta-lib/funcs.html.
                        If ta_list is set to 'all', it will build all features available in TA-Lib. Otherwise, it will only build features as specified in the feature_list. If ta_list is set to None, it will not build any extra features in TA-Lib. It could be useful when you only want to resample it but not building any extra technical analysis features.
    Return:
        df <pd.DataFrame>: A DataFrame object where each column represents a unique feature (with name), and each row represents a sampled timestamp (with time).
    '''
    df = df.copy()
    if freq is not None:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime')

        # Building Open High Low Close Volume features for different frequencies
        ohlc_df = df.resample(freq)['prices'].agg(['first', 'max', 'min', 'last'])
        v_df = df.resample(freq)['total_volumes'].sum()
        df = pd.merge(ohlc_df, v_df, on='datetime')
        df.columns = ['open', 'high', 'low', 'close', 'volume']
    else:
        # First fill missing close price of the resampled df with previous close price
        df['close'] = df['close'].ffill()
        df['volume'] = df['volume'].fillna(0.0)
        
        # Then fill missing open high low price of the resampled df with the previous close price (copied to the same row)
        df = pd.concat([df['open'].fillna(df['close']),
                        df['high'].fillna(df['close']),
                        df['low'].fillna(df['close']),
                        df['close'],
                        df['volume']], axis=1)

    # Building advanced features using TA-Lib
    # For MAVP, it allows moving average with variable periods specified, we'll just skip this function.

    df = df.replace(np.inf, np.nan).dropna(how='all', axis=1).dropna(how='any', axis=0)
    if ta_list is not None:
        if ta_list == 'all':
            ta_list = talib.get_functions()
        for i, x in enumerate(ta_list):
            try:
                output = eval('abstract.' + x + '(df)')
                output.name = x.lower() if type(output) == pd.core.series.Series else None
                df = pd.merge(df, pd.DataFrame(output), left_on = df.index, right_on = output.index)
                df = df.set_index('key_0')
            except:
                pass
        df.index.names = ['datetime']

    return df


def build_features(symbol, cfg, ta_list=None, ys=None, rang=None):

    '''
    To build features for the target address.
    Input(address <str>) -> DataFrame
    Read the craweled Dataframe of the specified address and add feature columns to it
    '''

    file_dir = 'price_data/' + symbol + '.pkl'
    with open(file_dir, 'rb') as f:
        df, update_time = pkl.load(f)

    x_df = build_ta_features(df, freq=cfg['delta'], ta_list=ta_list)
    
    # Adding rolling features
    columns = copy(x_df.columns)
    for i in cfg['roll']:
        for c in columns:
            x_df[c + '-r' + str(i)] = x_df[c].shift(i)
    
    if cfg['norm'] == 'MinMax':
        x_df = (x_df - x_df.min()) / (x_df.max() - x_df.min())
    if cfg['norm'] == 'Normal':
        x_df = (x_df - x_df.mean()) / x_df.std()

    if ys is None:
        return x_df[-2:-1]
    else:
        if rang != None:
            x_df = x_df[:int(rang*x_df.shape[0])]
        # Adding labels
        y_df = x_df.copy()
        for target in ys:
            interval, feature = target.split('-')
            p, f = get_period_and_frequency(interval)
            y_df[target] = (x_df[feature].shift(periods=-p, freq=f) - x_df[feature]).dropna() > 0
        y_df = y_df.drop(columns=x_df.columns)  # Features in x_df are dropped

        # Dropping all the np.inf and np.nan values in dataframes
        x_df = x_df.replace(np.inf, np.nan).dropna(how='all', axis=1).dropna(how='any', axis=0)
        y_df = y_df.replace(np.inf, np.nan).dropna(how='all', axis=1).dropna(how='any', axis=0)

        # Truncating series to make sure their indices are the same
        idx1 = y_df.index
        idx2 = x_df.index
        intersect = idx1.intersection(idx2)
        x_df = x_df.loc[intersect]
        y_df = y_df.loc[intersect]

        # Converting it to pd.Series if there is only one prediction target column
        if len(y_df.columns) == 1:
            y_df = y_df.iloc[:, 0] # Convert it to pd.Series

    return x_df, y_df


if __name__ == '__main__':
    update_all()
