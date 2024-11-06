from random import seed
from copy import copy
from os import path
import warnings
warnings.filterwarnings("ignore")

import csv
import requests
import pickle
import pandas as pd
import itertools
import numpy as np
from progress.bar import Bar
import pathlib

import datetime

import torch
from torch import nn

from .price_crawler import build_features
from .rnn_utils import get_model, pd_to_tensor


# [<Prediction Interval>-<Feature Name>]
TARGET = ['1D-close', '7D-close', '30D-close']

# Number of trials
TRIAL = 10

# TA features to be used
TA_FEATURES = ['ROC', 'MOM', 'EMA', 'SMA', 'VAR', 'MACD', 'ADX', 'RSI']

SIZE_RANGE = [64, 96]                     # Size of hidden layers within the model
DELTA_RANGE = ['24H', '12H', '6H', '3H']      # Possible duration of each period
MODEL_RANGE = ['GRU', 'LSTM', 'RNN']          # Possible Models
NORMALIZE = ['MinMax', 'Normal']              # Possible data normalization method
LR_RANGE = [0.01, 0.001]                      # Possible learning rates


def generate_config(device='cpu'):
    seed()

    all_configs = itertools.product([1, 2, 3], DELTA_RANGE, MODEL_RANGE, NORMALIZE, SIZE_RANGE, LR_RANGE)
    configs = [{
        'roll': c[0],
        'delta': c[1],
        'model': c[2],
        'norm': c[3],
        'hidden': c[4],
        'lr': c[5],
        'targets': 1,
        'device': device,
        } for c in all_configs]

    # config['roll'] = 1
    # config['delta'] = choice(DELTA_RANGE)
    # config['model'] = choice(MODEL_RANGE)
    # config['norm'] = choice(NORMALIZE)
    # config['hidden'] = choice(SIZE_RANGE)
    # config['lr'] = choice(LR_RANGE)
    
    target = [copy(TARGET) for _ in configs]

    features = [['open', 'close', 'high', 'low', 'volume'] + [s.lower() for s in TA_FEATURES] for _ in configs]
    for k, feature in enumerate(features):
        feature += [j + '-r' + str(i + 1) for j in feature for i in range(configs[k]['roll'])]
    # features += [c + '-r' + str(i + 1) for i in range(config['roll']) for c in features]

    return configs, target, features


def to_csv_lines(cfg, target, train_acc, val_acc):
    lines = []
    hyper_paramters = ['model', 'delta', 'norm', 'hidden', 'lr']
    for y, t, v in zip(target, train_acc, val_acc):
        row = [y] + [str(cfg[p]) for p in hyper_paramters] + [str(t), str(v)]
        line = ','.join(row) + '\n'
        lines.append(line)
    return lines


def load_data(address, target, cfg):
    # print(len(target))
    X, y, columns = build_features(address, freq=cfg['delta'], ta_list=TA_FEATURES, ys=target, roll=cfg['roll'])

    # df = pd.read_pickle('price_data/archive/' + str(coin_id) + '.pkl')
    
    # X.join(y).to_pickle(root + str(key) + ".pkl")
    # y = df[target]
    # X = df[df.column.difference(target)]

    cfg['input'] = len(X.columns)
    output_models = []

    if cfg['norm'] == 'MinMax':
        X = (X - X.min()) / (X.max() - X.min())
    if cfg['norm'] == 'Normal':
        X = (X - X.mean()) / X.std()

    cut = int(len(X) * 0.8)
    X, y = pd_to_tensor(X, y, device)
    y = y.unsqueeze(-1)
    train_x, train_y = X[:cut], y[:cut]
    
    val_x, val_y = X[cut:], y[cut:]

    return train_x, train_y, val_x, val_y, columns


def train(data, cfg, target, device='cpu'):

    '''
    Input(address <str>, configuration <dict>, features <list>, target <dict>, log <file>) -> list
    This function gives a list of trained model for each time horizon using the configuration above
    '''
    train_x, train_y, val_x, val_y, _ = data
    # print(train_x.shape)
    original_x = train_x.detach().clone()

    rnn = get_model(cfg)

    optim = torch.optim.Adam(rnn.parameters(), lr=cfg['lr'])
    criterion = nn.BCELoss()
    i = 1
    overfit = 0
    previous_loss = 999
    in_long_run = False
    while True:
        try:
            optim.zero_grad()
            train_pred = rnn(train_x)
            loss = criterion(train_pred, train_y)
            loss.backward()
            optim.step()
            val_pred = rnn(val_x)
            val_loss = criterion(val_pred, val_y)
            if val_loss > previous_loss:
                overfit += 1
            else:
                overfit = 0
            if overfit > 3:
                break
            previous_loss = val_loss
        except RuntimeError:
            # print(original_x.shape)
            # break
            return None, None, None, None

        i += 1
        if i > 4000:
            break
        if i % 1000 == 0:
            print('long run', i % 1000)
            in_long_run = True

    train_pred = rnn(original_x)
    train_acc = ((train_pred > 0.5) == train_y).float()
    train_acc = torch.mean(train_acc, dim=0).tolist()[0]
    val_pred = rnn(val_x)
    val_acc = ((val_pred > 0.5) == val_y).float()
    val_acc = torch.mean(val_acc, dim=0).tolist()[0]
    lines = to_csv_lines(cfg, target, train_acc, val_acc)
    return rnn, lines, val_acc, in_long_run

class ValBar(Bar):
    suffix = '%(percent).1f%% - %(eta)ds - Best Val: %(best_val).2f'
    best_val = 0.0
    # @property
    # def best_val(self):
    #     return self.best_val

if __name__ == '__main__':
    device = 'cuda:0'
    # Number of trials
    # TRIAL = 5
    # tokens = pd.read_csv('tokens.csv')
    
    # addresses = set(tokens.Address)
    existing_models = [f.stem.split('-') for f in pathlib.Path('tuning_models/').iterdir()]
    existing_models = ['-'.join(f[:-2]) for f in existing_models]


    print(existing_models)
    data_files = pathlib.Path(r'C:\Users\gabri\FYP\price_data').iterdir()

    new_data_files = [(f, datetime.date.fromtimestamp(f.stat().st_mtime)) for f in data_files]
    new_data_files = [f[0] for f in new_data_files if f[1] >= datetime.date.today() - datetime.timedelta(days=1) and f[0].stem not in existing_models]

    print(new_data_files)
    
    for f in new_data_files:
    # address = '0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2'
        # if f.stem == 'huobi-btc' or f.stem == 'parsiq' or f.stem == 'tbtc':
        #     continue
        print(f.stem)
        address = f.stem
        
        file_name = f.stem + '_rnn.csv'
        if path.exists('tuning_logs/' + file_name):
            log = open('tuning_logs/' + file_name, 'a')
            # continue
        else:
            log = open('tuning_logs/' + file_name, 'w')
            log.write('TARGET, MODEL, DELTA, NORM, HIDDEN, LR, TRAIN, VAL\n')

        csv_lines = []
        cfgs, targets, featureses = generate_config(device)
        # print("Trial:", TRIAL, len(cfgs), len(targets), len(featureses))
        for t in TARGET:
            best_model = None
            best_config = None
            best_val = 0
            best_columns = None
            with ValBar(f'Training ({t})') as bar:
                for cfg, features in zip(cfgs, featureses):
                    data = load_data(address, [t], cfg)
                    columns = data[-1]
                    model, lines, val_acc, long_run = train(data, cfg, [t], device)
                    if model == None:
                        continue
                    val_acc = np.mean(val_acc)
                    # print(val_acc)
                    # csv_lines += lines
                    log.writelines(lines)
                    if long_run:
                        best_val = val_acc
                        best_model = model
                        best_config = cfg
                        bar.best_val = best_val
                        best_columns = columns
                        break
                    # print(i + 1, 'trials completed')
                    if (val_acc > best_val):
                        best_val = val_acc
                        best_model = model
                        best_config = cfg
                        bar.best_val = best_val
                        best_columns = columns
                        if val_acc >= 0.999:
                            break
                    bar.next()
                    # break 

                if best_model != None:
                    pickle.dump([best_model.to('cpu'), best_config, best_columns], open(f"tuning_models/{address}-{t}.pkl", "wb"))                  

                # except FileNotFoundError:
                #     # log.writelines(csv_lines)
                #     print('Not found', address)
                #     break
                    # if i == TRIAL - 1:
                    #     bar.next()
                    # log.close()

        # log.writelines(csv_lines)
        log.close()
        print("Best val:", best_val)
        
                    
