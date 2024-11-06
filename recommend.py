import numpy as np
import pickle as pk
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import os
import shap
import torch

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .crawler import get_address_transactions
from .algorithm import UserKNN, ReLU
from .price_crawler import build_features, cg
from .rnn_utils import get_model, pd_to_tensor
import sys

API_KEY = 'UCJ24GP9ICCR28QNPDNCXZ27VHWIG442F6'
TA_FEATURES = ['ROC', 'MOM', 'EMA', 'SMA', 'VAR', 'MACD', 'ADX', 'RSI']
NO_RECOMMEND = 5
TARGET = ['1D-close', '7D-close', '30D-close']
NO_TOKEN = 30


class Profile:

    def __init__(self, address=None):
        self.address = address
        self.status = 'Ok'
        self.knn_scores = None
        self.survey_scores = None


def get_price(symbol):
    r = cg.get_price(symbol, 'usd')
    try:
        r = r[symbol]['usd']
    except KeyError:
        r = 0
    return str(round(r, 2))


def cf_recommend(uid, address, profiles, data, tokens):

    # Get transactions record of given address
    address = address.lower()
    transactions = get_address_transactions(address, API_KEY)

    # Create the server side profile for the user
    p = Profile(address)
    profiles[uid] = p

    # Error occur while getting transaction records
    if isinstance(transactions, str):
        p.status = transactions
        return

    # Create the vector representation of transaction records
    transactions = transactions[transactions['to'] == address]
    transactions['contractAddress'] = transactions['contractAddress'].str.lower()
    bought = transactions['contractAddress'].value_counts()
    vec = np.array([bought.loc[t] if t in bought.index else 0 for t in tokens['Address']])

    # Evaluate and store results from UserKNN
    output = UserKNN(data, vec, tokens.index)

    # Handle the case where there is an error in the data
    if isinstance(output, str):
        p.status = output
    else:
        p.knn_scores = output


def survey_recommend(uid, profiles, token_attribute, form):

    if uid in profiles:
        p = profiles[uid]
    else:
        profiles[uid] = Profile()
        p = profiles[uid]

    # Extract the usrvey input from HTML
    tolerance = float(form['risk'])
    amount = float(form['amount'])
    horizon = form['horizon']

    scores = {}
    max_score = 0
    for t in token_attribute.index:
        mean_return = token_attribute.loc[t, horizon + '_m']
        es = token_attribute.loc[t, horizon + '_es']
        score = mean_return - ReLU(tolerance - es) * amount

        scores[t] = score 
        max_score = score if score > max_score else max_score

    for t in scores.keys():
        scores[t] = scores[t] * 100 / (max_score + 0.0001)

    print("Done survey!")
    p.survey_scores = scores


def combine_score(knn_scores, survey_scores, horizon, trend_data):
    overall_score = {}

    tokens = list(knn_scores.keys())
    # recommendation = sorted(tokens, reverse=True, key=lambda t: overall_score[t])
    for i, t in enumerate(tokens):
        try:
            # trend, metrics = predict_trend([t], None)
            trend, metrics = trend_data[t]
            metrics = np.array(metrics)
            metrics = np.concatenate([np.mean(metrics[:, 0:3], axis=0), np.max(metrics[:, 3:], axis=0)])
            
            trend = [trend]
            metrics = [metrics]
            
        except ValueError:
            trend = [[0, 0, 0]]
            metrics = [[0, 0, 0, 0, 0, 0, 0]]
        except KeyError:
            trend = [[0, 0, 0]]
            metrics = [[0, 0, 0, 0, 0, 0, 0]]

        trend = trend[0]
        metrics = metrics[0]    
        npl = 0

        if horizon == "week":
            trend[0] *= 0.75
            trend[1] *= 0.2
            trend[2] *= 0.05
            npl = metrics[3]
        elif horizon == "month":
            trend[0] *= 0.25
            trend[1] *= 0.7
            trend[2] *= 0.05
            npl = metrics[4]
        elif horizon == "season":
            trend[0] *= 0.1
            trend[1] *= 0.4
            trend[2] *= 0.5
            npl = metrics[5]
        elif horizon == "year":
            trend[0] *= 0.05
            trend[1] *= 0.1
            trend[2] *= 0.85
            npl = metrics[6]
        
        # print(metrics)
        acc_mult = metrics[0]**2 if metrics[0] > 0.7 else metrics[0]**4
        
        overall_score[t] = 0.6 * knn_scores[t] * acc_mult + 0.4 * survey_scores[t]
        overall_score[t] = overall_score[t] * (sum(trend) * 0.999 + 0.001)
        overall_score[t] = overall_score[t] * 0.1 + overall_score[t] * 0.9 * npl # Profit score

    recommendation = sorted(tokens, reverse=True, key=lambda t: overall_score[t])[:NO_RECOMMEND]
    
    return recommendation, overall_score


def predict_trend(recommendations, tokens):
    # Get price and prediction for tokens
    predictions = []
    metrics = []

    for symbol in recommendations:
        token_predict = []
        if os.path.exists('models/' + symbol + '-1D-close.pkl'):
            acc =0
            f1 = 0
            precision = 0
            recall = 0
            for target in ['1D-close', '7D-close', '30D-close']:
                f = open('models/' + symbol +'-'+ target + '.pkl', 'rb')
                
                model, cfg, columns = pk.load(f)
                model = model.to(cfg["device"])
                
                x, labels = build_features(symbol, cfg, ta_list=TA_FEATURES, ys=[target], rang=0.2)
  
                x = x[columns]
                # exit(0)
                x, labels = pd_to_tensor(x, labels)
                x = x.to(cfg["device"])
                with torch.no_grad():
                    y = model(x).cpu().numpy() > 0.5
                    y = y.astype(int)[:, :, 0]

                labels = labels.numpy().astype(int)
                y_uniq, y_count = np.unique(y, return_counts=True)
                if len(y_uniq) == 1:
                    y[y_count//2] = 1 - y_uniq[0]  

                acc += accuracy_score(labels, y)
                f1 += f1_score(labels, y, zero_division='warn')          
                precision += precision_score(labels, y, zero_division=0)
                recall += recall_score(labels, y, zero_division=0)
                token_predict.append(int(y[-1]))
                predictions.append(token_predict)
            
            metrics.append([acc / 3, f1 / 3, precision / 3])
        else:
            predictions.append([0, 0, 1])
            metrics.append([0, 0, 0])

    return predictions, metrics
