import pathlib
import pickle
import itertools
import torch
import numpy as np
import datetime
import backtrader as bt
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .price_crawler import build_features, update_all
from .rnn_utils import get_model, pd_to_tensor
import rnn_utils

from .backtrader import CustomIndicator, CustomStrategy, CustomPandasData


TARGETS = ["1D-close", "7D-close", "30D-close"]
TA_FEATURES = ['ROC', 'MOM', 'EMA', 'SMA', 'VAR', 'MACD', 'ADX', 'RSI']


def predict_trend(model, address, cfg, target, columns):
    # Get price and prediction for tokens              
    x_df, labels_df = build_features(address, cfg, ta_list=TA_FEATURES, ys=[target])

    try:
        x_df = x_df[columns]
    except KeyError as e:
        print(e)
        return 0, [0, 0, 0]

    x, labels = pd_to_tensor(x_df, labels_df)
    x = x.to(cfg["device"])
    try:
        with torch.no_grad():
            y = model(x).cpu().numpy()
    except RuntimeError as e:
        print(e)
        return 0, [0, 0, 0]

    labels = labels.numpy().astype(int)
    y_uniq, y_count = np.unique(y, return_counts=True)
    if len(y_uniq) == 1:
        y[y_count//2] = 1 - y_uniq[0]  

    best_y = None
    best_acc = 0
    best_threshold = 0
    for threshold in range(25, 75, 5):
        y_test = y > (threshold / 100)
        y_test = y_test.astype(int)[:, 0]
        acc = accuracy_score(labels, y_test)
        if acc > best_acc:
            best_acc = acc
            best_y = y_test
            best_threshold = threshold
    
    y = best_y

    f1 = f1_score(labels, y, zero_division='warn')          
    precision = precision_score(labels, y, zero_division=0)

    y_df = pd.DataFrame(y, index=x_df.index, columns=["prediction"])

    caps = [np.minimum(15, y_df.shape[0]), np.minimum(30, y_df.shape[0]), np.minimum(180, y_df.shape[0]), np.minimum(540, y_df.shape[0])]
    pnls = []
    for cap in caps:
        new_df = pd.concat([x_df[:cap], y_df[:cap]], axis=1)
        cerebro = bt.Cerebro()

        cerebro.broker = bt.brokers.BackBroker()
        cerebro.broker.setcash(1000)
        cerebro.broker.setcommission(commission=0.001)

        # Integrate with data source and custom strategies
        cerebro.adddata(data=CustomPandasData(dataname=new_df))
        cerebro.addstrategy(CustomStrategy)

        # Start simulation
        start_portfolio_value = cerebro.broker.getvalue()
        cerebro.run()

        end_portfolio_value = cerebro.broker.getvalue()

        pnls.append((end_portfolio_value - start_portfolio_value) / 1000 * 100)

    return y[0][0], [acc, f1, precision, pnls[0], pnls[1], pnls[2], pnls[3]]

def read_trend_day():
    token_results = {}
    trend_path = pathlib.Path("trend_data/" + str(datetime.date.today()) + "_trend.pkl")
    if trend_path.exists():
        with open("trend_data/" + str(datetime.date.today()) + "_trend.pkl", "rb") as f:
            token_results = pickle.load(f)
    
    return token_results

def get_trend(tokens, TREND_DATA):
    predictions_all = []
    metrics_all = []
    for t in tokens:     
        models = TREND_DATA[t]
        predictions = {}
        metrics = {}
        for model_type in models.keys():
            trend, metric = TREND_DATA[t][model_type]
            metric = np.array(metric)
            metric = np.concatenate([np.mean(metric[:, 0:3], axis=0), np.max(metric[:, 3:], axis=0)])
            predictions[model_type] = trend
            metrics[model_type] = metric
        
        predictions_all.append(predictions)
        metrics_all.append(metrics)
    
    return predictions_all, metrics_all

def get_pnl(metrics, horizon):
    metric = metrics["RNN"]
    if horizon == "week":
        pnl = metric[3]
    elif horizon == "month":
        pnl = metric[4]
    elif horizon == "season":
        pnl = metric[5]
    elif horizon == "year":
        pnl = metric[6]
    
    return pnl

def update_trend_day(tokens):
    token_results = {}
    trend_path = pathlib.Path("trend_data/" + str(datetime.date.today()) + "_trend.pkl")
    if trend_path.exists():
        with open("trend_data/" + str(datetime.date.today()) + "_trend.pkl", "rb") as f:
            token_results = pickle.load(f)
        return token_results

    update_all()
    i = 0
    print("Updating trend results")
    for index, token in tokens.iterrows():
        print(index)
        address = token['Address']
        coin_id = index
        all_models_exist = True

        for target, model_type in itertools.product(TARGETS, ["RNN", "GRU", "LSTM"]):
            model_path = pathlib.Path("models/" + coin_id + "-" + target + "-" + model_type + ".pkl")
            if not model_path.exists():
                all_models_exist = False

        if not all_models_exist:
            continue

        token_results[coin_id] = {}
        
        for model_type in ["RNN", "GRU", "LSTM"]:
            trends = []
            all_metrics = []
            for target in TARGETS: 
                model_path = pathlib.Path("models/" + coin_id + "-" + target + "-" + model_type + ".pkl")
                with open(model_path, 'rb') as f:
                    model, cfg, columns = torch.load(f, map_location='cpu', pickle_module=pickle)

                    if torch.cuda.is_available():
                        device = cfg["device"]
                    else:
                        device = 'cpu'

                    cfg["device"] = device
                    model.device = device
                    model = model.to(device)
                    [m.flatten_parameters() for m in model.rnn]
                    model.rnn = [m.to(device) for m in model.rnn]

                    trend, metrics = predict_trend(model, coin_id, cfg, target, columns)
                    trends.append(trend)
                    all_metrics.append(metrics)
                    i += 1
        
            token_results[coin_id][model_type] = [trends, all_metrics]

    with open("trend_data/" + str(datetime.date.today()) + "_trend.pkl", "wb") as f:
        pickle.dump(token_results, f)

    return token_results
