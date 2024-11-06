import pickle as pk
import json
from threading import Thread

import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from recommender_module.price_crawler import update_loop
from recommender_module.recommend import *
from recommender_module.trend import update_trend_day, read_trend_day, get_trend, get_pnl
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Load user transaction data 
with open('data/user_array.pkl', 'rb') as usr:
    USER_ARRAY = pk.load(usr)

AUTO_UPDATE = False

TOKENS = pd.read_csv('data/tokens.csv')
TOKENS['Address'] = TOKENS['Address'].str.lower()
TOKENS.set_index('CoinGeckoID', inplace=True)

for idx, row in TOKENS.iterrows():
    row['Description'] = "" if row['Description'] == np.nan else row['Description']

ATTRIBUTES = pd.read_csv('data/tokens_attribute.csv')
ATTRIBUTES.set_index('CoinGeckoID', inplace=True)

WORKERS = {}
PROFILES = {}

update_trend_day(TOKENS)

scheduler = BackgroundScheduler()
trigger = IntervalTrigger(days=1)
scheduler.add_job(update_trend_day, trigger=trigger, args=[TOKENS])
scheduler.start()

app = Flask(__name__, static_folder='static')


@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/admin')
def admin():
    return app.send_static_file('admin/index.html')


@app.route('/<path:target>')
def resource(target):
    return app.send_static_file(target)


@app.route('/survey', methods=['POST'])
def survey():
    if request.form['haveAddress'] == 'true':
        uid = request.form['id']
        address = request.form['address']
        worker = Thread(target=cf_recommend, args=[uid, address, PROFILES, USER_ARRAY, TOKENS])
        WORKERS[uid] = worker
        worker.start()
        greet = 'We got your Address. Lets see your Investment Preference'
    else:
        greet = 'No Address is fine. Lets see your Investment Preference'
    return render_template('survey.html', message=greet)

@app.route('/admin/survey', methods=['POST'])
def admin_survey():
    if request.form['haveAddress'] == 'true':
        uid = request.form['id']
        address = request.form['address']
        worker = Thread(target=cf_recommend, args=[uid, address, PROFILES, USER_ARRAY, TOKENS])
        WORKERS[uid] = worker
        worker.start()
        greet = 'We got your Address. Lets see your Investment Preference'
    else:
        greet = 'No Address is fine. Lets see your Investment Preference'
    return render_template('admin/survey.html', message=greet)


def create_message(p, token, acc, avg_knn, npl, horizon):
    msg = ""

    if acc > 0.5:
        if acc > 0.9:
            msg = "The very high confidence of the model for the predicted trend"
        elif acc > 0.75:
            msg = "The high confidence of the model for the predicted trend"
        else:
            msg = "The above average confidence of the model for the predicted trend"
        
        if p.survey_scores[token] > 0.5:
            if p.survey_scores[token] > 0.75:
                msg += " as well as a high return chances"
            else:
                msg += " as well as an above average return chances"
            
        if p.knn_scores[token] > avg_knn:
            if p.knn_scores[token] > avg_knn * 1.5:
                msg += " and a high similarity score with other good wallets"
            else:
                msg += " and an above average similarity score with other good wallets"
        
        msg += " make this coin a good recomendation."
    else:
        if p.survey_scores[token] > 0.5:
            if p.survey_scores[token] > 0.75:
                msg = "A high return chances"
            else:
                msg = "An above average return chances"
            
            if p.knn_scores[token] > avg_knn:
                if p.knn_scores[token] > avg_knn * 1.5:
                    msg += " and a high similarity score with other good wallets"
                else:
                    msg += " and an above average similarity score with other good wallets"
            
            msg += " make this coin a good recomendation."
        else:
            if p.knn_scores[token] > avg_knn:
                if p.knn_scores[token] > avg_knn * 1.5:
                    msg += "A high similarity score with other good wallets"
                else:
                    msg += "An above average similarity score with other good wallets"
            
            msg += " make this coin a good recomendation."

    if horizon == "week":
        horizon_msg = "some days ago"
    elif horizon == "month":
        horizon_msg = "some weeks ago"
    elif horizon == "season":
        horizon_msg = "some months ago"
    elif horizon == "year":
        horizon_msg = "a year ago"

    msg += f"\tFollowing this coin recommendation strategy investing {horizon_msg}, profit would be {npl:.2f}%" 
    return msg

@app.route('/result', methods=['POST'])
def result():
    TREND_DATA = read_trend_day()
    code = ['r1', 'r2', 'r3', 'r4', 'r5']
    uid = request.form['id']
    output = {}
    if uid in WORKERS:
        worker = WORKERS[uid]
        worker.join()
        p = PROFILES[uid]
        if p.status != 'Ok':
            return render_template('error.html', message=p.status)
        survey_recommend(uid, PROFILES, ATTRIBUTES, request.form)
        recommendations, overall_score = combine_score(p.knn_scores, p.survey_scores, request.form['horizon'], TREND_DATA)
        predictions, metrics = get_trend(recommendations[:5], TREND_DATA)
        avg_knn = np.mean([x for x in p.knn_scores.values()])
        for c, t, m, i in zip(code, recommendations, metrics, range(5)):
            sorted_models = sorted(m.keys(), key=lambda k: m[k][0], reverse=True)
            pnl = get_pnl(m, request.form['horizon'])
            output[c] = TOKENS.loc[t, 'Name']
            output[c + '-acc'] = '%.2f' % m[sorted_models[0]][0] 
            output[c + '-f1'] = '%.2f' % m[sorted_models[0]][1] 
            output[c + '-prec'] = '%.2f' % m[sorted_models[0]][2] 
            output[c + '-img'] = TOKENS.loc[t, 'Image']
            output[c + '-des'] = TOKENS.loc[t, 'Description'] + "\n" + create_message(p, t, m["RNN"][0], avg_knn, pnl, request.form['horizon'])
            output[c + '-1d'] = 'Up' if predictions[i]["RNN"][0] > 0 else 'Down'
            output[c + '-7d'] = 'Up' if predictions[i]["RNN"][1] > 0 else 'Down'
            output[c + '-30d'] = 'Up' if predictions[i]["RNN"][2] > 0 else 'Down'
    else:
        survey_recommend(uid, PROFILES, ATTRIBUTES, request.form)
        p = PROFILES[uid]
        p.knn_scores = {r: 1.0 for r in TOKENS.index}

        recommendations, overall_score = combine_score(p.knn_scores, p.survey_scores, request.form['horizon'], TREND_DATA)
        predictions, metrics = get_trend(recommendations[:5], TREND_DATA)
        avg_knn = np.mean([x for x in p.knn_scores.values()])
        for c, t, m, i in zip(code, recommendations, metrics, range(5)):
            sorted_models = sorted(m.keys(), key=lambda k: m[k][0], reverse=True)
            pnl = get_pnl(m, request.form['horizon'])
            output[c] = TOKENS.loc[t, 'Name']
            output[c + '-acc'] = '%.2f' % m[sorted_models[0]][0] 
            output[c + '-f1'] = '%.2f' % m[sorted_models[0]][1] 
            output[c + '-prec'] = '%.2f' % m[sorted_models[0]][2] 
            output[c + '-img'] = TOKENS.loc[t, 'Image']
            output[c + '-des'] = TOKENS.loc[t, 'Description'] + "\n" + create_message(p, t, m["RNN"][0], avg_knn, pnl, request.form['horizon'])
            output[c + '-1d'] = 'Up' if predictions[i]["RNN"][0] > 0 else 'Down'
            output[c + '-7d'] = 'Up' if predictions[i]["RNN"][1] > 0 else 'Down'
            output[c + '-30d'] = 'Up' if predictions[i]["RNN"][2] > 0 else 'Down'
            # output[c + '-message'] = create_message(p, t, m[0], avg_knn)

    return render_template('result.html', data=json.dumps(output))

@app.route('/admin/result', methods=['POST'])
def admin_result():
    TREND_DATA = read_trend_day()
    code = ['r1', 'r2', 'r3', 'r4', 'r5']
    uid = request.form['id']
    output = {}
    if uid in WORKERS:
        worker = WORKERS[uid]
        worker.join()
        p = PROFILES[uid]
        if p.status != 'Ok':
            return render_template('error.html', message=p.status)
        survey_recommend(uid, PROFILES, ATTRIBUTES, request.form)
        recommendations, overall_score = combine_score(p.knn_scores, p.survey_scores, request.form['horizon'], TREND_DATA)
        predictions, metrics = get_trend(recommendations[:5], TREND_DATA)
        avg_knn = np.mean([x for x in p.knn_scores.values()])
        for c, t, m, i in zip(code, recommendations, metrics, range(5)):
            sorted_models = sorted(m.keys(), key=lambda k: m[k][0], reverse=True)
            pnl = get_pnl(m,  request.form['horizon'])
            output[c] = TOKENS.loc[t, 'Name']
            output[c + '-acc'] = ','.join(['%.2f' % m[k][0] for k in sorted_models])
            output[c + '-f1'] = ','.join(['%.2f' % m[k][1] for k in sorted_models])
            output[c + '-prec'] = ','.join(['%.2f' % m[k][2] for k in sorted_models])
            output[c + '-models'] = ','.join([k for k in sorted_models])
            output[c + '-img'] = TOKENS.loc[t, 'Image']
            output[c + '-des'] = TOKENS.loc[t, 'Description'] + "\n" + create_message(p, t, m["RNN"][0], avg_knn, pnl, request.form['horizon'])
            output[c + '-1d'] = 'Up' if predictions[i]["RNN"][0] > 0 else 'Down'
            output[c + '-7d'] = 'Up' if predictions[i]["RNN"][1] > 0 else 'Down'
            output[c + '-30d'] = 'Up' if predictions[i]["RNN"][2] > 0 else 'Down'
    else:
        survey_recommend(uid, PROFILES, ATTRIBUTES, request.form)
        p = PROFILES[uid]
        p.knn_scores = {r: 1.0 for r in TOKENS.index}
        recommendations, overall_score = combine_score(p.knn_scores, p.survey_scores, request.form['horizon'], TREND_DATA)
        predictions, metrics = get_trend(recommendations[:5], TREND_DATA)
        avg_knn = np.mean([x for x in p.knn_scores.values()])
        for c, t, m, i in zip(code, recommendations, metrics, range(5)):
            sorted_models = sorted(m.keys(), key=lambda k: m[k][0], reverse=True)
            pnl = get_pnl(m,  request.form['horizon'])
            output[c] = TOKENS.loc[t, 'Name']
            output[c + '-acc'] = ','.join(['%.2f' % m[k][0] for k in sorted_models])
            output[c + '-f1'] = ','.join(['%.2f' % m[k][1] for k in sorted_models])
            output[c + '-prec'] = ','.join(['%.2f' % m[k][2] for k in sorted_models])
            output[c + '-models'] = ','.join([k for k in sorted_models])
            output[c + '-img'] = TOKENS.loc[t, 'Image']
            output[c + '-des'] = TOKENS.loc[t, 'Description'] + "\n" + create_message(p, t, m["RNN"][0], avg_knn, pnl, request.form['horizon'])
            output[c + '-1d'] = 'Up' if predictions[i]["RNN"][0] > 0 else 'Down'
            output[c + '-7d'] = 'Up' if predictions[i]["RNN"][1] > 0 else 'Down'
            output[c + '-30d'] = 'Up' if predictions[i]["RNN"][2] > 0 else 'Down'

    return render_template('admin/result.html', data=json.dumps(output))


if __name__ == '__main__':
    if AUTO_UPDATE:
        update = Thread(target=update_loop, args=['data/tokens.csv'])
        update.start()
    app.run(port=80, host='0.0.0.0', debug=True)
