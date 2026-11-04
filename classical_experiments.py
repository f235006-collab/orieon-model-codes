import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA
from common import prepare_data, metric_dict, save_predictions, update_metrics, RESULTS

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)
data, features, protocol = prepare_data()
n = len(data)
nt = protocol['n_train']
nv = protocol['n_val']
train = data.iloc[:nt]
val = data.iloc[nt:nt+nv]
test = data.iloc[nt+nv:]


def evaluate_and_store(model, frame, split_name, predictions):
    m = metric_dict(frame.y_close_t1.values, predictions, frame.Gold_Close.values)
    update_metrics(model, split_name, m)
    save_predictions(model, split_name, frame.Date.dt.strftime('%Y-%m-%d'), frame.y_close_t1.values,
                     predictions, frame.Gold_Close.values)
    print(model, split_name, json.dumps(m), flush=True)

# Persistence is the central reference for a non-stationary price-level target.
for name, frame in [('val', val), ('test', test)]:
    evaluate_and_store('Persistence', frame, name, frame.Gold_Close.values)

# ARIMA order chosen using validation RMSE only.
orders = [(1,1,0), (0,1,1), (1,1,1), (2,1,0), (0,1,2)]
arima_val = []
for order in orders:
    start = time.time()
    try:
        fit = ARIMA(train.Gold_Close.values, order=order).fit()
        preds = []
        for actual in val.Gold_Close.values:
            preds.append(float(fit.forecast(1)[0]))
            fit = fit.append([actual], refit=False)
        m = metric_dict(val.y_close_t1.values, preds, val.Gold_Close.values)
        arima_val.append({'order': order, **m})
        print('ARIMA', order, 'validation RMSE', m['RMSE'], 'seconds', time.time()-start, flush=True)
    except Exception as exc:
        print('ARIMA failed', order, repr(exc), flush=True)
with open(RESULTS / 'arima_validation.json', 'w') as f:
    json.dump(arima_val, f, indent=2)
best = min(arima_val, key=lambda r: r['RMSE'])
best_order = tuple(best['order'])
with open(RESULTS / 'selected_classical.json', 'w') as f:
    json.dump({'ARIMA_order': best_order}, f, indent=2)
evaluate_and_store('ARIMA', val, 'val', None) if False else None
# Refit the selected order on training data only; roll through validation and test without refitting.
fit = ARIMA(train.Gold_Close.values, order=best_order).fit()
for name, frame in [('val', val), ('test', test)]:
    preds = []
    for actual in frame.Gold_Close.values:
        preds.append(float(fit.forecast(1)[0]))
        fit = fit.append([actual], refit=False)
    evaluate_and_store(f'ARIMA{best_order}', frame, name, preds)

Xtr = train[features].values
ytr = train.y_close_t1.values
Xv = val[features].values
Xt = test[features].values

# Ridge regression with training-only scaling.
scaler = StandardScaler().fit(Xtr)
Ztr, Zv, Zt = scaler.transform(Xtr), scaler.transform(Xv), scaler.transform(Xt)
ridge_results = []
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    model = Ridge(alpha=alpha).fit(Ztr, ytr)
    p = model.predict(Zv)
    m = metric_dict(val.y_close_t1.values, p, val.Gold_Close.values)
    ridge_results.append({'alpha': alpha, **m})
best_ridge = min(ridge_results, key=lambda r: r['RMSE'])
model = Ridge(alpha=best_ridge['alpha']).fit(Ztr, ytr)
for name, frame, X in [('val', val, Zv), ('test', test, Zt)]:
    evaluate_and_store('Ridge', frame, name, model.predict(X))

# Random forest: small predeclared grid, selected on validation RMSE.
rf_results = []
for depth in [5, 10, None]:
    for leaf in [1, 5, 15]:
        model = RandomForestRegressor(n_estimators=250, max_depth=depth,
                    min_samples_leaf=leaf, max_features=0.7, random_state=SEED, n_jobs=-1)
        model.fit(Xtr, ytr)
        p = model.predict(Xv)
        m = metric_dict(val.y_close_t1.values, p, val.Gold_Close.values)
        rf_results.append({'max_depth': depth, 'min_samples_leaf': leaf, **m})
        print('RF', depth, leaf, 'val RMSE', m['RMSE'], flush=True)
best_rf = min(rf_results, key=lambda r: r['RMSE'])
model = RandomForestRegressor(n_estimators=500, max_depth=best_rf['max_depth'],
            min_samples_leaf=best_rf['min_samples_leaf'], max_features=0.7,
            random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
for name, frame, X in [('val', val, Xv), ('test', test, Xt)]:
    evaluate_and_store('RandomForest', frame, name, model.predict(X))

# LightGBM: compact grid with fixed seed and validation-only selection.
lgb_results = []
configs = [
    (3, 15, 0.03), (5, 31, 0.03), (-1, 31, 0.03),
    (3, 15, 0.05), (5, 31, 0.05), (-1, 31, 0.05),
]
for depth, leaves, lr in configs:
    model = LGBMRegressor(n_estimators=400, learning_rate=lr, max_depth=depth,
                num_leaves=leaves, subsample=0.85, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=SEED, verbosity=-1, n_jobs=-1)
    model.fit(Xtr, ytr)
    p = model.predict(Xv)
    m = metric_dict(val.y_close_t1.values, p, val.Gold_Close.values)
    lgb_results.append({'max_depth': depth, 'num_leaves': leaves, 'learning_rate': lr, **m})
    print('LGBM', depth, leaves, lr, 'val RMSE', m['RMSE'], flush=True)
best_lgb = min(lgb_results, key=lambda r: r['RMSE'])
model = LGBMRegressor(n_estimators=400, learning_rate=best_lgb['learning_rate'],
            max_depth=best_lgb['max_depth'], num_leaves=best_lgb['num_leaves'],
            subsample=0.85, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=SEED, verbosity=-1, n_jobs=-1).fit(Xtr, ytr)
for name, frame, X in [('val', val, Xv), ('test', test, Xt)]:
    evaluate_and_store('LightGBM', frame, name, model.predict(X))

with open(RESULTS / 'classical_tuning.json', 'w') as f:
    json.dump({'Ridge': ridge_results, 'RandomForest': rf_results, 'LightGBM': lgb_results,
               'selected_Ridge': best_ridge, 'selected_RandomForest': best_rf,
               'selected_LightGBM': best_lgb}, f, indent=2)
print('CLASSICAL EXPERIMENTS COMPLETE', flush=True)
