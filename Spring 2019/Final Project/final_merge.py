import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
def datasets():
    X = np.load('ml19spring/X_train.npz', mmap_mode='r')['arr_0']
    y = np.load('ml19spring/Y_train.npz', mmap_mode='r')['arr_0']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=7500, random_state=42)
    X_test = np.load('ml19spring/X_test.npz', mmap_mode='r')['arr_0']
    return X_train, X_val, y_train, y_val, X_test

X_train, X_val, y_train, y_val, X_test = datasets()

xgb_penetration_rate = xgb.XGBRegressor(
    n_estimators=1000,
    subsample_freq=1,
    subsample=0.7,
    colsample_bytree=0.7,
    learning_rate=0.1,
    max_depth=8,
    num_leaves=256,
    objective='reg:logistic',
    n_jobs=-1
)
xgb_penetration_rate.fit(
    X_train, y_train[:, 0],
    early_stopping_rounds=5,
    eval_metric='mae',
    eval_set=[(X_val, y_val[:, 0])]
)
xgb_mesh = xgb.XGBRegressor(
    n_estimators=1000,
    subsample_freq=1,
    subsample=0.7,
    colsample_bytree=0.7,
    learning_rate=0.1,
    max_depth=8,
    num_leaves=256,
    reg_alpha=0.05,
    objective='reg:squarederror',
    n_jobs=-1
)
xgb_mesh.fit(
    X_train, y_train[:, 1],
    early_stopping_rounds=5,
    eval_metric='mae',
    eval_set=[(X_val, y_val[:, 1])]
)
xgb_alpha = xgb.XGBRegressor(
    n_estimators=1000,
    subsample_freq=1,
    subsample=0.7,
    colsample_bytree=0.7,
    learning_rate=0.1,
    max_depth=8,
    num_leaves=256,
    objective='reg:logistic',
    n_jobs=-1
)
xgb_alpha.fit(
    X_train, y_train[:, 2],
    early_stopping_rounds=5,
    eval_metric='mae',
    eval_set=[(X_val, y_val[:, 2])]
)
y_test = np.c_[xgb_penetration_rate.predict(X_test),
               xgb_mesh_size.predict(X_test)]
np.savetxt("Y_temp.csv", y_test, delimiter=",")


print("RandomForest...")
tree = DecisionTreeRegressor(
    max_depth=4,
    max_leaf_nodes=16,
    criterion='mae',
    
)
clf = BaggingRegressor(
    tree, 
    n_estimators=240, 
    max_samples=0.1,
    random_state=76,
    bootstrap_features=True,
    max_features=0.1,
    verbose=3,
    n_jobs=-1
)

clf.fit(X_train, y_train)
print("done!")
print("score:", mean_absolute_error(clf.predict(X_val), y_val))
y_test = clf.predict(X_test)
np.savetxt("Y_test.csv", y_test, delimiter=",")
print(y_test)

gbm_penetration_rate = lgb.LGBMRegressor(
    n_estimators=1000,
    subsample_freq=1,
    subsample=0.7,
    colsample_bytree=0.7,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=256,
    objective='xentropy',
    device='gpu',
)
gbm_penetration_rate.fit(
    X_train, y_train[:, 0],
    eval_set=[(X_val, y_val[:, 0])],
    early_stopping_rounds=5,
    eval_metric='mae'
)
gbm_mesh_size = lgb.LGBMRegressor(
    n_estimators=1000,
    subsample_freq=1,
    subsample=0.8,
    colsample_bytree=0.8,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=256,
    reg_alpha=0.05,
    objective='mae'
)

gbm_mesh_size.fit(
    X_train, y_train[:, 1],
    eval_set=[(X_val, y_val[:, 1])],
    early_stopping_rounds=5,
    eval_metric='mae'
)
gbm_alpha = lgb.LGBMRegressor(
    n_estimators=1000,
    subsample_freq=1,
    subsample=0.8,
    colsample_bytree=0.8,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=256,
    objective='xentropy'
)

gbm_alpha.fit(
    X_train, y_train[:, 2],
    eval_set=[(X_val, y_val[:, 2])],
    early_stopping_rounds=5,
    eval_metric='mae'
)
y_test = np.c_[gbm_penetration_rate.predict(X_test),
               gbm_mesh_size.predict(X_test),
               gbm_alpha.predict(X_test)]
np.savetxt("Y_test.csv", y_test, delimiter=",")

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_pred))

y_pred = np.c_[
    gbm_penetration_rate.predict(X_val),
    gbm_mesh_size.predict(X_val),
    gbm_alpha.predict(X_val)
]

print(mean_absolute_percentage_error(y_val[:, 2], y_pred[:, 2]))

gbm_penetration_rate = lgb.LGBMRegressor(
    n_estimators=200,
    subsample_freq=1,
    subsample=0.8,
    colsample_bytree=0.8,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=256,
    objective='xentropy',
    device='gpu',
)

xgb_penetration_rate = xgb.XGBRegressor(
    n_estimators=200,
    subsample_freq=1,
    subsample=0.7,
    colsample_bytree=0.7,
    learning_rate=0.1,
    max_depth=8,
    num_leaves=256,
    objective='reg:logistic',
    n_jobs=-1
)

meta_reg = Ridge()

stregr = StackingRegressor(regressors=[gbm_penetration_rate, xgb_penetration_rate], 
                           meta_regressor=meta_reg)

stregr.fit(
    X_train, y_train[:, 0]
)
print(1 - stregr.score(X_val, y_val[:, 0]))