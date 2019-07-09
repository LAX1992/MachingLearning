import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

data1 = np.load('ml19spring/X_train.npz')
data2 = np.load('ml19spring/Y_train.npz')
data3 = np.load('ml19spring/X_test.npz')

trainX_data = data1['arr_0']
trainY_data = data2['arr_0']
testX_data = data3['arr_0']

X_train, X_vali, Y_train, Y_vali = train_test_split(trainX_data, trainY_data, test_size = 0.3, random_state = 42)

'''------------------step0---------------------------'''
print("training......step", 0)
xgb_model =xgb.XGBRegressor(
    n_estimators=1000,
    #n_estimators=500,
    subsample=0.7,
    colsample_bytree=0.7,
    learning_rate=0.1,
    max_depth=13,
    objective='reg:logistic',
    n_jobs=3,
)
    
xgb_model.fit(
    X_train, Y_train[:, 0],
    early_stopping_rounds=5,
    eval_metric='mae',
    eval_set=[(X_vali, Y_vali[:, 0])]
)

print("predicting......")
xgb_model.save_model("001"+str(1)+".model")
xgb_model = xgb.Booster(model_file="001"+str(1)+".model")

dtest = xgb.DMatrix(testX_data)
y_pred = xgb_model.predict(dtest)
np.savetxt('result_'+str(1)+'.csv', y_pred, delimiter = ',')

'''------------------step1---------------------------'''
print("training......step", 1)
xgb_model =xgb.XGBRegressor(
    n_estimators=1000,
    #n_estimators=500,
    subsample=0.7,
    colsample_bytree=0.7,
    learning_rate=0.1,
    #5
    max_depth=13,
    objective='reg:squarederror',
    n_jobs=3
)
    
xgb_model.fit(
    X_train, Y_train[:, 1],
    early_stopping_rounds=5,
    eval_metric='mae',
    eval_set=[(X_vali, Y_vali[:, 1])]
)

print("predicting......")
xgb_model.save_model("001"+str(2)+".model")
xgb_model = xgb.Booster(model_file="001"+str(2)+".model")

dtest = xgb.DMatrix(testX_data)
y_pred = xgb_model.predict(dtest)
np.savetxt('result_'+str(2)+'.csv', y_pred, delimiter = ',')

'''------------------step2---------------------------'''
print("training......step", 2)
xgb_model =xgb.XGBRegressor(
    n_estimators=1000,
    #n_estimators=500,
    subsample=0.7,
    colsample_bytree=0.7,
    learning_rate=0.1,
    max_depth=13,
    objective='reg:logistic',
    n_jobs=3
)
    
xgb_model.fit(
    X_train, Y_train[:, 2],
    early_stopping_rounds=5,
    eval_metric='mae',
    eval_set=[(X_vali, Y_vali[:, 2])]
)

print("predicting......")
xgb_model.save_model("001"+str(3)+".model")
xgb_model = xgb.Booster(model_file="001"+str(3)+".model")

dtest = xgb.DMatrix(testX_data)
y_pred = xgb_model.predict(dtest)
np.savetxt('result_'+str(3)+'.csv', y_pred, delimiter = ',')

data1.close()
data2.close()
data3.close()