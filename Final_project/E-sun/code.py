import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.
# import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV,ParameterGrid

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def hit_rate(y_true, y_pred):
    result = np.subtract(y_true,y_pred)
    result= np.absolute(result)
    result = np.divide(result,y_true)
    t = [x for x in result if x<=0.1]
    return (len(t))/y_pred.size 

def eval_score(hitRate,MAPE):
    score = round(hitRate,4)*10000
    if MAPE>1: score += 1
    else: score += (1-MAPE)
    return score

def get_score(y_true, y_pred):
    y_true=np.expm1(y_true)
    y_pred=np.expm1(y_pred)
    MAPE = mean_absolute_percentage_error(y_true,y_pred)
    hitRate = hit_rate(y_true,y_pred)
    score = eval_score(hitRate, MAPE)
    print("MAPE: %f" % (MAPE))
    print("Hit Rate: %f" % (hitRate))
    print("Score: %f" % (score))
    return score

df = pd.read_csv('D:/E-sun/train.csv')
test = pd.read_csv('D:/E-sun/test.csv')

df.shape

train_drop=df.copy()
train_drop=train_drop.drop(['building_id'],axis=1)

test_drop=test.copy()
test_drop=test_drop.drop(['building_id'],axis=1)
test_drop.shape

# sns.distplot(np.log1p(train_drop['total_price']))

train_drop['target']=np.log1p(train_drop['total_price'].values)
train_drop.head()

X, Y = train_drop.iloc[:,:-2],train_drop.iloc[:,-1]

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=123)

xgboost = xgb.XGBRegressor(learning_rate=0.1, n_estimators=2000, tree_method='gpu_hist',
                             max_depth=7, min_child_weight=1,
                             gamma=0, subsample=0.9,
                             colsample_bytree=0.7,
                             objective='reg:linear', nthread=-1,
                             scale_pos_weight=1, seed=0,
                             reg_alpha=0)

params = {'learning_rate':[0.1,0.01],'subsample':[.7,.8,.9],'colsample_bytree':[.7,.8,.9],'n_estimators':[5000,10000,15000,20000]}

tuples=[tuple(para.values()) for para in list(ParameterGrid(params))]

index = pd.MultiIndex.from_tuples(tuples,names=list(ParameterGrid(params))[0].keys())
record=pd.DataFrame(columns=['train score','val score'],index=index)

best_score=0
for param in ParameterGrid(params):
    print ("parameter:", param)
    
    xgboost.set_params(**param)
    xgboost.fit(X_train,y_train)

    y_pred_val=xgboost.predict(X_val)
    y_pred_train=xgboost.predict(X_train)
    
    print('val score:')
    score=get_score(y_val,y_pred_val)
    record.loc[tuple(param.values())]['val score']=score
    
    if score > best_score:
        best_score = score
        best_grid = param
    
    print('train score:')
    score=get_score(y_train,y_pred_train)
    record.loc[tuple(param.values())]['train score']=score
    
    

print("best score:", best_score) 
print ("best parameter:", best_grid)

