import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib import pyplot
import os
os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'



data = np.random.rand(800, 297)  # 5 entities, each contains 10 features
val = np.random.rand(100, 297)
test = np.random.rand(10, 297)
label = np.random.randint(2, size=800)
val_label = np.random.randint(2, size=100)

dtrain = xgb.DMatrix(data, label=label)
dval = xgb.DMatrix(val, label=val_label)
dtest = xgb.DMatrix(test)

param = {'max_depth':2,
         'eta':0.2,
         'subsample' : 0.9,
         'colsample_bytree': 0.8,
         'silent':1,
         'objective':'binary:logistic',
         'eval_metric':['logloss', 'auc', 'error']
         #'eval_metric':'logloss'
         }

num_round = 30

watchlist = [(dval,'val'), (dtrain,'train')]

evals_result = {}

print("start training...")

bst = xgb.train(param, dtrain, num_round, watchlist, evals_result=evals_result)

#print(evals_result['train']['error'])
#print(evals_result['val']['error'])

#print(evals_result['train']['logloss'])
#print(evals_result['val']['logloss'])


results = evals_result
epochs = len(evals_result['val']['error'])
x_axis = range(0, epochs)

plt.figure()
plt.plot(x_axis, results['train']['logloss'], label='Train')
plt.plot(x_axis, results['val']['logloss'], label='Validation')
plt.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
save_path = "/home/leo/ant/score/"
plt.savefig(save_path + "LogLoss_{}.png".format(0.2))
#plt.show()


plt.figure()
plt.plot(x_axis, results['train']['error'], label='Train')
plt.plot(x_axis, results['val']['error'], label='Validation')
plt.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
save_path = "/home/leo/ant/score/"
plt.savefig(save_path + "Error{}.png".format(0.2))
#plt.show()

plt.figure()
plt.plot(x_axis, results['train']['auc'], label='Train')
plt.plot(x_axis, results['val']['auc'], label='Validation')
plt.legend()
plt.ylabel('AUC')
plt.title('XGBoost AUC')
save_path = "/home/leo/ant/score/"
plt.savefig(save_path + "AUC{}.png".format(0.2))
#plt.show()

"""
res = xgb.cv(param, dtrain, num_round, nfold=10, metrics={'error'}, seed=0)

#print('Access logloss metric directly from evals_result:')
#print(evals_result['eval']['logloss'])
#saver = bst.save_model("/home/leo/ant/model/xbg.model")
#bst.dump_model("/home/leo/ant/model/xbg.txt")
#bst.dump_model('/home/leo/ant/model/xbg.txt','/home/leo/ant/model/features.txt',  with_stats=True)
#fscore = bst.get_fscore()
#print("get_fscore : " , fscore)
#features = pd.Series(bst.get_fscore()).sort_values(ascending=False)
#np.save("/home/leo/ant/cuuu.npy", features)
#cuii = np.load("/home/leo/ant/cuuu.npy")
#print(cuii)
#features.to_csv("/home/leo/ant/cuuu.csv")
#df = pd.DataFrame(data = fscore, index =["importance"])
#df.to_csv("/home/leo/ant/cuuu.csv")

#xgb.plot_importance(bst)
#xgb.plot_tree(bst, num_trees=2)
#plt.show()
### predict using first 2 tree
#preds = bst.predict(dtest, ntree_limit=bst.best_iteration)
print(bst.best_iteration)
"""
