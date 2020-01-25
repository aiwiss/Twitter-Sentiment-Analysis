from scipy import stats
import os, time, datetime as dt
import numpy as np, pandas as pd

base_path = os.environ['OneDrive'] + '\\Sentiment Analysis\\Data\\Results\\Iteration1\\02Ngram\\BiGram\\'
results_path = ''

t1 = pd.read_csv(base_path + 'D-Tree.csv')
t2 = pd.read_csv(base_path + 'LinearSVC.csv')

acc1 = t1['test_accuracy']
acc2 = t2['test_accuracy']
print(f't1 accuracy: {"{0:.3f}".format(np.mean(acc1))}\n')
print(f't1 accuracy: {"{0:.3f}".format(np.mean(acc2))}\n')

t, p = stats.ttest_ind(t1['test_accuracy'],t2['test_accuracy'])
print(f't= {t}\n\n')
print(f'p= {p}\n\n')