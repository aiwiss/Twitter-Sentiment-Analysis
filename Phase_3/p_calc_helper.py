import helper, models
import numpy as np, pandas as pd, scipy as sp

results_path = helper.results_path

classifier_names = models.classifier_names

# read the baseline model performance results
baseline_data = pd.read_csv(f'{results_path}Iteration1\\01Baseline\\LinearSVC.csv')
baseline = baseline_data['test_accuracy']

results = {}

# read the produced results in one iteration
for i in range(7):
    scores = pd.read_csv(f'{results_path}Combination\\UniGram\\{classifier_names[i]}.csv')
    results[classifier_names[i]] = pd.DataFrame.from_dict(scores)

# calculate p values between each iteration model and baseline model performance results
with open(f'{results_path}p-values\\baseline_results.txt', 'a', encoding='utf-8') as f:
    for i in range(7):
        a = results[classifier_names[i]]['test_accuracy']
        for j in range(7):
            if i == j:
                continue
            
            b = results[classifier_names[j]]['test_accuracy']
            t, p = sp.stats.ttest_ind(a,b)
            f.write(f'{classifier_names[i]} - {classifier_names[j]}\n')
            f.write(f't = {t}\np = {p}\n\n')