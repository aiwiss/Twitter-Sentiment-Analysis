import matplotlib.pyplot as plt
import helper, models
import numpy as np, pandas as pd, scipy as sp

results_path = helper.results_path

classifier_names = models.classifier_names

results = {}

# read the results produced by cross-validation
for i in range(7):
    scores = pd.read_csv(f'{results_path}Iteration1\\01Baseline\\{classifier_names[i]}.csv')
    results[classifier_names[i]] = pd.DataFrame.from_dict(scores)


# fit and score time plotting
fit_time_avg = []
score_time_avg = []
for i in range(7):
    # fit_time_avg[classifier_names[i]] = np.mean(results[classifier_names[i]]['fit_time'])
    # score_time_avg[classifier_names[i]] = np.mean(results[classifier_names[i]]['score_time'])
    fit_time_avg.append(np.mean(results[classifier_names[i]]['fit_time']))
    score_time_avg.append(np.mean(results[classifier_names[i]]['score_time']))



print(fit_time_avg)
print(score_time_avg)

fig = plt.figure()
ax = fig.add_subplot(111)

N = 7
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars

# declare diagram bars parameters
rects1 = ax.bar(ind, fit_time_avg, width, color='r')
rects2 = ax.bar(ind+width, score_time_avg, width, color='g')

# set labels text
ax.set_ylabel('Time (minutes)')
ax.set_xticks(ind+width)
ax.set_xticklabels( classifier_names )
ax.legend( (rects1[0], rects2[0]), ('Fit time', 'Score time') )

# rotate classifier names text so it fits better in the diagram
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

plt.show()