# Helper script to get data annotation results

import pandas as pd
import helper

data_path = helper.data_path

dataset_lex = pd.read_csv(data_path + 'dataset-lex.csv')
dataset_lex_neg = pd.read_csv(data_path + 'dataset-lex-neg.csv')

print(dataset_lex.shape)
print(dataset_lex_neg.shape)

print(dataset_lex.groupby('target_names').count())
print(dataset_lex_neg.groupby('target_names').count())

# dataset-lex
# negative       29064
# neutral       114333
# positive       36971

# dataset-lex-neg
# negative       23333
# neutral       104142
# positive       52893
