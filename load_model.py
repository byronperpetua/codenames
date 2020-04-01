
import numpy as np
import pandas as pd

# https://github.com/commonsense/conceptnet-numberbatch
model = pd.read_csv('numberbatch-en-19.08.txt.gz',
                    delimiter=' ',
                    skiprows=1,
                    header=None,
                    index_col=0,
                    names=['v'+str(k) for k in range(300)])
model = model.loc[model.index.notnull(), :]
model.to_pickle('numberbatch.gz')

# https://github.com/hermitdave/FrequencyWords/tree/master/content/2016/en
freq = pd.read_csv('en_full.txt',
                   delimiter=' ',
                   header=None,
                   index_col=0,
                   names=['freq'])
freq['rnk'] = freq.freq.rank(ascending=False, method='min')

vocab_rank = (model.merge(freq, how='left', left_index=True, 
                               right_index=True)
                         .rnk)
vocab_rank.to_pickle('vocab_rank.gz')


