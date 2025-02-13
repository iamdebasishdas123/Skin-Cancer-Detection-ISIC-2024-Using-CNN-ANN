import pandas as pd

def imbalenced(train, neg_sample,pos_sample):

    negative_df=train[train["target"]== 0].sample(frac=neg_sample,random_state=42)
    positive_df=train[train["target"]== 1].sample(frac=pos_sample,random_state=42)
    return pd.concat([negative_df,positive_df])