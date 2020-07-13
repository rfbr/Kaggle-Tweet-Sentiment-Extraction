import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np


if __name__ == '__main__':
    df = pd.read_csv('../../data/train.csv')
    df = df.dropna().reset_index(drop=True)
    df = df[df['sentiment'].isin(
        ['positive', 'negative'])].reset_index(drop=True)
    print(df.head())
    df['kfold'] = -1

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx,
               val_idx) in enumerate(kfold.split(X=df, y=df['sentiment'])):
        df.loc[val_idx, 'kfold'] = fold
    df.to_csv('../../data/train_fold_np.csv', index=False)
