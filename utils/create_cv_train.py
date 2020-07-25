import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_k_fold_csv(path='../../../data/train.csv', n_splits=5):
    df = pd.read_csv(path)
    df = df.dropna().reset_index(drop=True)
    df = df[df['sentiment'].isin(
        ['positive', 'negative'])].reset_index(drop=True)
    df['kfold'] = -1

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (_,
               val_idx) in enumerate(kfold.split(X=df, y=df['sentiment'])):
        df.loc[val_idx, 'kfold'] = fold
    df.to_csv(os.path.join(os.path.dirname(path),
                           'train_fold.csv'), index=False)


if __name__ == '__main__':
    create_k_fold_csv()
