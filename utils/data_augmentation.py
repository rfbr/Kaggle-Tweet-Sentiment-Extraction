import nlpaug.augmenter.word as naw
import numpy as np
import pandas as pd
import re
import string
from sklearn.model_selection import StratifiedKFold

aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="substitute", top_k=50,
    skip_unknown_word=True, include_detail=True, device='cuda')


def punc_formating(text, punc=set(string.punctuation)):
    words = re.findall(r"[\w']+|[.,*!?;:`-]", text)
    ans = words[0]
    tmp = words[0]
    for word in words[1:]:
        if word in punc or tmp == '`':
            ans += word
        else:
            ans += ' '+word
        tmp = word
    return ans


def find_start_end(text, selected_text):
    len_selected_text = len(selected_text)
    start_idx = -1
    end_idx = -1
    for idx in (i for i, e in enumerate(text) if e == selected_text[0]):
        if text[idx:idx + len_selected_text] == selected_text:
            start_idx = idx
            end_idx = idx + len_selected_text - 1
            break
    return start_idx, end_idx


def data_augmentation(text, selected_text, sentiment, fold):
    tmp = re.findall(r"[\w']+|[.,*!?;:`-]", text)
    text = ' '.join([w for w in tmp])
    tmp = re.findall(r"[\w']+|[.,*!?;:`-]", selected_text)
    selected_text = ' '.join([w for w in tmp])

    start_idx, end_idx = find_start_end(text, selected_text)

    try:
        augmented_text, infos = aug.augment(text)
        for info in infos:
            delta = len(info['new_token']) - len(info['orig_token'])
            if info['orig_start_pos'] < start_idx:
                start_idx += delta
                end_idx += delta
            elif start_idx <= info['orig_start_pos'] < end_idx:
                end_idx += delta
        return punc_formating(augmented_text), punc_formating(augmented_text[start_idx:end_idx+1]), sentiment, fold
    except:
        return np.nan, np.nan, np.nan, np.nan


if __name__ == '__main__':
    # load dataset and assign fold
    df = pd.read_csv('../../data/train.csv')
    df = df.dropna().reset_index(drop=True)
    df['fold'] = -1

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx,
               val_idx) in enumerate(kfold.split(X=df, y=df['sentiment'])):
        df.loc[val_idx, 'fold'] = fold
    aug_df = pd.DataFrame()
    aug_df['text'], aug_df['selected_text'], aug_df['sentiment'], aug_df['fold'] = zip(
        *df.apply(lambda x: data_augmentation(x['text'],
                                              x['selected_text'],
                                              x['sentiment'],
                                              x['fold']), axis=1))
    aug_df = aug_df.dropna()
    main_df = pd.concat([df.drop('textID', axis=1), aug_df],
                        axis=0).reset_index(drop=True)
    main_df.to_csv('../../data/train_aug_fold.csv', index=False)
