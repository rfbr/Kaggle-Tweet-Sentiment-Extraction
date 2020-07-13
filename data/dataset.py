import nlpaug.augmenter.word as naw
from sklearn.model_selection import StratifiedKFold
import string
import pandas as pd
import torch
import numpy as np
from utils import config
import pickle
import re
import pdb
from torch.utils.data import Dataset
from collections import defaultdict


def misspel_check(text, misspell):
    corrected_text = []
    mapp_ = defaultdict(str)
    for w in text.split():
        if (misspell.get(w)):
            corrected_text.append(misspell[w])
            mapp_[misspell[w]] = w
        else:
            corrected_text.append(w)
            if not (mapp_.get(w)):
                mapp_[w] = w
    return ' '.join([w for w in corrected_text]), mapp_


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


def data_augmentation(text, selected_text):
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
        return punc_formating(augmented_text), punc_formating(augmented_text[start_idx:end_idx+1])
    except:
        return text, selected_text


def find_word(tweet, word):
    n = len(word.split())
    idx = tweet.find(word)
    while idx > 0 and tweet[idx-1] != ' ':
        idx -= 1
    return ' '.join(w for w in tweet[idx:].split()[:n+1])


def clean_selected_text(tweet, selected_text):
    tweet_set = set(tweet.split())
    splitted = selected_text.split()
    if len(splitted) == 1 and splitted[0] not in tweet_set:
        return find_word(tweet, splitted[0])
    if splitted[0] not in tweet_set:
        return ' '.join(w for w in splitted[1:])
    if splitted[-1] not in tweet_set:
        last_word = find_word(tweet, selected_text)
        return last_word
    return selected_text


class TweetDataset:
    def __init__(self, tweets, sentiments, selected_texts, augmentation=True, threshold=0.3, denoise=.5):
        self.threshold = threshold
        self.augmentation = augmentation
        self.denoise = denoise
        self.tweets = tweets
        self.sentiments = sentiments
        self.selected_texts = selected_texts
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, key):
        sentiment = self.sentiments[key]
        # if sentiment != 'neutral' and self.threshold != 0:
        #     if self.augmentation and np.random.rand() < self.denoise:
        #         selected_text = clean_selected_text(
        #             self.tweets[key], str(self.selected_texts[key]))
        #         selected_text = selected_text.strip()
        #     else:
        #         selected_text = str(self.selected_texts[key])
        # else:
        selected_text = str(self.selected_texts[key])
        if self.augmentation and np.random.rand() < self.threshold:
            tweet, selected_text = data_augmentation(
                str(self.tweets[key]), selected_text)
        else:
            tweet = ' '.join(str(self.tweets[key]).split())
            selected_text = ' '.join(selected_text.split())

        len_selected_text = len(selected_text)
        start_idx = -1
        end_idx = -1
        for idx in (i for i, e in enumerate(tweet) if e == selected_text[0]):
            if tweet[idx:idx + len_selected_text] == selected_text:
                start_idx = idx
                end_idx = idx + len_selected_text
                break

        char_targets = [0] * len(tweet)
        # Character based approched because some words can be split from the
        # tweet to the selected text
        if start_idx != -1 and end_idx != -1:
            for idx in range(start_idx, end_idx):
                char_targets[idx] = 1

        tokenized_tweet = self.tokenizer.encode(tweet)
        tweet_ids = tokenized_tweet.ids
        tweet_offsets = tokenized_tweet.offsets

        target_idx = []
        for idx, (start_idx, end_idx) in enumerate(tweet_offsets):
            # Even if partial match, consider the word as a target
            if sum(char_targets[start_idx:end_idx]) != 0:
                target_idx.append(idx)
        if target_idx == []:
            print(f'Tweet: {tweet}')
            print(f'Selected text: {str(self.selected_texts[key])}')
            print(f'Denoised selected text: {selected_text}')
        targets_start = target_idx[0]
        targets_end = target_idx[-1]

        sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

        input_ids = [0] + [sentiment_id[sentiment]] + \
            [2] + [2] + tweet_ids + [2]
        token_type_ids = [0] * (len(tweet_ids) + 5)
        mask = [1] * len(token_type_ids)
        tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
        targets_start += 4
        targets_end += 4
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([1] * padding_length)
            mask = mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
        else:
            input_ids = input_ids[:self.max_len - 1]
            input_ids += [2]
            mask = mask[:self.max_len]
            token_type_ids = token_type_ids[:self.max_len]
            tweet_offsets = tweet_offsets[:self.max_len - 1]
            tweet_offsets += [(0, 0)]
            targets_start = np.minimum(targets_start, self.max_len - 1)
            targets_end = np.minimum(targets_end, self.max_len - 1)

        out = {
            'ids': torch.tensor(input_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets_start': torch.tensor(targets_start, dtype=torch.long),
            'targets_end': torch.tensor(targets_end, dtype=torch.long),
            'original_tweet': tweet,
            'original_selected_text': selected_text,
            'sentiment': sentiment,
            'offsets': torch.tensor(tweet_offsets, dtype=torch.long),
        }
        return out


class CorrTweetDataset:
    def __init__(self, tweets, sentiments, selected_texts):
        self.tweets = tweets
        self.sentiments = sentiments
        self.selected_texts = selected_texts
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, key):
        tmp = re.findall(r"[\w']+|[.,*!?;:`-]", self.tweets[key])
        tweet = ' '.join([w for w in tmp])
        corrected_tweet, mapp = misspel_check(tweet, UMAPPING)
        tmp = re.findall(r"[\w']+|[.,*!?;:`-]", self.selected_texts[key])
        selected_text = ' '.join([w for w in tmp])

        corrected_selected_text, _ = misspel_check(selected_text, UMAPPING)
        sentiment = self.sentiments[key]
        len_selected_text = len(corrected_selected_text)
        start_idx = -1
        end_idx = -1

        if (len(corrected_selected_text) == 0):
            print(selected_text)
            print(f'Selected text: {self.selected_texts[key]}')
            print(f'Corrected: {corrected_selected_text}')

        for idx in (i for i, e in enumerate(corrected_tweet)
                    if e == corrected_selected_text[0]):
            if corrected_tweet[idx:idx +
                               len_selected_text] == corrected_selected_text:
                start_idx = idx
                end_idx = idx + len_selected_text
                break
        # quick fix
        # last word was cut
        tmp = corrected_selected_text
        if start_idx == -1 and end_idx == -1:
            corrected_selected_text = corrected_selected_text.rsplit(' ', 1)[0]
            len_selected_text = len(corrected_selected_text)
            for idx in (i for i, e in enumerate(corrected_tweet)
                        if e == corrected_selected_text[0]):
                if corrected_tweet[
                        idx:idx +
                        len_selected_text] == corrected_selected_text:
                    start_idx = idx
                    end_idx = idx + len_selected_text
                    break
        # front word was cut
        if start_idx == -1 and end_idx == -1:
            corrected_selected_text = tmp.split(' ', 1)[1]
            len_selected_text = len(corrected_selected_text)
            for idx in (i for i, e in enumerate(corrected_tweet)
                        if e == corrected_selected_text[0]):
                if corrected_tweet[
                        idx:idx +
                        len_selected_text] == corrected_selected_text:
                    start_idx = idx
                    end_idx = idx + len_selected_text
                    break
        char_targets = [0] * len(corrected_tweet)
        # Character based approched because some words can be split from the
        # tweet to the selected text
        if start_idx != -1 and end_idx != -1:
            for idx in range(start_idx, end_idx):
                char_targets[idx] = 1

        tokenized_tweet = self.tokenizer.encode(corrected_tweet)
        tweet_ids = tokenized_tweet.ids
        tweet_offsets = tokenized_tweet.offsets
        target_idx = []

        for idx, (start_idx, end_idx) in enumerate(tweet_offsets):
            # Even if partial match, consider the word as a target
            if sum(char_targets[start_idx:end_idx]) > 0:
                target_idx.append(idx)

        targets_start = target_idx[0]
        targets_end = target_idx[-1]
        sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

        input_ids = [0] + [sentiment_id[sentiment]] + \
            [2] + [2] + tweet_ids + [2]
        mask = [1] * len(input_ids)
        tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
        targets_start += 4
        targets_end += 4
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([1] * padding_length)
            mask = mask + ([0] * padding_length)
            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
        else:
            print('Higher than max len')
            input_ids = input_ids[:self.max_len - 1]
            input_ids += [2]
            mask = mask[:self.max_len]
            tweet_offsets = tweet_offsets[:self.max_len - 1]
            tweet_offsets += [(0, 0)]
            targets_start = np.minimum(targets_start, self.max_len - 1)
            targets_end = np.minimum(targets_end, self.max_len - 1)
        # print(list(mapp.items()))
        out = {
            'ids':
            torch.tensor(input_ids, dtype=torch.long),
            'mask':
            torch.tensor(mask, dtype=torch.long),
            'targets_start':
            torch.tensor(targets_start, dtype=torch.long),
            'targets_end':
            torch.tensor(targets_end, dtype=torch.long),
            'original_tweet':
            corrected_tweet,
            'original_selected_text':
            ' '.join(str(self.selected_texts[key]).split()),
            'sentiment':
            sentiment,
            'offsets':
            torch.tensor(tweet_offsets, dtype=torch.long),
            'mapping':
            mapp
        }

        return out


class_mapping = {
    'anger': 0,
    'boredom': 0,
    'empty': 0,
    'enthusiasm': 2,
    'fun': 2,
    'happiness': 2,
    'hate': 0,
    'love': 2,
    'neutral': 1,
    'relief': 2,
    'sadness': 0,
    'surprise': 2,
    'worry': 0
}


class ClassDataset:
    def __init__(self, tweets, sentiments):
        self.tweets = tweets
        self.sentiments = sentiments
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, key):
        tweet = ' '.join(str(self.tweets[key]).split())
        sentiment = self.sentiments[key]

        tokenized_tweet = self.tokenizer.encode(tweet)
        tweet_ids = tokenized_tweet.ids
        mask = tokenized_tweet.attention_mask

        padding_length = self.max_len - len(tweet_ids)
        if padding_length > 0:
            tweet_ids = tweet_ids + ([1] * padding_length)
            mask = mask + ([0] * padding_length)

        else:
            tweet_ids = tweet_ids[:self.max_len]
            mask = mask[:self.max_len]

        out = {
            'ids': torch.tensor(tweet_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'sentiment': class_mapping[sentiment]
        }
        return out
