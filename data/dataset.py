import torch
import numpy as np
import pandas as pd
from utils import config


class TweetDataset:
    def __init__(self, tweets, sentiments, selected_texts):
        self.tweets = tweets
        self.sentiments = sentiments
        self.selected_texts = selected_texts
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, key):
        tweet = ' '.join(str(self.tweets[key]).split())
        selected_text = ' '.join(str(self.selected_texts[key]).split())
        sentiment = self.sentiments[key]
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
            if sum(char_targets[start_idx:end_idx]) > 0:
                target_idx.append(idx)
        targets_start = target_idx[0]
        targets_end = target_idx[-1]

        sentiment_id = {
            'positive': 1313,
            'negative': 2430,
            'neutral': 7974
        }

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
            input_ids = input_ids[:self.max_len]
            mask = mask[:self.max_len]
            token_type_ids = token_type_ids[:self.max_len]
            tweet_offsets = tweet_offsets[:self.max_len]
            targets_start = np.minimum(targets_start, self.max_len-1)
            targets_end = np.minimum(targets_end, self.max_len-1)

        out = {
            'ids': torch.tensor(input_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets_start': torch.tensor(targets_start, dtype=torch.long),
            'targets_end': torch.tensor(targets_end, dtype=torch.long),
            'original_tweet': tweet,
            'original_selected_text': selected_text,
            'sentiment': sentiment,
            'offsets': torch.tensor(tweet_offsets, dtype=torch.long)
        }
        return out


if __name__ == '__main__':
    df = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop=True)
    dataset = TweetDataset(tweets=df.text.values,
                           selected_texts=df.selected_text.values,
                           sentiments=df.sentiment.values)
    print(dataset[0])
