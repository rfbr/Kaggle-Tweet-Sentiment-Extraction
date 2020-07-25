import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.averagemeter import AverageMeter
from utils.loss_function import smooth_ce
from utils.metric import jaccard


def find_max_prob(log_p_start, log_p_end):
    """
    Find the couple (start, end) which maximize the probability P(start&end)
    """
    n = len(log_p_start)
    sum_log_prob = np.zeros((n, n))
    max_sum = np.NINF
    pred_start = -1
    pred_end = -1
    for i in range(n):
        for j in range(i, n):
            sum_log_prob[i, j] = log_p_start[i] + log_p_end[j]
            if (max_sum < sum_log_prob[i, j]):
                pred_start = i
                pred_end = j
                max_sum = sum_log_prob[i, j]
    return pred_start, pred_end


def predict_selected_text(original_tweet, predicted_start, predicted_end,
                          sentiment, offsets):
    if sentiment == 'neutral':
        return original_tweet

    pred_selected_text = ''
    for idx in range(predicted_start, predicted_end + 1):
        pred_selected_text += original_tweet[offsets[idx][0]:offsets[idx][1]]
        if (idx + 1) < len(offsets) and offsets[idx][1] < offsets[idx + 1][0]:
            pred_selected_text += " "
    return pred_selected_text


def training(data_loader, model, optimizer, device, scheduler, eps):
    model.train()
    losses = AverageMeter()
    data_tqdm = tqdm(data_loader, total=len(data_loader))
    for data in data_tqdm:
        ids = data['ids']
        mask = data['mask']
        targets_start = data['targets_start']
        targets_end = data['targets_end']
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        optimizer.zero_grad()

        logits_start, logits_end = model(ids=ids, mask=mask)

        loss = smooth_ce(logits_start, logits_end, targets_start,
                         targets_end, eps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        data_tqdm.set_postfix(loss=losses.avg)
    torch.cuda.empty_cache()


def evaluating(data_loader, model, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for data in data_loader:
            ids = data['ids']
            mask = data['mask']
            original_selected_text = data['original_selected_text']
            original_tweet = data['original_tweet']
            sentiment = data['sentiment']
            targets_start = data['targets_start']
            targets_end = data['targets_end']
            offsets = data['offsets'].numpy()

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            logits_start, logits_end = model(ids=ids, mask=mask)

            prob_start = nn.functional.log_softmax(logits_start,
                                                   -1).cpu().detach().numpy()
            prob_end = nn.functional.log_softmax(logits_end,
                                                 -1).cpu().detach().numpy()

            for idx, tweet in enumerate(original_tweet):
                selected_text = original_selected_text[idx]
                tweet_sentiment = sentiment[idx]
                log_p_start = prob_start[idx]
                log_p_end = prob_end[idx]
                pred_start, pred_end = find_max_prob(log_p_start, log_p_end)
                predicted_selected_text = predict_selected_text(
                    original_tweet=tweet,
                    predicted_start=pred_start,
                    predicted_end=pred_end,
                    sentiment=tweet_sentiment,
                    offsets=offsets[idx])
                jac = jaccard(selected_text.strip(), predicted_selected_text)
                scores.append(jac)
    return np.mean(scores)
