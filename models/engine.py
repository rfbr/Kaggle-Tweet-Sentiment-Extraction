from utils.averagemeter import AverageMeter
from utils.metric import jaccard
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import string
from utils import config
import re

punc = set(string.punctuation)


def find_closest_word(tweet, selected_text):
    n = len(selected_text.split())
    idx = tweet.find(selected_text)
    while idx > 0 and tweet[idx-1] != ' ':
        idx -= 1
    return tweet[idx:].split()[:n][-1]


def post_processing(text, selected_text):
    full_punc = True

    for w in re.findall(r"[\w']+|[.,*!?;:`-]", selected_text):
        if w not in punc:
            full_punc = False
    if full_punc:
        return selected_text

    closest_word = find_closest_word(text, selected_text)
    splitted = re.findall(r"[\w']+|[.,*!?;:-]", closest_word)
    if splitted[-1] in punc:
        combinaison = [splitted[0]]
        for i in range(1, len(splitted)):
            combinaison.append(combinaison[-1] + splitted[i])
        return (' '.join(w for w in selected_text.split()[:-1]) + ' ' + ' '.join(w for w in combinaison)).strip()
    elif len(splitted) > 1 and splitted[-2] in punc:
        combinaison = [splitted[0]]
        for i in range(1, len(splitted)):
            combinaison.append(combinaison[-1] + splitted[i])
        return (' '.join(w for w in selected_text.split()[:-1]) + ' ' + ' '.join(w for w in combinaison)).strip()
    else:
        return (' '.join(w for w in selected_text.split()[:-1]) + ' ' + closest_word).strip()


def find_max_prob(log_p_start, log_p_end):
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


def pos_weight(pred_tensor, pos_tensor, neg_weight=1, pos_weight=1):
    # neg_weight for when pred position < target position
    # pos_weight for when pred position > target position
    gap = torch.argmax(pred_tensor, dim=1) - pos_tensor
    gap = gap.type(torch.float32)
    return torch.where(gap < 0, -neg_weight * gap, pos_weight * gap)


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss(reduce='none')  # do reduction later

    start_loss = loss_fct(start_logits, start_positions) * \
        pos_weight(start_logits, start_positions, .5, .5)
    end_loss = loss_fct(end_logits, end_positions) * \
        pos_weight(end_logits, end_positions, .5, .5)

    start_loss = torch.mean(start_loss)
    end_loss = torch.mean(end_loss)

    total_loss = (start_loss + end_loss)
    return total_loss


def cal_loss(pred, gold, trg_pad_idx, eps, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = eps
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        loss = F.cross_entropy(pred,
                               gold,
                               ignore_index=trg_pad_idx,
                               reduction='mean')

    return loss


# def loss_function(output_1, output_2, target_1, target_2):
#     loss_1 = nn.CrossEntropyLoss()(output_1, target_1)
#     loss_2 = nn.CrossEntropyLoss()(output_2, target_2)
#     return loss_1 + loss_2


def loss_function(output_1, output_2, target_1, target_2, eps):
    loss_1 = cal_loss(output_1, target_1, -100, eps)
    loss_2 = cal_loss(output_2, target_2, -100, eps)
    return loss_1 + loss_2


def remove_endding_link(text):
    words = [w for w in text.strip().split(' ')]
    if words[-1].startswith('http'):
        return ' '.join(w for w in words[:-1])
    else:
        return ' '.join(w for w in words)


def remove_name(text):
    words = [w for w in text.strip().split(' ')]
    if words[0].startswith('_'):
        return ' '.join(w for w in words[1:])
    else:
        return ' '.join(w for w in words)


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


def remove_punc(text):
    words = re.findall(r"[\w']+|[.,*!?;:`-]", text)
    last_word = words[-1]
    while len(words) > 2 and last_word in punc and words[-2] in punc:
        words = words[:-1]
    return punc_formating(' '.join([w for w in words]))


def remove_starting_punc(text):
    while len(text) > 1 and text[0] in punc:
        text = text[1:]
    return text


def predict_selected_text(original_tweet, predicted_start, predicted_end,
                          sentiment, offsets):
    # or len(original_tweet) < 20
    if sentiment == 'neutral':
        return original_tweet

    pred_selected_text = ''
    for idx in range(predicted_start, predicted_end + 1):
        pred_selected_text += original_tweet[offsets[idx][0]:offsets[idx][1]]
        if (idx + 1) < len(offsets) and offsets[idx][1] < offsets[idx + 1][0]:
            pred_selected_text += " "
    # if len(pred_selected_text.split()) > 4:
    #     pred_selected_text = post_processing(
    #         original_tweet.strip(), pred_selected_text)
    # pred_selected_text = remove_endding_link(pred_selected_text)
    # if sentiment == 'negative':
    #     pred_selected_text = remove_name(pred_selected_text)
    # pred_selected_text = remove_punc(pred_selected_text)
    return pred_selected_text


def training(data_loader, model, optimizer, device, scheduler, eps):
    model.train()
    losses = AverageMeter()
    tk = tqdm(data_loader, total=len(data_loader))
    for batch_idx, data in enumerate(tk):
        ids = data['ids']
        mask = data['mask']
        token_type_ids = data['token_type_ids']
        targets_start = data['targets_start']
        targets_end = data['targets_end']
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        optimizer.zero_grad()

        logits_start, logits_end = model(ids=ids, mask=mask)

        loss = loss_function(logits_start, logits_end, targets_start,
                             targets_end, eps)
        loss.backward()
        #
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        tk.set_postfix(loss=losses.avg)
    torch.cuda.empty_cache()


def evaluating(data_loader, model, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
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

                # if jac == 0 and len(predicted_selected_text.split()) == 1:
                #     print('Jaccard de 0')
                #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                #     print(f'Tweet: {tweet}')
                #     print(f'Selected_text: {selected_text}')
                #     print(f'Predicted text: {predicted_selected_text}')
                #     print(f'Proba start: {np.exp(log_p_start[pred_start])}')
                #     print(f'Proba end: {np.exp(log_p_end[pred_end])}')
                #     print()
                # if jac == 1 and len(predicted_selected_text.split()) == 1:
                #     print('Jaccard de 1')
                #     print(f'Tweet: {tweet}')
                #     print(f'Selected_text: {selected_text}')
                #     print(f'Predicted text: {predicted_selected_text}')
                #     print(f'Proba start: {np.exp(log_p_start[pred_start])}')
                #     print(f'Proba end: {np.exp(log_p_end[pred_end])}')
                #     print()
                scores.append(jac)

    return np.mean(scores)
