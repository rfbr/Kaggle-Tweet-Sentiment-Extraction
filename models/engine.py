from utils.averagemeter import AverageMeter
from utils.metric import jaccard
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def loss_function(output_1, output_2, target_1, target_2):
    loss_1 = nn.CrossEntropyLoss()(output_1, target_1)
    loss_2 = nn.CrossEntropyLoss()(output_2, target_2)
    return 3*loss_1 + loss_2


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


def predict_selected_text(original_tweet, predicted_start, predicted_end, sentiment, offsets):

    if sentiment == "neutral" or len(original_tweet.strip().split()) < 4:
        return original_tweet

    pred_selected_text = ''
    for idx in range(predicted_start, predicted_end+1):
        pred_selected_text += original_tweet[offsets[idx][0]: offsets[idx][1]]
        if (idx+1) < len(offsets) and offsets[idx][1] < offsets[idx+1][0]:
            pred_selected_text += " "

    pred_selected_text = remove_endding_link(pred_selected_text)
    if sentiment == 'negative':
        pred_selected_text = remove_name(pred_selected_text)
    return pred_selected_text


def training(data_loader, model, optimizer, device, scheduler):
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

        logits_start, logits_end = model(
            ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_function(logits_start, logits_end, targets_start,
                             targets_end)
        loss.backward()
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
            token_type_ids = data["token_type_ids"]
            original_selected_text = data['original_selected_text']
            original_tweet = data['original_tweet']
            sentiment = data['sentiment']
            targets_start = data['targets_start']
            targets_end = data['targets_end']
            offsets = data['offsets'].numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            logits_start, logits_end = model(
                ids=ids, mask=mask, token_type_ids=token_type_ids)

            prob_start = logits_start.cpu().detach().numpy()
            prob_end = logits_end.cpu().detach().numpy()

            # log softmax + faire un tableau avec somme des probs et chercher le max
            for idx, tweet in enumerate(original_tweet):
                selected_text = original_selected_text[idx]
                tweet_sentiment = sentiment[idx]
                p_start = prob_start[idx]
                p_end = prob_end[idx]
                p_ = np.argmax(p_start)
                predicted_selected_text = predict_selected_text(
                    original_tweet=tweet,
                    predicted_start=np.argmax(p_start),
                    predicted_end=p_ + np.argmax(p_end[p_:]),
                    sentiment=tweet_sentiment,
                    offsets=offsets[idx]
                )
                scores.append(jaccard(selected_text.strip(),
                                      predicted_selected_text))

    return np.mean(scores)


def predicting(data_loader, model, device):
    model.eval()
    final_output = []
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            ids = data['ids']
            mask = data['mask']
            token_type_ids = data["token_type_ids"]
            original_tweet = data['original_tweet']
            sentiment = data['sentiment']
            targets_start = data['targets_start']
            targets_end = data['targets_end']
            offsets = data['offsets'].numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            logits_start, logits_end = model(
                ids=ids, mask=mask, token_type_ids=token_type_ids)

            prob_start = logits_start.cpu().detach().numpy()
            prob_end = logits_end.cpu().detach().numpy()

            for idx, tweet in enumerate(original_tweet):
                tweet_sentiment = sentiment[idx]
                p_start = prob_start[idx]
                p_end = prob_end[idx]

                predicted_selected_text = predict_selected_text(
                    original_tweet=tweet,
                    predicted_start=np.argmax(p_start),
                    predicted_end=np.argmax(p_end),
                    sentiment=tweet_sentiment,
                    offsets=offsets[idx]
                )
                final_output.append(predicted_selected_text)
    return final_output
