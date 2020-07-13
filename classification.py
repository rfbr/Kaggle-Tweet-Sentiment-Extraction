import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.seed import set_seed
from data.dataset import ClassDataset
from models.model import Transformer
import torch
import torch.nn as nn
from utils import config
from tqdm import tqdm
from utils.averagemeter import AverageMeter
from transformers import AdamW, get_linear_schedule_with_warmup
import os


def train(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = AverageMeter()
    tqdm_ = tqdm(data_loader, total=len(data_loader))
    for batch_idx, data in enumerate(tqdm_):
        ids = data['ids']
        mask = data['mask']
        sentiment = data['sentiment']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        sentiment = sentiment.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(ids=ids, mask=mask)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, sentiment)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        tqdm_.set_postfix(loss=losses.avg)
    torch.cuda.empty_cache()


def evaluate(data_loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(data_loader):
            ids = data['ids']
        mask = data['mask']
        sentiment = data['sentiment']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        sentiment = sentiment.to(device, dtype=torch.long)

        logits = model(ids=ids, mask=mask)
        _, predicted = torch.max(logits, 1)
        total += sentiment.size(0)
        correct += (predicted == sentiment).sum().item()
    return correct * 100 / total


if __name__ == '__main__':
    set_seed(42)
    df = pd.read_csv('../../data/TextEmotion.csv').dropna().reset_index(
        drop=True)
    df = df.drop(['tweet_id', 'author'], axis=1)
    df['content'] = df['content'].str.replace('@', '_')

    df_train, df_valid = train_test_split(df, test_size=.1)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    df_train = ClassDataset(tweets=df_train['content'],
                            sentiments=df_train['sentiment'])
    df_train = DataLoader(df_train, shuffle=True, batch_size=32, num_workers=6)

    df_valid = ClassDataset(tweets=df_valid['content'],
                            sentiments=df_valid['sentiment'])
    df_valid = DataLoader(df_valid,
                          shuffle=False,
                          batch_size=16,
                          num_workers=6)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device: ", device)
    model = Transformer(nb_layers=2)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.001
        },
        {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        },
    ]
    num_train_steps = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    best_acc = 0
    for epoch in range(config.EPOCHS):
        train(df_train, model, optimizer, device, scheduler)
        acc = evaluate(df_valid, model, device)
        print(f'Acc validation score: {acc}')
        if acc > best_acc:
            torch.save(
                model.state_dict(),
                os.path.join(config.SAVED_MODEL_PATH, f'class_model.bin'))
            best_acc = acc
