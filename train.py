import torch
import os
import pandas as pd
from models.model import Net
from models.model_2 import Transformer
from models import engine
from utils import config
from utils.freeze import freeze, unfreeze, freeze_layer, unfreeze_layer
from utils.seed import set_seed
from utils.adam_params import custom_params
from sklearn.model_selection import train_test_split
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from data.dataset import TweetDataset


def set_up_optimizer_scheduler(model,
                               num_training_steps,
                               weight_decay,
                               num_warmup_steps=0,
                               warmup_prop=0, lr_transfo=1e-3,
                               lr=5e-4, lr_decay=1):
    opt_params = custom_params(
        model, lr=lr, weight_decay=weight_decay,
        lr_transfo=lr_transfo, lr_decay=lr_decay)
    optimizer = AdamW(opt_params, lr=lr, betas=(.5, 0.999))
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_training_steps=num_training_steps,
                                                num_warmup_steps=num_training_steps)
    return optimizer, scheduler


def run():
    set_seed(2020)
    main_df = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop=True)

    df_train, df_valid = train_test_split(main_df,
                                          test_size=.1)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = TweetDataset(
        tweets=df_train['text'].values,
        selected_texts=df_train['selected_text'].values,
        sentiments=df_train['sentiment'].values)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    shuffle=True,
                                                    batch_size=config.TRAIN_BATCH_SIZE,
                                                    num_workers=6)

    valid_dataset = TweetDataset(
        tweets=df_valid['text'].values,
        selected_texts=df_valid['selected_text'].values,
        sentiments=df_valid['sentiment'].values)

    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    shuffle=False,
                                                    batch_size=config.VALID_BATCH_SIZE,
                                                    num_workers=1)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device: ", device)
    model = Net()
    model = Transformer(nb_layers=2)
    print(model)
    model.to(device)

    num_training_steps = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    # best_jaccard = 0

    freeze(model)
    for layer in ['logit', 'pooler', 'intermediate_1', 'intermediate_2']:
        unfreeze_layer(model, layer)

    weight_decay = 0
    epochs = 3
    lr = 1e-3
    num_training_steps = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * epochs)
    optimizer, scheduler = set_up_optimizer_scheduler(model,
                                                      num_training_steps,
                                                      weight_decay,
                                                      lr=lr)
    for epoch in range(epochs):
        engine.training(train_data_loader, model,
                        optimizer, device, scheduler)
        jaccard = engine.evaluating(valid_data_loader, model, device)
        print(f'Jaccard validation score: {jaccard}')

    unfreeze(model)

    epochs = 2
    num_training_steps = int(
        len(df_train) / 32 * epochs)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    shuffle=True,
                                                    batch_size=32,
                                                    num_workers=12)

    lr_transfo = 3e-5
    lr = 1e-4
    lr_decay = 0.975
    optimizer, scheduler = set_up_optimizer_scheduler(model,
                                                      num_training_steps,
                                                      weight_decay,
                                                      lr_transfo=lr_transfo,
                                                      lr=lr,
                                                      lr_decay=lr_decay)
    for epoch in range(epochs):
        engine.training(train_data_loader, model,
                        optimizer, device, scheduler)
        jaccard = engine.evaluating(valid_data_loader, model, device)
        print(f'Jaccard validation score: {jaccard}')
    torch.save(model.state_dict(),
               os.path.join(config.SAVED_MODEL_PATH,
                            'new_model.bin'))


if __name__ == '__main__':
    run()
