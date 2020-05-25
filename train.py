import torch
import os
import pandas as pd
from models.model import Net
from models.model_2 import Transformer
#
from models.model_beam_search import Transformer_beam
#
from models import engine
from utils import config
from utils import metric
from utils.freeze import freeze, unfreeze, freeze_layer, unfreeze_layer
from utils.seed import set_seed
from utils.adam_params import custom_params
from sklearn.model_selection import train_test_split
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from data.dataset import TweetDataset
import numpy as np


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
                                                num_warmup_steps=num_warmup_steps)
    return optimizer, scheduler


def run():
    set_seed(42)
    main_df = pd.read_csv(config.TRAINING_FILE)
    folds = main_df['fold'].unique()
    scores = []
    for fold in folds:
        print(f'Fold {fold}')
        df_train = main_df[main_df['fold'] != fold].reset_index(drop=True)
        df_train = df_train[df_train['sentiment'].isin(
            ['positive', 'negative'])]
        df_valid = main_df[main_df['fold'] == fold].reset_index(drop=True)

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
        # model = Transformer(nb_layers=2)
        #
        model = Transformer_beam()
        #
        model.to(device)

        best_jaccard = 0
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        num_train_steps = int(
            len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
        optimizer = AdamW(optimizer_parameters, lr=5e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )
        for epoch in range(config.EPOCHS):
            engine.training(train_data_loader, model,
                            optimizer, device, scheduler)
            # jaccard = engine.evaluating(valid_data_loader, model, device)
            #
            jaccard = engine.evaluating_beam(valid_data_loader, model, device)
            #
            print(f'Jaccard validation score: {jaccard}')
            if jaccard > best_jaccard:
                torch.save(model.state_dict(),
                           os.path.join(config.SAVED_MODEL_PATH,
                                        f'model_{fold}_gru_bs.bin'))
                best_jaccard = jaccard
        # freeze(model)
        # for layer in ['logit', 'pooler', 'intermediate_1', 'intermediate_2']:
        #     unfreeze_layer(model, layer)

        # weight_decay = 0
        # epochs = 4
        # lr = 5e-3
        # num_training_steps = int(
        #     len(df_train) / config.TRAIN_BATCH_SIZE * epochs)
        # optimizer, scheduler = set_up_optimizer_scheduler(model,
        #                                                   num_training_steps,
        #                                                   weight_decay,
        #                                                   lr=lr)
        # for epoch in range(epochs):
        #     engine.training(train_data_loader, model,
        #                     optimizer, device, scheduler)
        #     jaccard = engine.evaluating(valid_data_loader, model, device)
        #     print(f'Jaccard validation score: {jaccard}')

        # unfreeze(model)

        # epochs = 4
        # num_training_steps = int(
        #     len(df_train) / 32 * epochs)
        # train_data_loader = torch.utils.data.DataLoader(train_dataset,
        #                                                 shuffle=True,
        #                                                 batch_size=32,
        #                                                 num_workers=12)

        # lr_transfo = 3e-5
        # lr = 1e-4
        # lr_decay = 0.95
        # optimizer, scheduler = set_up_optimizer_scheduler(model,
        #                                                   num_training_steps,
        #                                                   weight_decay,
        #                                                   lr_transfo=lr_transfo,
        #                                                   lr=lr,
        #                                                   lr_decay=lr_decay)
        # for epoch in range(epochs):
        #     engine.training(train_data_loader, model,
        #                     optimizer, device, scheduler)
        #     jaccard = engine.evaluating(valid_data_loader, model, device)
        #     print(f'Jaccard validation score: {jaccard}')
        #     if jaccard > best_jaccard:
        #         torch.save(model.state_dict(),
        #                    os.path.join(config.SAVED_MODEL_PATH,
        #                                 f'model_{fold}.bin'))
        #         best_jaccard = jaccard
        scores.append(best_jaccard)
    print(f'Cross validation score: {np.mean(scores)} +/-{np.std(scores)}')


if __name__ == '__main__':
    run()
