import torch
import os
import pandas as pd
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
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from data.dataset import TweetDataset, CorrTweetDataset
import numpy as np
from models.bs_model import StartTransformer, EndTransformer


def set_up_optimizer_scheduler(model,
                               num_training_steps,
                               weight_decay,
                               num_warmup_steps=0,
                               warmup_prop=0,
                               lr_transfo=1e-3,
                               lr=5e-4,
                               lr_decay=1):
    opt_params = custom_params(model,
                               lr=lr,
                               weight_decay=weight_decay,
                               lr_transfo=lr_transfo,
                               lr_decay=lr_decay)
    optimizer = AdamW(opt_params, lr=lr, betas=(.5, 0.999))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps)
    return optimizer, scheduler


def my_collate(batch):
    batch_dic = {}
    keys = batch[0].keys()
    for key in keys:
        batch_dic[key] = []
    for item in batch:
        for key in keys:
            batch_dic[key].append(item[key])
    return batch_dic


def run(seed=42,
        lr=8e-5,
        bs=config.TRAIN_BATCH_SIZE,
        epoch=config.EPOCHS,
        threshold=.3,
        denoise=.5,
        dropout=.1,
        eps=.1):
    set_seed(seed)
    main_df = pd.read_csv(config.TRAINING_FILE)
    folds = main_df['kfold'].unique()
    scores = []
    for fold in folds:
        print(f'Fold {fold}')
        df_train = main_df[main_df['kfold'] != fold].reset_index(drop=True)
        df_valid = main_df[main_df['kfold'] == fold].reset_index(drop=True)
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        df_valid = df_valid.sample(frac=1).reset_index(drop=True)
        train_dataset = TweetDataset(
            tweets=df_train['text'].values,
            selected_texts=df_train['selected_text'].values,
            sentiments=df_train['sentiment'].values,
            threshold=threshold,
            denoise=denoise)

        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        shuffle=True,
                                                        batch_size=bs,
                                                        num_workers=6)

        valid_dataset = TweetDataset(
            tweets=df_valid['text'].values,
            selected_texts=df_valid['selected_text'].values,
            sentiments=df_valid['sentiment'].values,
            threshold=0)

        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            shuffle=False,
            batch_size=config.VALID_BATCH_SIZE,
            num_workers=3)
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("device: ", device)
        model = Transformer(nb_layers=2, dropout=dropout)
        model.to(device)

        #
        # model_start = StartTransformer()
        # model_end = EndTransformer()
        #
        # model_start.to(device)
        # model_end.to(device)
        # model.load_state_dict(torch.load("../saved_models/class_model.bin"),
        #                       strict=False)

        best_jaccard = 0
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
                'params': [
                    p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.0
            },
        ]

        num_train_steps = int(len(df_train) / bs * epoch)
        optimizer = AdamW(optimizer_parameters, lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

        for _ in range(epoch):
            engine.training(train_data_loader, model, optimizer, device,
                            scheduler, eps)
            jaccard = engine.evaluating(valid_data_loader, model, device)

            print(f'Jaccard validation score: {jaccard}')
            if jaccard > best_jaccard:
                # torch.save(
                #     model.state_dict(),
                #     os.path.join(config.SAVED_MODEL_PATH, f'model_{fold}.bin'))
                best_jaccard = jaccard
        # param_optimizer = list(model.named_parameters())
        # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        # optimizer_parameters = [
        #     {
        #         'params': [
        #             p for n, p in param_optimizer
        #             if not any(nd in n for nd in no_decay)
        #         ],
        #         'weight_decay':
        #         0.001
        #     },
        #     {
        #         'params': [
        #             p for n, p in param_optimizer
        #             if any(nd in n for nd in no_decay)
        #         ],
        #         'weight_decay':
        #         0.0
        #     },
        # ]

        # num_train_steps = int(
        #     len(df_train) / bs * epoch)
        # optimizer = AdamW(optimizer_parameters, lr=lr)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

        # freeze(model)
        # for layer in ['logits', 'pooler', 'intermediate_1', 'intermediate_2', 'drop_1', 'drop_2']:
        #     unfreeze_layer(model, layer)
        # for _ in range(epoch):
        #     engine.training(train_data_loader, model,
        #                     optimizer, device, scheduler)
        #     jaccard = engine.evaluating(valid_data_loader, model, device)

        #     print(f'Jaccard validation score: {jaccard}')
        #     if jaccard > best_jaccard:
        #         # torch.save(
        #         #     model.state_dict(),
        #         #     os.path.join(config.SAVED_MODEL_PATH, f'model_{fold}.bin'))
        #         best_jaccard = jaccard
        # unfreeze(model)
        # weight_decay = 0
        # epochs = 4
        # lr = 5e-4
        # num_training_steps = int(
        #     len(df_train) / config.TRAIN_BATCH_SIZE * epochs)
        # optimizer, scheduler = set_up_optimizer_scheduler(model,
        #                                                   num_training_steps,
        #                                                   weight_decay,
        #                                                   lr=lr)
        # for epoch in range(epochs):
        #     engine.training(train_data_loader, model, optimizer, device,
        #                     scheduler)
        #     jaccard = engine.evaluating(valid_data_loader, model, device)
        #     print(f'Jaccard validation score: {jaccard}')

        # unfreeze(model)

        # epochs = 3
        # num_training_steps = int(len(df_train) / 64 * epochs)
        # train_data_loader = torch.utils.data.DataLoader(train_dataset,
        #                                                 shuffle=True,
        #                                                 batch_size=64,
        #                                                 num_workers=12)
        # weight_decay = 0.001
        # lr_transfo = 8e-5
        # lr = 0.001
        # lr_decay = 1
        # optimizer, scheduler = set_up_optimizer_scheduler(
        #     model,
        #     num_training_steps,
        #     weight_decay,
        #     lr_transfo=lr_transfo,
        #     lr=lr,
        #     lr_decay=lr_decay)
        # for epoch in range(epochs):
        #     engine.training(train_data_loader, model, optimizer, device,
        #                     scheduler)
        #     jaccard = engine.evaluating(valid_data_loader, model, device)
        #     print(f'Jaccard validation score: {jaccard}')
        #     # if jaccard > best_jaccard:
        #     #     torch.save(model.state_dict(),
        #     #                os.path.join(config.SAVED_MODEL_PATH,
        #     #                             f'model_{fold}.bin'))
        #     # best_jaccard = jaccard
        scores.append(best_jaccard)
    print(f'Cross validation score: {np.mean(scores)} +/-{np.std(scores)}')


if __name__ == '__main__':
    epochs = [2, 3, 4, 5]
    lrs = [1.5e-5, 2e-5, 2.5e-5, 3e-5, 3.5e-5]
    # for lr in lrs:
    #     for epoch in epochs:
    #         print("LR: ", lr)
    #         print("Epoch: ", epoch)
    #         run(seed=42, lr=lr, bs=32,
    #             epoch=epoch)
    # seeds = np.random.randint(low=42, high=10e3, size=8)
    # for seed in seeds:
    #     print(seed)
    #     run(seed=seed, lr=3e-5, bs=32, epoch=3)
    augs = [.3]
    denoises = [.3]
    eps = [.2, .3, .4]
    dropouts = [.1, .2, .3, .4]
    for ep in eps:
        for dropout in dropouts:
            print(f'Eps: {ep}')
            print(f'Dropout: {dropout}')
            run(lr=3e-5,
                bs=32,
                epoch=3,
                threshold=.3,
                denoise=.3,
                dropout=dropout,
                eps=ep)
