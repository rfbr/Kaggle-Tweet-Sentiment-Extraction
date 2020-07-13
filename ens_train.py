import torch
import os
import pandas as pd
from models.model_2 import Transformer
#
from models.ensemble_model import EnsembleNet
#
from models import ensemble_engine
from utils import config
from utils import metric
from utils.freeze import freeze, unfreeze, freeze_layer, unfreeze_layer
from utils.seed import set_seed
from utils.adam_params import custom_params
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from data.dataset import TweetDataset
import numpy as np


def run():
    set_seed(42)
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
            threshold=0,
            denoise=0)

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=32,
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
        model = EnsembleNet()
        model.to(device)

        model_0 = Transformer(nb_layers=2)
        model_0.to(device)
        model_0.load_state_dict(torch.load(
            "../la_puissance/model_0_3.bin"))
        model_0.eval()

        model_1 = Transformer(nb_layers=2)
        model_1.to(device)
        model_1.load_state_dict(torch.load(
            "../la_puissance/model_1_3.bin"))
        model_1.eval()

        model_2 = Transformer(nb_layers=2)
        model_2.to(device)
        model_2.load_state_dict(torch.load(
            "../la_puissance/model_2_3.bin"))
        model_2.eval()

        model_3 = Transformer(nb_layers=2)
        model_3.to(device)
        model_3.load_state_dict(torch.load(
            "../la_puissance/model_3_3.bin"))
        model_3.eval()

        model_4 = Transformer(nb_layers=2)
        model_4.to(device)
        model_4.load_state_dict(torch.load(
            "../la_puissance/model_4_3.bin"))
        model_4.eval()

        ensemble = [model_0, model_1, model_2, model_3, model_4]
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

        num_train_steps = int(
            len(df_train) / 32 * 3)
        optimizer = AdamW(optimizer_parameters, lr=3e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

        for _ in range(3):
            ensemble_engine.training(train_data_loader, model, ensemble, optimizer, device,
                                     scheduler)
            jaccard = ensemble_engine.evaluating(
                valid_data_loader, model, ensemble, device)

            print(f'Jaccard validation score: {jaccard}')
            if jaccard > best_jaccard:
                best_jaccard = jaccard
        scores.append(best_jaccard)
    print(f'Cross validation score: {np.mean(scores)} +/-{np.std(scores)}')


if __name__ == '__main__':
    run()
