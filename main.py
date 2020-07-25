import torch
import pandas as pd

from models import engine
from models.model import Transformer
from utils import config

from utils.seed import set_seed
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from data.dataset import TweetDataset
import numpy as np


def run(seed=42,
        lr=3e-5,
        bs=config.TRAIN_BATCH_SIZE,
        epoch=config.EPOCHS,
        threshold=.3,
        eps=.1):
    set_seed(seed)
    main_df = pd.read_csv(config.TRAINING_FILE)
    folds = main_df['kfold'].unique()
    scores = []
    for fold in sorted(folds):
        print(f'Fold {fold}')

        df_train = main_df[main_df['kfold'] != fold].reset_index(drop=True)
        df_valid = main_df[main_df['kfold'] == fold].reset_index(drop=True)

        train_dataset = TweetDataset(
            tweets=df_train['text'].values,
            selected_texts=df_train['selected_text'].values,
            sentiments=df_train['sentiment'].values,
            threshold=threshold)

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
            num_workers=6)

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("device: ", device)
        model = Transformer(nb_layers=2)
        model.to(device)

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
        scores.append(best_jaccard)
    print(f'Cross validation score: {np.mean(scores)} +/-{np.std(scores)}')


if __name__ == '__main__':
    run()
