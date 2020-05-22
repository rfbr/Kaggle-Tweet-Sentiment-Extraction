import torch
import pandas as pd
from models.model import Net
from data.dataset import TweetDataset
from utils import config
from models.engine import predicting
if __name__ == '__main__':
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Net()
    model.to(device)
    model.load_state_dict(torch.load("../saved_models/best_model.bin"))

    df_test = pd.read_csv(config.TEST_FILE)
    df_test['selected_text'] = df_test['text'].values
    test_dataset = TweetDataset(
        tweets=df_test['text'],
        selected_texts=df_test['selected_text'],
        sentiments=df_test['sentiment']
    )

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1
    )
    predictions = predicting(data_loader, model, device)
    submission = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
    submission['selected_text'] = predictions
    submission.to_csv('submission.csv', index=False)
