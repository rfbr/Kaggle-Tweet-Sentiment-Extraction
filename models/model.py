from utils import config
import transformers
import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        conf = transformers.RobertaConfig.from_pretrained(
            config.ROBERTA_MODEL)
        conf.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(
            config.ROBERTA_MODEL, config=conf)
        # self.drop_1 = nn.Dropout(.1)
        self.l1 = nn.Linear(768*2, 2)

    def forward(self, ids, mask, token_type_ids):
        _, _, hidden_states = self.roberta(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        x = torch.cat((hidden_states[-1], hidden_states[2]), dim=-1)
        x = self.drop_1(x)
        logits = self.l1(x)
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
