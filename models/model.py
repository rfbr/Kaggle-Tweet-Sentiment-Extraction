import torch
import torch.nn as nn

import transformers
from utils import config


class RoBERTaMultiPooler(nn.Module):
    """
    Custom RoBERTa Pooling head that takes the n last layers.
    """

    def __init__(self, nb_layers=1, input_size=768, output_size=1536):
        """
        Constructor.

        Parameters:
            nb_layers {int}: number of layers to consider (default: {1}).
            input_size {int}: size of the input features (default: {768}).
            output_size {int}: size of the output features (default: {1536}).
        """
        super(RoBERTaMultiPooler, self).__init__()

        self.nb_layers = nb_layers
        self.input_size = input_size
        self.poolers = nn.ModuleList([])
        for _ in range(nb_layers):
            pooler = nn.LSTM(input_size,
                             hidden_size=output_size//(2*nb_layers),
                             batch_first=True,
                             bidirectional=True,
                             num_layers=2)
            self.poolers.append(pooler)

    def forward(self, hidden_states):
        """
        Usual torch forward function.

        Parameters:
            hidden_states {list of torch tensors}: hidden states of the model, the last one being at index 0.

        Returns:
            torch tensor: pooled features.
            torch tensor: hidden states of the considered layers.
        """
        outputs = []
        hs = []
        for i, (state) in enumerate(hidden_states[:self.nb_layers]):
            pooled = torch.relu(self.poolers[i](state)[0])
            outputs.append(pooled)
            hs.append(state)
        return torch.cat(outputs, -1), torch.cat(hs, -1)


class Transformer(nn.Module):
    def __init__(self, nb_layers=1, dropout=.1, pooler_ft=1536):
        """
        Constructor.

        Parameters:
            nb_layers {int}: number of layers to consider for the pooler (default: {1})
            dropout {float}: dropout rate.
            pooler_ft {int}: number of features for the pooler (default: {1536}).
        """
        super(Transformer, self).__init__()
        conf = transformers.RobertaConfig.from_pretrained(config.ROBERTA_MODEL)
        conf.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(
            config.ROBERTA_MODEL, config=conf)
        self.nb_layers = nb_layers

        self.nb_features = self.roberta.pooler.dense.out_features

        self.pooler = RoBERTaMultiPooler(nb_layers=nb_layers,
                                         input_size=self.nb_features,
                                         output_size=pooler_ft)
        self.intermediate_1 = nn.Linear(pooler_ft,
                                        pooler_ft)
        self.intermediate_2 = nn.Linear(pooler_ft, pooler_ft//2)
        self.drop_1 = nn.Dropout(dropout)
        self.drop_2 = nn.Dropout(dropout)
        self.logits = nn.Linear(pooler_ft//2, 2)

    def forward(self, ids, mask):
        """
        Usual torch forward function.

        Parameters:
            ids {torch tensor}: indices of input sequence tokens in the vocabulary.
            mask {torch tensor}: mask to avoid performing attention on padding token indices.

        Returns:
            start_logits {torch tensor}: start logits.
            end_logits {torch tensor}: end logits.
        """
        _, _, hidden_states = self.roberta(input_ids=ids, attention_mask=mask)
        hidden_states = hidden_states[::-1]
        x, hs = self.pooler(hidden_states)
        x = self.drop_1(x)
        x = torch.tanh(self.intermediate_1(x) + hs)
        x = torch.tanh(self.intermediate_2(x))
        x = self.drop_2(x)
        logits = self.logits(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits
