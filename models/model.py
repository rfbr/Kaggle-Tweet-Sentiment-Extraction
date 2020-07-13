import torch.nn as nn
import torch
import transformers
from utils import config
from torch.autograd import Variable
import torch.nn.functional as F


class RoBERTaMultiPooler(nn.Module):
    """
    Custom RoBERTa Pooling head that takes the n last layers
    """
    def __init__(self, nb_layers=1, input_size=768, output_size=768):
        """
        Constructor

        Arguments:
            nb_layers {int} -- Number of layers to consider (default: {1})
            input_size {int} -- Size of the input features (default: {768})
            output_size {int} -- Size of the output features (default: {768})
        """
        super(RoBERTaMultiPooler, self).__init__()

        self.nb_layers = nb_layers
        self.input_size = input_size
        self.poolers = nn.ModuleList([])
        for i in range(nb_layers):
            pooler = nn.LSTM(input_size,
                             hidden_size=384,
                             batch_first=True,
                             bidirectional=True,
                             num_layers=2)
            # pooler = nn.Sequential(
            #     nn.Linear(input_size, output_size), nn.ReLU())
            self.poolers.append(pooler)

    def forward(self, hidden_states):
        """
        Usual torch forward function

        Arguments:
            hidden_states {list of torch tensors} -- Hidden states of the model, the last one being at index 0

        Returns:
            torch tensor -- Pooled features
        """
        outputs = []
        hs = []
        for i, (state) in enumerate(hidden_states[:self.nb_layers]):
            # pooled = self.poolers[i](state)
            pooled = torch.relu(self.poolers[i](state)[0])
            outputs.append(pooled)
            hs.append(state)
        return torch.cat(outputs, -1), torch.cat(hs, -1)


class Transformer(nn.Module):
    def __init__(self, nb_layers=1, pooler_ft=None):
        """
        Constructor

        Arguments:
            model {string} -- Transformer to build the model on. Expects "camembert-base".

        Keyword Arguments:
            nb_layers {int} -- Number of layers to consider for the pooler (default: {1})
            pooler_ft {[type]} -- Number of features for the pooler. If None, use the same number as the transformer (default: {None})
            avg_pool {bool} -- Whether to use average pooling instead of pooling on the first tensor (default: {False})
        """
        super(Transformer, self).__init__()
        conf = transformers.RobertaConfig.from_pretrained(config.ROBERTA_MODEL)
        conf.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(
            config.ROBERTA_MODEL, config=conf)
        self.nb_layers = nb_layers

        self.nb_features = self.roberta.pooler.dense.out_features

        if pooler_ft is None:
            pooler_ft = self.nb_features

        if nb_layers != 1:
            self.pooler = RoBERTaMultiPooler(nb_layers=nb_layers,
                                             input_size=self.nb_features,
                                             output_size=pooler_ft)
        else:
            self.pooler = nn.Sequential(
                # nn.Dropout(.1),
                nn.Linear(self.nb_features, pooler_ft),
                nn.ReLU())

        self.intermediate_1 = nn.Linear(nb_layers * pooler_ft,
                                        nb_layers * pooler_ft)
        self.intermediate_2 = nn.Linear(nb_layers * pooler_ft, pooler_ft)
        self.class_logit = nn.Linear(pooler_ft, 3)

    def forward(self, ids, mask):
        """
        Usual torch forward function

        Arguments:
            tokens {torch tensor} -- Sentence tokens

        Returns:
            torch tensor -- Class logits
            torch tensor -- Pooled features
        """
        _, _, hidden_states = self.roberta(input_ids=ids, attention_mask=mask)
        hidden_states = hidden_states[::-1]
        x, hs = self.pooler(hidden_states)
        # x = self.drop_1(x)
        x = torch.relu(self.intermediate_1(x) + hs)
        # x = self.drop_2(x)
        x = torch.relu(self.intermediate_2(x))
        # x = self.drop_3(x)
        logits = self.class_logit(x.mean(1))

        return logits
