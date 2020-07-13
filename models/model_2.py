import torch.nn as nn
import torch
import transformers
from utils import config
from torch.autograd import Variable
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super(EncoderBlock, self).__init__()
        self.intermediate_1 = nn.Linear(n_in,
                                        n_in)
        self.intermediate_2 = nn.Linear(n_in,
                                        n_out)

    def forward(self, x, hs=None):
        if hs is None:
            x = torch.relu(self.intermediate_1(x))
        else:
            x = torch.relu(self.intermediate_1(x)+hs)
        x = torch.relu(self.intermediate_2(x))
        return x


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
                             num_layers=1)
            # for layer_p in pooler._all_weights:
            #     for p in layer_p:
            #         if 'weight' in p:
            #             torch.nn.init.xavier_normal_(pooler.__getattr__(p))
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
            pooled = self.poolers[i](state)[0]
            outputs.append(pooled)
            hs.append(state)
        return torch.cat(outputs, -1), torch.cat(hs, -1)


class CNNMultiPooler(nn.Module):
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
        super(CNNMultiPooler, self).__init__()

        self.nb_layers = nb_layers
        self.input_size = input_size
        self.poolers = nn.ModuleList([])
        for i in range(nb_layers):
            pooler = nn.Sequential(
                nn.Conv1d(768, 768, 2),
                nn.BatchNorm1d(768),
                nn.Tanh(),
                nn.Conv1d(768, 768, 1),
                nn.BatchNorm1d(768),
                nn.Tanh())
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
            t_state = torch.nn.functional.pad(state.transpose(1, 2), (1, 0))
            pooled = self.poolers[i](t_state).transpose(1, 2)
            outputs.append(pooled)
            hs.append(state)
        return torch.cat(outputs, -1), torch.cat(hs, -1)


class Transformer(nn.Module):
    def __init__(self, nb_layers=1, dropout=.1, pooler_ft=None):
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

        self.pooler = RoBERTaMultiPooler(nb_layers=nb_layers,
                                         input_size=self.nb_features,
                                         output_size=pooler_ft)
        self.intermediate_1 = nn.Linear(nb_layers * pooler_ft,
                                        nb_layers * pooler_ft)
        self.intermediate_2 = nn.Linear(nb_layers * pooler_ft, pooler_ft)
        self.drop_1 = nn.Dropout(dropout)
        self.drop_2 = nn.Dropout(dropout)
        self.logits = nn.Linear(pooler_ft, 2)
        # nn.init.normal_(self.intermediate_1.weight, std=1)
        # nn.init.normal_(self.intermediate_1.bias, 0)
        # nn.init.normal_(self.intermediate_2.weight, std=1)
        # nn.init.normal_(self.intermediate_2.bias, 0)
        # nn.init.normal_(self.logits.weight, std=1)
        # nn.init.normal_(self.logits.bias, 0)
        # self.lr1 = nn.ReLU()
        # self.lr2 = nn.ReLU()
        # self.block_1 = EncoderBlock(n_in=2*768, n_out=2*768)
        # self.block_2 = EncoderBlock(n_in=2*768, n_out=2*768)
        # self.block_3 = EncoderBlock(n_in=2*768, n_out=2*768)
        # self.drop_1 = nn.Dropout(.1)
        # self.drop_2 = nn.Dropout(.1)
        # self.drop_3 = nn.Dropout(.1)
        # self.start_logits = nn.Linear(4*768, 1)
        # self.end_logits = nn.Linear(4*768, 1)

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

        # hs = self.drop_3(
        #     torch.cat((hidden_states[0], hidden_states[1]), dim=-1))
        # block_1 = self.block_1(hs)
        # block_2 = self.block_2(self.drop_1(block_1), hs)
        # block_3 = self.block_3(self.drop_2(block_2), hs)

        # start_cat = torch.cat((block_1, block_2), -1)
        # end_cat = torch.cat((block_1, block_3), -1)
        # start_logits = self.start_logits(start_cat)
        # end_logits = self.end_logits(end_cat)

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
