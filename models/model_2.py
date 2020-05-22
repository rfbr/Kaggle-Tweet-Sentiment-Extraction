import torch.nn as nn
import torch
import transformers
from utils import config
from torch.autograd import Variable
import torch.nn.functional as F


class KimCNN(nn.Module):
    def __init__(self, embed_num, embed_dim, kernel_num, kernel_sizes, dropout, static):
        super(KimCNN, self).__init__()
        V = embed_num
        D = embed_dim
        Co = kernel_num
        Ks = kernel_sizes

        self.static = static
        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, config.MAX_LEN)

    def forward(self, x):

        if self.static:
            x = Variable(x)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.ea(conv(x)).squeeze(3)
             for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2)
             for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        output = self.fc1(x)  # (N, C)
        return output


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
            pooler = nn.Sequential(
                nn.Linear(input_size, output_size), nn.ReLU())
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

        for i, (state) in enumerate(hidden_states[: self.nb_layers]):
            pooled = self.poolers[i](state)
            outputs.append(pooled)

        return torch.cat(outputs, -1)


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
        conf = transformers.RobertaConfig.from_pretrained(
            config.ROBERTA_MODEL)
        conf.output_hidden_states = True
        self.transformer = transformers.RobertaModel.from_pretrained(
            config.ROBERTA_MODEL, config=conf)
        self.nb_layers = nb_layers

        self.nb_features = self.transformer.pooler.dense.out_features

        if pooler_ft is None:
            pooler_ft = self.nb_features

        if nb_layers != 1:
            self.pooler = RoBERTaMultiPooler(
                nb_layers=nb_layers,
                input_size=self.nb_features,
                output_size=pooler_ft)
        else:
            self.pooler = nn.Sequential(
                # nn.Dropout(.1),
                nn.Linear(self.nb_features, pooler_ft),
                nn.ReLU(),
            )
        self.drop_1 = nn.Dropout(.15)
        self.drop_2 = nn.Dropout(.15)
        self.intermediate = nn.Linear(nb_layers*pooler_ft, nb_layers*pooler_ft)
        self.logit = nn.Linear(nb_layers*pooler_ft, 2)

    def forward(self, ids, mask, token_type_ids):
        """
        Usual torch forward function

        Arguments:
            tokens {torch tensor} -- Sentence tokens

        Returns:
            torch tensor -- Class logits
            torch tensor -- Pooled features
        """
        _, _, hidden_states = self.transformer(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        hidden_states = hidden_states[::-1]

        x = self.pooler(hidden_states)
        x = self.drop_1(x)
        res = x
        x = self.intermediate(x)
        x = self.drop_2(x + res)
        logits = self.logit(x + res)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
