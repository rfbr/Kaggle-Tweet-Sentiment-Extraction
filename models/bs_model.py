import torch.nn as nn
import transformers
from utils import config
import torch


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


class StartTransformer(nn.Module):
    def __init__(self):
        """
        Constructor

        Arguments:
            model {string} -- Transformer to build the model on. Expects "camembert-base".

        Keyword Arguments:
            nb_layers {int} -- Number of layers to consider for the pooler (default: {1})
            pooler_ft {[type]} -- Number of features for the pooler. If None, use the same number as the transformer (default: {None})
            avg_pool {bool} -- Whether to use average pooling instead of pooling on the first tensor (default: {False})
        """
        super(StartTransformer, self).__init__()
        conf = transformers.RobertaConfig.from_pretrained(config.ROBERTA_MODEL)
        conf.output_hidden_states = True
        self.transformer = transformers.RobertaModel.from_pretrained(
            config.ROBERTA_MODEL, config=conf)
        self.pooler = RoBERTaMultiPooler(nb_layers=2,
                                         input_size=768,
                                         output_size=384)

        self.intermediate_1 = nn.Linear(2 * 768, 2 * 768)
        self.intermediate_2 = nn.Linear(2 * 768, 768)
        self.logit = nn.Linear(768, 1)

    def forward(self, ids, mask):
        _, _, hidden_states = self.transformer(input_ids=ids,
                                               attention_mask=mask)
        hidden_states = hidden_states[::-1]
        last_emb = hidden_states[0]
        x, hs = self.pooler(hidden_states)
        x = torch.relu(self.intermediate_1(x) + hs)
        x = torch.relu(self.intermediate_2(x))
        start_logits = self.logit(x).squeeze(-1)
        return start_logits, last_emb


class EndTransformer(nn.Module):
    def __init__(self):
        super(EndTransformer, self).__init__()
        conf = transformers.RobertaConfig.from_pretrained(config.ROBERTA_MODEL)
        conf.output_hidden_states = True
        self.transformer = transformers.RobertaModel.from_pretrained(
            config.ROBERTA_MODEL, config=conf)
        self.end_token_gru = nn.GRU(768,
                                    768,
                                    batch_first=True,
                                    bidirectional=True)
        self.end_token_logits = nn.Sequential(nn.Linear(2*768, 768),
                                              nn.ReLU(), nn.Linear(768, 1))

    def forward(self, start_embedding, ids, mask):
        _, _, hidden_states = self.transformer(input_ids=ids,
                                               attention_mask=mask)
        embeddings = hidden_states[-1]
        outputs, _ = self.end_token_gru(embeddings, start_embedding)
        end_logits = self.end_token_logits(outputs).squeeze(-1)
        return end_logits
