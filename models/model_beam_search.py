import torch.nn as nn
import torch
import transformers
from utils import config
from torch.autograd import Variable
import torch.nn.functional as F

class Transformer_beam(nn.Module):
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
        super(Transformer_beam, self).__init__()
        conf = transformers.RobertaConfig.from_pretrained(
            config.ROBERTA_MODEL)
        conf.output_hidden_states = True
        self.transformer = transformers.RobertaModel.from_pretrained(
            config.ROBERTA_MODEL, config=conf)

        self.start_token = nn.Sequential(nn.Linear(768, 768), nn.Dropout(.1), nn.ReLU(), nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 1))

        # self.end_token = nn.Sequential(nn.Linear(2 * 768, 768), nn.Dropout(.1), nn.Tanh(), nn.LayerNorm(768), nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 1)) # , nn.ReLU(), nn.Linear(768, 1))

        self.end_token_gru = nn.GRU(768, 768, batch_first=True, bidirectional=True)
        self.end_token_logits = nn.Sequential(nn.Linear(2 * 768, 768), nn.ReLU(), nn.Linear(768, 1))

    def forward(self, ids, mask, token_type_ids, start_positions=None, beam_size_start=5, beam_size_end=5):
        """
        Usual torch forward function

        Arguments:
            tokens {torch tensor} -- Sentence tokens

        Returns:
            torch tensor -- Class logits
            torch tensor -- Pooled features
        """
        embeddings, _, layers = self.transformer(input_ids=ids, attention_mask=mask)
        
        # embeddings = torch.cat((layers[-1], layers[-2]), dim=2)
        # beam search 
        # x = 1e30
        slen, hsz = embeddings.shape[-2:]
        start_scores = self.start_token(embeddings).squeeze() 
        
        if start_positions is not None: # TRAINING

            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz) -> start position repeated hsz times along the last axis
            start_embs = embeddings.gather(-2, start_positions).squeeze()  # shape (bsz, 1, hsz) -> retrieve embedding of the start position for each elt in batch
            h_0 = torch.cat([start_embs.unsqueeze(0)] * 2, dim=0)
            outputs, _ = self.end_token_gru(embeddings, h_0)
            end_scores = self.end_token_logits(outputs).squeeze()
            return start_scores, end_scores 

        else: #TESTING
            # during inference, compute the end logits based on beam search 
            bsz, slen, hsz = embeddings.size()
            start_scores = F.log_softmax(start_scores, dim=-1)
            start_top_scores, start_top_index = torch.topk(start_scores, beam_size_start, dim=-1)  # shape (bsz, beam_size_start)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, beam_size_start, hsz)
            start_embs = embeddings.gather(-2, start_top_index_exp)  # shape (bsz, beam_size_start, hsz)

            
            end_scores = []
            for i in range(beam_size_start):
                h_0 = torch.cat([start_embs[:, i, :].unsqueeze(0)] * 2, dim=0)
                outputs, _ = self.end_token_gru(embeddings, h_0)
                end_scores.append(self.end_token_logits(outputs))      

            end_scores = torch.cat(end_scores, dim=2)
            end_scores = F.log_softmax(end_scores, dim=1)  # shape (bsz, slen, beam_size_start)
            end_top_scores, end_top_index = torch.topk(end_scores, beam_size_end, dim=1)  # shape (bsz, end_n_top, start_n_top)
            end_top_scores = end_top_scores.view(-1, beam_size_end * beam_size_start)
            end_top_index = end_top_index.view(-1, beam_size_end * beam_size_start)
            
            return start_top_scores, start_top_index, end_top_scores, end_top_index
