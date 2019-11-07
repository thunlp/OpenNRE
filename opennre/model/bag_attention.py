import torch
from torch import nn, optim
from .base_model import BagRE

class BagAttention(BagRE):
    """
    Instance attention for bag-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, bag):
        """
        Args:
            bag: bag of sentences with the same entity pair
                [{
                  'text' or 'token': ..., 
                  'h': {'pos': [start, end], ...}, 
                  't': {'pos': [start, end], ...}
                }]
        Return:
            (relation, score)
        """
        self.eval()
        tokens = []
        pos1s = []
        pos2s = []
        masks = []
        for item in bag:
            if 'text' in item:
                token, pos1, pos2, mask = self.tokenizer(item['text'], 
                    item['h']['pos'], item['t']['pos'], is_token=False, padding=True)
            else:
                token, pos1, pos2, mask = self.tokenizer(item['token'], 
                    item['h']['pos'], item['t']['pos'], is_token=True, padding=True)
            tokens.append(token)
            pos1s.append(pos1)
            pos2s.append(pos2)
            masks.append(mask)
        tokens = torch.cat(tokens, 0) # (n, L)
        pos1s = torch.cat(pos1s, 0)
        pos2s = torch.cat(pos2s, 0)
        masks = torch.cat(masks, 0) 
        scope = torch.tensor([[0, len(bag)]]).long() # (1, 2)
        bag_logits = self.forward(None, scope, tokens, pos1s, pos2s, masks, train=False).squeeze(0) # (N) after softmax
        score, pred = bag_logits.max()
        score = score.item()
        pred = pred.item()
        rel = self.id2rel[pred]
        return (rel, score)
    
    def forward(self, label, scope, token, pos1, pos2, mask=None, train=True, bag_size=None):
        """
        Args:
            label: (B), label of the bag
            scope: (B), scope for each bag
            token: (nsum, L), index of tokens
            pos1: (nsum, L), relative position to head entity
            pos2: (nsum, L), relative position to tail entity
            mask: (nsum, L), used for piece-wise CNN
        Return:
            logits, (B, N)
        """
        if mask is not None:
            rep = self.sentence_encoder(token, pos1, pos2, mask) # (nsum, H)
        else:
            rep = self.sentence_encoder(token, pos1, pos2) # (nsum, H)

        # Attention
        if train:
            bag_rep = []
            query = torch.zeros((rep.size(0))).long()
            if torch.cuda.is_available():
                query = query.cuda()
            for i in range(len(scope)):
                query[scope[i][0]:scope[i][1]] = label[i]
            att_mat = self.fc.weight.data[query] # (nsum, H)
            att_score = (rep * att_mat).sum(-1) # (nsum)
            if bag_size is None:
                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
                    softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]]) # (n)
                    bag_rep.append((softmax_att_score.unsqueeze(-1) * bag_mat).sum(0)) # (n, 1) * (n, H) -> (n, H) -> (H)
                bag_rep = torch.stack(bag_rep, 0) # (B, H)
            else:
                batch_size = label.size(0)
                rep = rep.view(batch_size, bag_size, -1) # (B, bag, H)
                att_mat = att_mat.view(batch_size, bag_size, -1) # (B, bag, H)
                att_score = att_score.view(batch_size, bag_size) # (B, bag)
                softmax_att_score = self.softmax(att_score) # (B, bag)
                bag_rep = (softmax_att_score.unsqueeze(-1) * rep).sum(1) # (B, bag, 1) * (B, bag, H) -> (B, bag, H) -> (B, H)
            bag_rep = self.drop(bag_rep)
            bag_logits = self.fc(bag_rep) # (B, N)
        else:
            bag_logits = []
            att_score = torch.matmul(rep, self.fc.weight.data.transpose(0, 1)) # (nsum, H) * (H, N) -> (nsum, N)
            for i in range(len(scope)):
                bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
                softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]].transpose(0, 1)) # (N, (softmax)n) 
                rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat) # (N, n) * (n, H) -> (N, H)
                logit_for_each_rel = self.softmax(self.fc(rep_for_each_rel)) # ((each rel)N, (logit)N)
                logit_for_each_rel = logit_for_each_rel.diag() # (N)
                bag_logits.append(logit_for_each_rel)
            bag_logits = torch.stack(bag_logits,0) # after **softmax**

        return bag_logits

