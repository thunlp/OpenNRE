import torch
from torch import nn, optim
from .base_model import BagRE

class BagOne(BagRE):
    """
    Instance one(max) for bag-level relation extraction.
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
            token, pos1, pos2, mask = self.sentence_encoder.tokenize(item)
            tokens.append(token)
            pos1s.append(pos1)
            pos2s.append(pos2)
            masks.append(mask)
        tokens = torch.cat(tokens, 0).unsqueeze(0) # (n, L)
        pos1s = torch.cat(pos1s, 0).unsqueeze(0)
        pos2s = torch.cat(pos2s, 0).unsqueeze(0)
        masks = torch.cat(masks, 0).unsqueeze(0)
        scope = torch.tensor([[0, len(bag)]]).long() # (1, 2)
        bag_logits = self.forward(None, scope, tokens, pos1s, pos2s, masks, train=False).squeeze(0) # (N) after softmax
        score, pred = bag_logits.max(0)
        score = score.item()
        pred = pred.item()
        rel = self.id2rel[pred]
        return (rel, score)
    
    def forward(self, label, scope, token, pos1, pos2, mask=None, train=True, bag_size=0):
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
        # Encode
        if bag_size > 0:
            token = token.view(-1, token.size(-1))
            pos1 = pos1.view(-1, pos1.size(-1))
            pos2 = pos2.view(-1, pos2.size(-1))
            if mask is not None:
                mask = mask.view(-1, mask.size(-1))
        else:
            begin, end = scope[0][0], scope[-1][1]
            token = token[:, begin:end, :].view(-1, token.size(-1))
            pos1 = pos1[:, begin:end, :].view(-1, pos1.size(-1))
            pos2 = pos2[:, begin:end, :].view(-1, pos2.size(-1))
            if mask is not None:
                mask = mask[:, begin:end, :].view(-1, mask.size(-1))
            scope = torch.sub(scope, torch.zeros_like(scope).fill_(begin))

        if train or bag_size > 0:
            if mask is not None:
                rep = self.sentence_encoder(token, pos1, pos2, mask) # (nsum, H) 
            else:
                rep = self.sentence_encoder(token, pos1, pos2) # (nsum, H) 
        else:
            rep = []
            bs = 256
            total_bs = len(token) // bs + (1 if len(token) % bs != 0 else 0)
            for b in range(total_bs):
                with torch.no_grad():
                    left = bs * b
                    right = min(bs * (b + 1), len(token))
                    if mask is not None:        
                        rep.append(self.sentence_encoder(token[left:right], pos1[left:right], pos2[left:right], mask[left:right]).detach()) # (nsum, H) 
                    else:
                        rep.append(self.sentence_encoder(token[left:right], pos1[left:right], pos2[left:right]).detach()) # (nsum, H) 
            rep = torch.cat(rep, 0)

        # Max
        if train:
            if bag_size == 0:
                bag_rep = []
                query = torch.zeros((rep.size(0))).long()
                if torch.cuda.is_available():
                    query = query.cuda()
                for i in range(len(scope)):
                    query[scope[i][0]:scope[i][1]] = label[i]

                for i in range(len(scope)): # iterate over bags
                    bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
                    instance_logit = self.softmax(self.fc(bag_mat)) # (n, N)
                    # select j* which scores highest on the known label
                    max_index = instance_logit[:, query[i]].argmax()  # (1)
                    bag_rep.append(bag_mat[max_index]) # (n, H) -> (H)
                bag_rep = torch.stack(bag_rep, 0) # (B, H)
                bag_rep = self.drop(bag_rep)
                bag_logits = self.fc(bag_rep) # (B, N)
            else:
                batch_size = label.size(0)
                query = label # (B)
                rep = rep.view(batch_size, bag_size, -1)
                instance_logit = self.softmax(self.fc(rep))
                max_index = instance_logit[torch.arange(batch_size), :, query].argmax(-1)
                bag_rep = rep[torch.arange(batch_size), max_index]

                bag_rep = self.drop(bag_rep)
                bag_logits = self.fc(bag_rep) # (B, N)

        else:
            if bag_size == 0:
                bag_logits = []
                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
                    instance_logit = self.softmax(self.fc(bag_mat)) # (n, N)
                    logit_for_each_rel = instance_logit.max(dim=0)[0] # (N)
                    bag_logits.append(logit_for_each_rel)
                bag_logits = torch.stack(bag_logits, 0) # after **softmax**
            else:
                batch_size = rep.size(0) // bag_size
                rep = rep.view(batch_size, bag_size, -1)
                bag_logits = self.softmax(self.fc(rep)).max(1)[0]

        return bag_logits

