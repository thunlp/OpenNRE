import torch
from torch import nn, optim
from ..data_loader import get_sentence_re_loader
from ..framework import FewShotRE
from ..util import *

class ProtoNetwork(FewShotRE):
    def __init__(self, senetnce_encoder):
        super().__init__()
        self.sentence_encoder = senetnce_encoder
        self.softmax = nn.Softmax(-1)
    
    def infer(self, support, query):
        rel2id = {}
        N = 0
        S = []
        # Process supporting set
        for ins in support:
            if 'text' in ins:
                token, pos1, pos2 = self.sentence_encoder.tokenize(ins['text'], ins['h']['pos'],
                    ins['t']['pos'], is_token=False)
                rep = self.sentence_encoder(token, pos1, pos2).squeeze(0) # (H)
            else:
                token, pos1, pos2 = self.sentence_encoder.tokenize(ins['token'], ins['h']['pos'],
                    ins['t']['pos'], is_token=True)
                rep = self.sentence_encoder(token, pos1, pos2).squeeze(0) # (H)
            if ins['relation'] not in rel2id:
                rel2id[ins['relation']] = N
                N += 1
                S.append([])
            S[rel2id[ins['relation']]].append(rep)
        id2rel = {}
        for rel, idx in rel2id.items():
            id2rel[idx] = rel
        
        # Prototype calculation
        protos = []
        for reps in S:
            tensor_reps = torch.stack(reps, 0) # (K, H)
            proto = tensor_reps.mean(0) # (H)
            protos.append(proto)
        protos = torch.stack(protos, 0) # (N, H)

        # Query
        result = []
        for ins in query:
            if 'text' in ins:
                token, pos1, pos2 = self.sentence_encoder.tokenize(ins['text'], ins['h']['pos'],
                    ins['t']['pos'], is_token=False)
                rep = self.sentence_encoder(token, pos1, pos2) # (1, H)
            else:
                token, pos1, pos2 = self.sentence_encoder.tokenize(ins['token'], ins['h']['pos'],
                    ins['t']['pos'], is_token=True)
                rep = self.sentence_encoder(token, pos1, pos2) # (1, H)
            # Euclidean
            dis = torch.pow(protos - rep, 2).sum(-1) # (N)
            dis = self.softmax(-dis)
            score, pred = dis.max(-1)
            score = score.item()
            rel = id2rel[pred.item()]
            result.append((rel, score))
        
        return result


class BERTPair(FewShotRE):
    def __init__(self, sentence_encoder):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.softmax = nn.Softmax(-1)
    
    def infer(self, support, query):
        result = []
        for Q in query:
            rel2id = {}
            N = 0
            sim = []
        
            # Calculate similarities between each supporting ins and query ins 
            for ins in support:
                if 'text' in ins:
                    token, sep, att_mask = self.sentence_encoder.tokenize(
                        ins['text'], ins['h']['pos'], ins['t']['pos'], 
                        Q['text'], Q['h']['pos'], Q['t']['pos'], is_token=False)
                    logits = self.sentence_encoder(token, sep, att_mask).squeeze(0) # (2)
                else:
                    token, sep, att_mask = self.sentence_encoder.tokenize(
                        ins['token'], ins['h']['pos'], ins['t']['pos'], 
                        Q['token'], Q['h']['pos'], Q['t']['pos'], is_token=True)
                    logits = self.sentence_encoder(token, sep, att_mask).squeeze(0) # (2)
                if ins['relation'] not in rel2id:
                    rel2id[ins['relation']] = N
                    N += 1
                    sim.append([])
                sim[rel2id[ins['relation']]].append(logits)
            id2rel = {}
            for rel, idx in rel2id.items():
                id2rel[idx] = rel

            logits = []     
            for sim_one in sim:
                sim_one = torch.stack(sim_one, 0) # (K, 2)
                sim_one = sim_one.mean(0) # (2)
                logits.append(sim_one)
            logits = torch.stack(logits) # (N, 2)

            # Calculate probabilities for each relation and none of the above
            logits_na, _ = logits[:, 0].min(0, keepdim=True) # (1)
            logits = logits[:, 1] # (N)
            logits = torch.cat([logits, logits_na], -1) # (N + 1)
            logits = self.softmax(logits) # (N + 1)
            score, pred = logits.max(-1)
            score = score.item()
            pred = pred.item()
            if pred in id2rel:
                rel = id2rel[pred]
            else:
                rel = 'None of the above'
            result.append((rel, score))
               
        return result