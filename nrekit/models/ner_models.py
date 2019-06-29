from ..framework import NER
from ..util import *
import spacy
import tagme

class SpacyNER(NER):
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load('en_core_web_sm')
    
    def ner(self, sentence, is_token=False):
        if is_token:
            raise Exception("Spacy NER only takes sentences with string format.")
        doc = self.nlp(sentence)
        result = []
        for ent in doc.ents:
            item = {'name': ent.text, 'pos': [ent.start_char, ent.end_char], 'label': ent.label_}
            result.append(item)
        return result

class TagmeNER(NER):
    def __init__(self):
        super().__init__()
        tagme.GCUBE_TOKEN = "01d0b820-2df2-4aab-ae12-d579ac0ac883-843339462"

    def ner(self, sentence, is_token=False):
        if is_token:
            raise Exception("Spacy NER only takes sentences with string format.")
        ann_result = tagme.annotate(sentence)
        if ann_result is None:
            return []
        result = []
        for ann in ann_result.get_annotations(0.1):
            item = {'name': ann.mention, 'pos': [ann.begin, ann.end], 'score': ann.score, 
                    'id': ann.entity_id, 'ori_name': ann.entity_title}
            result.append(item)
        filtered_result = []
        for ent in result:
            be_included = False
            for father in result:
                if father != ent and father['pos'][0] <= ent['pos'][0] and ent['pos'][1] <= father['pos'][1]:
                    be_included = True
                    break
            if not be_included:
                filtered_result.append(ent)
        return filtered_result

        