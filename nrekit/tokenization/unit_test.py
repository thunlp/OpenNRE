from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import six
import tensorflow as tf

from .bert_tokenizer import BertTokenizer
from .basic_tokenizer import BasicTokenizer
from .word_piece_tokenizer import WordpieceTokenizer
from .utils import *

class TokenizationTest(tf.test.TestCase):
    
    def test_full_tokenizer(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing", ","
        ]
        f = open("v.txt", "w")
        if six.PY2:
            f.write("".join([x + "\n" for x in vocab_tokens]))
        else:
            a = "".join([x + "\n" for x in vocab_tokens])
            f.write(a)
        f.close()
  
        vocab_file = 'v.txt'
  
        tokenizer = BertTokenizer(vocab_file)
  
        tokens, gg = tokenizer.tokenize(u"UNwant\u00E9d,running")
        self.assertAllEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])
  
  
        self.assertAllEqual(tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])
  
        res,gg =  tokenizer.basic_tokenizer.tokenize(u"ah\u535A\u63A8zz")
  
        self.assertAllEqual(res, [u"ah", u"\u535A", u"\u63A8", u"zz"])
  
        
        tokenizer = BasicTokenizer(do_lower_case=True)
        res, gg = tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  ")
        self.assertAllEqual(res, ["hello", "!", "how", "are", "you", "?"])
        
        res, _ = tokenizer.tokenize(u"H\u00E9llo")
        self.assertAllEqual(res, ["hello"])
  
        tokenizer = BasicTokenizer(do_lower_case=False)
        res, _ = tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  ")
        self.assertAllEqual(res, ["HeLLo", "!", "how", "Are", "yoU", "?"])
  
  
    def test_wordpiece_tokenizer(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing"
        ]
  
        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
          vocab[token] = i
        tokenizer = WordpieceTokenizer(vocab=vocab)
  
        res, _ = tokenizer.tokenize("")
        self.assertAllEqual(res, [])
        res, _ = tokenizer.tokenize("unwanted running")
  
        self.assertAllEqual(
            res,
            ["un", "##want", "##ed", "runn", "##ing"])
        res, _ = tokenizer.tokenize("unwantedX running")
  
        self.assertAllEqual(res, ["[UNK]", "runn", "##ing"])
  
    def test_convert_tokens_to_ids(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing"
        ]
  
        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
          vocab[token] = i
  
        self.assertAllEqual(
            convert_tokens_to_ids(
                vocab, ["un", "##want", "##ed", "runn", "##ing"]), [7, 4, 5, 8, 9])
  
    def test_is_whitespace(self):
        tokenizatio = BasicTokenizer(do_lower_case=True)
        self.assertTrue(is_whitespace(u" "))
        self.assertTrue(is_whitespace(u"\t"))
        self.assertTrue(is_whitespace(u"\r"))
        self.assertTrue(is_whitespace(u"\n"))
        self.assertTrue(is_whitespace(u"\u00A0"))
  
        self.assertFalse(is_whitespace(u"A"))
        self.assertFalse(is_whitespace(u"-"))
  
    def test_is_control(self):
        tokenizatio = BasicTokenizer(do_lower_case=True)
        self.assertTrue(is_control(u"\u0005"))
        self.assertFalse(is_control(u"A"))
        self.assertFalse(is_control(u" "))
        self.assertFalse(is_control(u"\t"))
        self.assertFalse(is_control(u"\r"))
  
    def test_is_punctuation(self):
        self.assertTrue(is_punctuation(u"-"))
        self.assertTrue(is_punctuation(u"$"))
        self.assertTrue(is_punctuation(u"`"))
        self.assertTrue(is_punctuation(u"."))
  
        self.assertFalse(is_punctuation(u"A"))
        self.assertFalse(is_punctuation(u" "))
  
    def test(self):
        self.test_full_tokenizer()
        self.test_wordpiece_tokenizer()
        self.test_convert_tokens_to_ids()
        self.test_is_whitespace()
        self.test_is_control()
        self.test_is_punctuation()
