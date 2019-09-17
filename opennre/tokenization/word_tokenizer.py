# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WordpieceTokenizer classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unicodedata

from .utils import (load_vocab,
                   convert_to_unicode,
                   clean_text,
                   split_on_whitespace,
                   convert_by_vocab,
                   tokenize_chinese_chars)

class WordTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab = None, unk_token="[UNK]"):
        self.vocab = load_vocab(vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_token = unk_token

    def tokenize(self, text):
        """    Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform tokenization
            using the given vocabulary.

            For example:
                input = "unaffable"
                output = ["un", "##aff", "##able"]

            Args:
                text: A single token or whitespace separated tokens. This should have already been passed through `BasicTokenizer`.
            Returns:
                output_tokens: A list of wordpiece tokens.
                current_positions: A list of the current positions for the original words in text .
        """
        text = convert_to_unicode(text)
        text = clean_text(text)
        text = tokenize_chinese_chars(text)
        # output_tokens = []
        token_list = split_on_whitespace(text)
        # for chars in token_list:
        #     # current_positions.append([])
        #     if chars in self.vocab:
        #         output_tokens.append(chars)
        #     else:
        #         output_tokens.append(self.unk_token)                
        return token_list

    def convert_tokens_to_ids(self, tokens, max_seq_length = None, blank_id = 0, unk_id = 1, uncased = True):
        return convert_by_vocab(self.vocab, tokens, max_seq_length, blank_id, unk_id, uncased=uncased)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)
