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

class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab = None, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = load_vocab(vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

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
        output_tokens = []
        current_positions = []
        token_list = split_on_whitespace(text)
        for chars in token_list:
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                if start > 0:
                    substr = "##" + chars[start:end]
                else:
                    substr = chars[start:end]
                cur_substr = None
                while start < end:
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                    substr = substr[:-1]
                if cur_substr is None:
                    is_bad = True
                    break
                else:
                    sub_tokens.append(cur_substr)
                    start = end
            current_positions.append([])
            if is_bad:
                current_positions[-1].append(len(output_tokens))
                output_tokens.append(self.unk_token)
                current_positions[-1].append(len(output_tokens))
            else:
                current_positions[-1].append(len(output_tokens))
                output_tokens.extend(sub_tokens)
                current_positions[-1].append(len(output_tokens))
        
        return output_tokens, current_positions

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)
