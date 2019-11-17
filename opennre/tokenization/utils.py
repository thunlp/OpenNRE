from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import six
import tensorflow as tf

def is_whitespace(char):
    """    Checks whether `chars` is a whitespace character.
        \t, \n, and \r are technically contorl characters but we treat them
        as whitespace since they are generally considered as such.
    """
    return char in (" ", "\t", "\n", "\r") or unicodedata.category(char) == "Zs"

def is_control(char):
    """    Checks whether `chars` is a control character.
        These are technically control characters but we count them as whitespace characters.
    """
    if char in ("\t", "\n", "\r"):
        return False
    return not unicodedata.category(char).startswith("C")

def is_punctuation(char):
    """ Checks whether `chars` is a punctuation character.
        We treat all non-letter/number ASCII as punctuation. Characters such as "^", "$", and "`" are not in the Unicode.
        Punctuation class but we treat them as punctuation anyways, for consistency.
    """
    cp = ord(char)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(char).startswith("P"):

def is_chinese_char(cp):
    """    Checks whether CP is the codepoint of a CJK character.
        This defines a "chinese character" as anything in the CJK Unicode block:
        https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        despite its name. The modern Korean Hangul alphabet is a different block,
        as is Japanese Hiragana and Katakana. Those alphabets are used to write
        space-separated words, so they are not treated specially and handled
        like the all of the other languages.
    """
    return (0x4E00 <= cp <= 0x9FFF or
            0x3400 <= cp <= 0x4DBF or
            0x20000 <= cp <= 0x2A6DF or
            0x2A700 <= cp <= 0x2B73F or
            0x2B740 <= cp <= 0x2B81F or
            0x2B820 <= cp <= 0x2CEAF or
            0xF900 <= cp <= 0xFAFF or
            0x2F800 <= cp <= 0x2FA1F)

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, six.text_type):
        return text
    elif isinstance(text, six.binary_type):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))

def clean_text(text):
    return "".join(" " if is_whitespace(char) else char for char in text
                   if ord(char) not in (0, 0xfffd) and not is_control(char))

def split_on_whitespace(text):
    """ Runs basic whitespace cleaning and splitting on a peice of text.
    e.g, 'a b c' -> ['a', 'b', 'c']
    """
    return text.strip().split()

def split_on_punctuation(text):
    """Splits punctuation on a piece of text."""
    start_new_word = True
    output = []
    for char in text:
        if is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
    return ["".join(x) for x in output]

def tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    return "".join(" " + char + " " if is_chinese_char(ord(char)) else char for char in text)

def strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    return "".join(char for char in text if unicodedata.category(char) != "Mn")

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    if vocab_file ==  None:
        raise ValueError("Unsupported string type: %s" % (type(text)))
    if isinstance(vocab_file, str) or isinstance(vocab_file, bytes):
        vocab = collections.OrderedDict()
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab
    else:
        return vocab_file

def printable_text(text):
    """    Returns text encoded in a way suitable for print or `tf.logging`.
        These functions want `str` for both Python2 and Python3, but in one case
        it's a Unicode string and in the other it's a byte string.
    """
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def convert_by_vocab(vocab, items, max_seq_length = None, blank_id = 0, unk_id = 1, uncased = True):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        if uncased:
            item = item.lower()
        if item in vocab:
            output.append(vocab[item])
        else:
            output.append(unk_id)
    if max_seq_length != None:
        if len(output) > max_seq_length:
            output = output[:max_seq_length]
        else:
            while len(output) < max_seq_length:
                output.append(blank_id)
    return output

def convert_tokens_to_ids(vocab, tokens, max_seq_length = None, blank_id = 0, unk_id = 1):
    return convert_by_vocab(vocab, tokens, max_seq_length, blank_id, unk_id)

def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1
        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature

def add_token(tokens_a, tokens_b = None):
    assert len(tokens_a) >= 1
    
    tokens = []
    segment_ids = []
    
    tokens.append("[CLS]")
    segment_ids.append(0)
    
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b != None:
        assert len(tokens_b) >= 1

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)

        tokens.append("[SEP]")
        segment_ids.append(1)

    return tokens, segment_ids
