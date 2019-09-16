from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import six

def is_whitespace(char):
    """    Checks whether `chars` is a whitespace character.
        \t, \n, and \r are technically contorl characters but we treat them
        as whitespace since they are generally considered as such.
    """
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def is_control(char):
    """    Checks whether `chars` is a control character.
        These are technically control characters but we count them as whitespace characters.
    """
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def is_punctuation(char):
    """ Checks whether `chars` is a punctuation character.
        We treat all non-letter/number ASCII as punctuation. Characters such as "^", "$", and "`" are not in the Unicode.
        Punctuation class but we treat them as punctuation anyways, for consistency.
    """
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

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
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or
        (cp >= 0x3400 and cp <= 0x4DBF) or
        (cp >= 0x20000 and cp <= 0x2A6DF) or
        (cp >= 0x2A700 and cp <= 0x2B73F) or
        (cp >= 0x2B740 and cp <= 0x2B81F) or
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or
        (cp >= 0x2F800 and cp <= 0x2FA1F)):
        return True
    return False

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def clean_text(text):
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        if is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def split_on_whitespace(text):
    """ Runs basic whitespace cleaning and splitting on a peice of text.
    e.g, 'a b c' -> ['a', 'b', 'c']
    """
    text = text.strip()
    if not text:
        return []
    return text.split()

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
    output = []
    for char in text:
        cp = ord(char)
        if is_chinese_char(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

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
