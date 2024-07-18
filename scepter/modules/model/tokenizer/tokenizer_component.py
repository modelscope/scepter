# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import html
import string
from functools import lru_cache
from urllib import parse

import ftfy
import regex as re
from bs4 import BeautifulSoup


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to
    avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (list(range(ord('!'),
                     ord('~') + 1)) + list(range(ord('¡'),
                                                 ord('¬') + 1)) +
          list(range(ord('®'),
                     ord('ÿ') + 1)))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace('_', ' ')
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans('', '', string.punctuation))
            for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def heavy_clean(text):
    text = str(text)
    text = parse.unquote_plus(text)
    text = text.strip().lower()
    text = re.sub('<person>', 'person', text)

    # urls:
    text = re.sub(
        r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa: E501
        '',
        text)
    text = re.sub(
        r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa: E501
        '',
        text)

    # html:
    text = BeautifulSoup(text, features='html.parser').text

    # @<nickname>
    text = re.sub(r'@[\w\d]+\b', '', text)

    # 31C0—31EF CJK Strokes
    # 31F0—31FF Katakana Phonetic Extensions
    # 3200—32FF Enclosed CJK Letters and Months
    # 3300—33FF CJK Compatibility
    # 3400—4DBF CJK Unified Ideographs Extension A
    # 4DC0—4DFF Yijing Hexagram Symbols
    # 4E00—9FFF CJK Unified Ideographs
    text = re.sub(r'[\u31c0-\u31ef]+', '', text)
    text = re.sub(r'[\u31f0-\u31ff]+', '', text)
    text = re.sub(r'[\u3200-\u32ff]+', '', text)
    text = re.sub(r'[\u3300-\u33ff]+', '', text)
    text = re.sub(r'[\u3400-\u4dbf]+', '', text)
    text = re.sub(r'[\u4dc0-\u4dff]+', '', text)
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    #######################################################

    # все виды тире / all types of dash --> "-"
    text = re.sub(
        r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',  # noqa: E501
        '-',
        text)

    # кавычки к одному стандарту
    text = re.sub(r'[`´«»“”¨]', '"', text)
    text = re.sub(r'[‘’]', "'", text)

    # &quot;
    text = re.sub(r'&quot;?', '', text)
    # &amp
    text = re.sub(r'&amp', '', text)

    # ip adresses:
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', text)

    # article ids:
    text = re.sub(r'\d:\d\d\s+$', '', text)

    # \n
    text = re.sub(r'\\n', ' ', text)

    # "#123"
    text = re.sub(r'#\d{1,3}\b', '', text)
    # "#12345.."
    text = re.sub(r'#\d{5,}\b', '', text)
    # "123456.."
    text = re.sub(r'\b\d{6,}\b', '', text)
    # filenames:
    text = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '',
                  text)

    text = re.sub(r'[\"\']{2,}', r'"', text)  # """AUSVERKAUFT"""
    text = re.sub(r'[\.]{2,}', r' ', text)  # """AUSVERKAUFT"""
    text = re.sub(
        re.compile(r'[' + '#®•©™&@·º½¾¿¡§~' + '\)' + '\(' + '\]' +  # noqa
                   '\[' +  # noqa
                   '\}' + '\{' + '\|' + '\\' + '\/' + '\*' +  # noqa
                   r']{1,}'),  # noqa
        r' ',
        text)  # ***AUSVERKAUFT***, #AUSVERKAUFT
    text = re.sub(r'\s+\.\s+', r' ', text)  # " . "

    # this-is-my-cute-cat / this_is_my_cute_cat
    regex2 = re.compile(r'(?:\-|\_)')
    if len(re.findall(regex2, text)) > 3:
        text = re.sub(regex2, ' ', text)

    text = basic_clean(text)

    text = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', text)  # jc6640
    text = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', text)  # jc6640vc
    text = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', text)  # 6640vc231

    text = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', text)
    text = re.sub(r'(free\s)?download(\sfree)?', '', text)
    text = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', text)
    text = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?',
                  '', text)
    text = re.sub(r'\bpage\s+\d+\b', '', text)

    # j2d1a2a...
    text = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', text)

    text = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', text)
    text = re.sub(r'\b\s+\:\s+', r': ', text)
    text = re.sub(r'(\D[,\./])\b', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', text)
    text = re.sub(r'^[\'\_,\-\:;]', r'', text)
    text = re.sub(r'[\'\_,\-\:\-\+]$', r'', text)
    text = re.sub(r'^\.\S+$', '', text)
    return text.strip()
