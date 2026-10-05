import nltk
import os
import unicodedata
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# PhenoRAG patch: put the vendored corpora on nltk.data.path before ``stopwords.words`` is
# evaluated in a class body below -- a machine without ~/nltk_data would otherwise die at import.
import hpo_extraction.ontology.nltk_data  # noqa: F401  (import for side effect)

project_path = os.path.dirname(__file__)+"/../"

# file path definition
# PhenoRAG patch: upstream read these from AutoPCR's own data/ directory. Both files are
# BYTE-IDENTICAL to the ones this repo already pins (both projects vendor PhenoBERT's), so they
# are read from resources/util/ instead -- one copy, the one this repository uses. See PATCHES.md.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
stopwords_file_path = os.path.join(_REPO_ROOT, "resources", "util", "stopwords.txt")
num2word_file_path = os.path.join(_REPO_ROOT, "resources", "util", "NUM.txt")
wnl = WordNetLemmatizer()


class WordItem:
    """
    英文单词的包装类
    """
    # 用于lemmatize
    lemma_dict = {}
    def __init__(self, text, start, end):
        self.text = text.lower()
        self.start = start
        self.end = end


def getNum2Word(file_path):
    Num2Word = {}
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            inter = line.strip().split("\t")
            Num2Word[inter[0]] = inter[1]
    return Num2Word


def toString(word_items):
    return ' '.join([i.text for i in word_items])


class PhraseItem:
    """
    英文短语的包装类，包含了文本信息和起止点信息
    """

    Num2Word = getNum2Word(num2word_file_path)
    StopWords = stopwords.words("english")

    def __init__(self, word_items):
        self.word_items = word_items
        self.simple_items = []
        self.simplify()
        self.locs_set = set([i.start for i in word_items])
        self.start_loc = self.word_items[0].start
        self.end_loc = self.word_items[-1].end
        self.no_flag = False
    
    @staticmethod
    def _simplify(word_items):
        simple_items = []
        for word_item in word_items:
            if word_item.text in PhraseItem.Num2Word:
                simple_items.append(WordItem(PhraseItem.Num2Word[word_item.text], word_item.start, word_item.end))
            elif word_item.text in PhraseItem.StopWords or isNum(word_item.text):
                continue
            else:
                simple_items.append(word_item)
        return simple_items

    def simplify(self):
        """
        对phrase_item进行简化，去除常用词以及替换数字
        :return:
        """
        self.simple_items = PhraseItem._simplify(self.word_items)

    @staticmethod
    def _toString(items):
        return ' '.join(i.text for i in items)

    @staticmethod
    def _toSimpleString(word_items):
        simple_items = []
        for word_item in word_items:
            if word_item.text in PhraseItem.Num2Word:
                simple_items.append(PhraseItem.Num2Word[word_item.text])
            elif word_item.text in PhraseItem.StopWords or isNum(word_item.text):
                continue
            else:
                simple_items.append(word_item.text)
        return ' '.join(simple_items)

    def toString(self):
        return PhraseItem._toString(self.word_items)

    def toSimpleString(self):
        return PhraseItem._toString(self.simple_items)

    def include(self, phrase_item):
        if self.locs_set.issubset(phrase_item.locs_set) or self.locs_set.issuperset(phrase_item.locs_set):
            return True
        return False

    def issubset(self, phrase_item):
        if self.locs_set.issubset(phrase_item.locs_set):
            return True
        return False

    def set_no_flag(self):
        self.no_flag = True

    def __len__(self):
        return len(self.word_items)


class SpanTokenizer:
    """
    基于NLTK工具包，自定义一个详细版本的Tokenizer
    """

    def __init__(self):
        self.tokenizer_big = PunktSentenceTokenizer()
        self.tokenizer_small = TreebankWordTokenizer()

    def tokenize(self, text):
        result = []
        sentences_span = self.tokenizer_big.span_tokenize(text)
        for start, end in sentences_span:
            sentence = text[start:end]
            tokens_span = self.tokenizer_small.span_tokenize(sentence)
            for token_start, token_end in tokens_span:
                result.append([start + token_start, start + token_end])
        return result


def strip_accents(s):
    """
    去除口音化，字面意思
    :param s:
    :return:
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def isNum(strings):
    """
    判断给定字符串是否为数字
    :param strings:
    :return:
    """
    try:
        float(strings)
        return True
    except ValueError:
        return False


def getStopWords():
    """
    返回stopwords_file_path给定的stop words
    :return:
    """
    stopwords = set()
    with open(stopwords_file_path, encoding="utf-8") as file:
        for line in file:
            stopwords.add(line.strip())
    return stopwords


def getSpliters():
    """
    用于分割短句的分割词
    :return:
    """
    spliters = set([word for word, pos in nltk.pos_tag(stopwords.words('english')) if pos in {'CC', 'WP', 'TO', 'WDT'}]+[',', '.', ':', ';', '(', ')', '[', ']', '/'])
    return spliters


def normalize_with_char_map(text):
    """
    预处理 text（两步：去掉 A-1 里的 '-', 折叠空白为单空格），
    并返回：
      new_text: 改完的字符串
      index_map: 长度 = len(new_text)，index_map[i] = 对应的原文字符下标
    """
    # 将文本处理成正常的小写形式
    text = strip_accents(text)
    text = re.sub("[\"\'\\\\\t‘’]", " ", text)
    # 对于换行符替换为句号，后续作为分割词
    text = re.sub("(?<=[\w])[\r\n]", ".", text)

    n = len(text)
    out_chars = []
    index_map = []

    i = 0
    while i < n:
        ch = text[i]

        # 规则1: 删除 [A-Z]-[0-9] 中的 '-'，不输出字符，只跳过
        if (ch == '-' and i > 0 and i+1 < n
            and text[i-1].isupper() and text[i+1].isdigit()):
            i += 1
            continue

        # 规则2: 空白折叠
        if ch.isspace():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            # 输出一个空格，映射到原文第一位空白
            out_chars.append(' ')
            index_map.append(i)
            i = j
            continue

        # 普通字符：原样输出，映射到自身位置
        out_chars.append(ch)
        index_map.append(i)
        i += 1

    index_map.append(n)
    text = ''.join(out_chars)
    return text, index_map