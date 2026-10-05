import os
import re
from .utils import *

# PhenoRAG patch: upstream built the Stanza pipeline at module import with no resources dir, which
# makes it reach for ~/stanza_resources (and the network) the moment anything imports ``ee``. This
# repo already builds the identical pipeline -- ``Pipeline(lang="en", package="mimic",
# processors={"ner": "i2b2"}, download_method=None)`` -- in hpo_extraction.data.segmentation.load_stanza, which is
# the only form that works on the offline compute nodes. Construction is deferred to first use so
# importing this module stays free. See PATCHES.md.
_clinical_ner_model = None


def set_stanza_dir(stanza_dir):
    """Point the (lazily built) clinical NER pipeline at a Stanza resources directory."""
    global _clinical_ner_model
    if stanza_dir:
        os.environ["STANZA_RESOURCES_DIR"] = str(stanza_dir)
    _clinical_ner_model = None


def get_clinical_ner_model():
    """The i2b2/mimic Stanza pipeline AutoPCR's entity extraction runs on."""
    global _clinical_ner_model
    if _clinical_ner_model is None:
        from hpo_extraction.data.segmentation import load_stanza

        stanza_dir = os.environ.get("STANZA_RESOURCES_DIR")
        if not stanza_dir:
            raise RuntimeError(
                "no Stanza resources directory configured -- call ee.pbert.set_stanza_dir() or "
                "set STANZA_RESOURCES_DIR. The compute nodes cannot download one."
            )
        _clinical_ner_model = load_stanza(stanza_dir, mode="NER")
    return _clinical_ner_model


def process_text2phrases(text, clinical_ner_model):
    """
    用于从文本中提取Clinical Text Segments
    :param text:自由文本
    :param clinical_ner_model: Stanza提供的预训练NER模型
    :return: List[PhraseItem]
    """
    tokenizer = SpanTokenizer()
    spliters = getSpliters()
    stopwords = getStopWords()
    # # 将文本处理成正常的小写形式
    # text = strip_accents(text.lower())
    # text = re.sub("[-_\"\'\\\\\t‘’]", " ", text)
    # # 对于换行符替换为句号，后续作为分割词
    # text = re.sub("(?<=[\w])[\r\n]", ".", text)

    clinical_docs = clinical_ner_model(text)
    sub_sentences = []

    for sent_c in clinical_docs.sentences:
        clinical_tokens = sent_c.tokens

        # Stanza
        flag = False
        curSentence = []
        tmp = set()
        for i in range(len(clinical_tokens)):
            wi = WordItem(clinical_tokens[i].text, clinical_tokens[i].start_char, clinical_tokens[i].end_char)
            if "PROBLEM" in clinical_tokens[i].ner and wi.text not in {',', '.', ':', ';', '(', ')', '[', ']'}:
                curSentence.append(wi)
            else:
                if len(curSentence) > 0:
                    phrase_item = PhraseItem(curSentence)
                    sub_sentences.append(phrase_item)
                    tmp.update(phrase_item.locs_set)
                    flag = True
                curSentence = []

        if len(curSentence) > 0:
            phrase_item = PhraseItem(curSentence)
            sub_sentences.append(phrase_item)
            tmp.update(phrase_item.locs_set)
            flag = True

        # phrase segmentation
        # 只有Stanza标记的句子才作补充
        if not flag:
            continue
        curSentence = []
        for i in range(len(clinical_tokens)):
            wi = WordItem(clinical_tokens[i].text, clinical_tokens[i].start_char, clinical_tokens[i].end_char)
            # 用于后续lemma比对
            text_lemma = wnl.lemmatize(clinical_tokens[i].text)
            if clinical_tokens[i].text not in WordItem.lemma_dict:
                WordItem.lemma_dict[clinical_tokens[i].text] = text_lemma
            if clinical_tokens[i].text in spliters:
                if len(curSentence) > 0:
                    phrase_item = PhraseItem(curSentence)
                    # 只有不与Stanza重叠的部分才加入
                    if len(phrase_item.locs_set & tmp) == 0:
                        sub_sentences.append(phrase_item)
                curSentence = []
            else:
                curSentence.append(wi)

        if len(curSentence) > 0:
            phrase_item = PhraseItem(curSentence)
            if len(phrase_item.locs_set & tmp) == 0:
                sub_sentences.append(phrase_item)

    # 否定检测
    for phrase_item in sub_sentences:
        flag = False
        for token in phrase_item.word_items:
            if token.text.lower() in {"no", "not", "none", "negative", "non", "never", "few", "lower", "fewer", "less",
                                      "normal"}:
                flag = True
                break
        if flag:
            phrase_item.set_no_flag()

    # 省略恢复
    sub_sentences_ = []
    for idx, pi in enumerate(sub_sentences):
        # 将含有and, or, / 的短句用tokenize进行拆分
        sub_locs = [[i + pi.start_loc, j + pi.start_loc] for i, j in
                    tokenizer.tokenize(text[pi.start_loc:pi.end_loc])]
        sub_phrases = []
        curr_phrase = []
        # 把以and, or, / 分割的短语提出
        for loc in sub_locs:
            wi = WordItem(text[loc[0]:loc[1]], loc[0], loc[1])
            if wi.text in {"and", "or", "/"}:
                if len(curr_phrase) > 0:
                    sub_phrases.append(PhraseItem(curr_phrase))
                    sub_phrases[-1].no_flag = pi.no_flag
                curr_phrase = []
            else:
                curr_phrase.append(wi)
        if len(curr_phrase) > 0:
            sub_phrases.append(PhraseItem(curr_phrase))
            sub_phrases[-1].no_flag = pi.no_flag

        # 首先把原始分割的短语都加进去
        for item in sub_phrases:
            sub_sentences_.append(item)

        # 只考虑A+B形式的恢复
        if len(sub_phrases) == 2:
            if len(sub_phrases[0]) >= 1 and len(sub_phrases[1]) == 1:
                tmp = sub_phrases[0].word_items[:-1][:]
                tmp.extend(sub_phrases[1].word_items)
                sub_sentences_.append(PhraseItem(tmp))
                sub_sentences_[-1].no_flag = pi.no_flag
            elif len(sub_phrases[0]) == 1 and len(sub_phrases[1]) >= 1:
                tmp = sub_phrases[0].word_items[:]
                tmp.extend(sub_phrases[1].word_items[1:])
                sub_sentences_.append(PhraseItem(tmp))
                sub_sentences_[-1].no_flag = pi.no_flag

    sub_sentences = sub_sentences_

    # print([i.toSimpleString() for i in sub_sentences])

    # 穷举短语 删除纯数字
    phrases_list = []
    for pi in sub_sentences:
        tmp = pi.toSimpleString()
        if isNum(tmp) or len(tmp) <= 1:
            continue
        for i in range(len(pi.simple_items)):
            for j in range(10):
                if i + j == len(pi.simple_items):
                    break
                if len(pi.simple_items[i:i + j + 1]) == 1:
                    tmp_str = pi.simple_items[i:i + j + 1][0].text
                    if tmp_str in stopwords or isNum(tmp_str):
                        continue
                phrases_list.append(PhraseItem(pi.simple_items[i:i + j + 1]))
                phrases_list[-1].no_flag = pi.no_flag

    # print(len(phrases_list))
    # print([i.toString() for i in phrases_list])
    return phrases_list


def process_text2phrases_pbert(text):
    # text = re.sub("(?<=[A-Z])-(?=[\d])", "", text)
    text, index_map = normalize_with_char_map(text)
    phrases_list = process_text2phrases(text, get_clinical_ner_model())
    return text, phrases_list, index_map