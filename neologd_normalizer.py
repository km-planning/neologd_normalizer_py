#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs, io, six, sys


class CodeRange(object):
    """
    文字の範囲を定義
    """

    __slots__ = ["codefrom", "codeto"]

    def __init__(self, codefrom, codeto):
        """
        Args:
          codefrom (int): 開始文字コード
          codeto (int):   終了文字コード
        """
        self.codefrom = codefrom
        self.codeto = codeto

    def contains(self, code):
        return (self.codefrom <= code and code <= self.codeto)


class CharGroup(object):
    """
    文字判定用基底クラス
    """

    # 文字コードの範囲
    entries = []

    @classmethod
    def contains(cls, code):
        ncode = ord(code)
        for coderange in cls.entries:
            if coderange.contains(ncode):
                return True
        return False


class JpGroup(CharGroup):
    """
    空白文字制御用の「ひらがな・全角カタカナ・半角カタカナ・漢字・全角記号」
    のような文字を判別する
    - 正規化後に判定するので半角記号も含める
    """

    # 文字コードの範囲
    entries = [
        CodeRange(0x4E00, 0x9FFF), # CJK UNIFIED IDEOGRAPHS
        CodeRange(0x3040, 0x309F), # HIRAGANA
        CodeRange(0x30A0, 0x30FF), # KATAKANA
        CodeRange(0x3000, 0x303F), # CJK SYMBOLS AND PUNCTUATION
        CodeRange(0xFF00, 0xFFEF), # HALFWIDTH AND FULLWIDTH FORMS
        CodeRange(0x0021, 0x002F), # 以下半角記号
        CodeRange(0x003A, 0x0040), #
        CodeRange(0x005B, 0x0060), #
        CodeRange(0x007B, 0x007F)  #
        ]



class AnkGroup(CharGroup):
    """
    半角英数字
    """
    entries = [
      CodeRange(0x0030, 0x0039), # [0-9]
      CodeRange(0x0041, 0x005A), # [A-Z]
      CodeRange(0x0061, 0x007A)  # [a-z]
    ]


def unseparatedPair(leftChar, rightChar):
    """
    結合させる文字の組み合わせを判定する
    Args:
      leftChar ():  前の文字
      rightChar (): 後の文字

    Returns:
      結合させる組み合わせの場合はTrue
    """
    # 前が全角みたいな場合
    return ((JpGroup.contains(leftChar) and
             (JpGroup.contains(rightChar) or AnkGroup.contains(rightChar))) or
            # 前が半角英数字の場合
            (AnkGroup.contains(leftChar) and JpGroup.contains(rightChar)))


class CodePoint(object):

    @classmethod
    def valueOf(cls, code):
        return six.unichr(code)

    def valueOfRange(cls, coderange):
        return [cls.valueOf(code) for code in coderange]


class SingleCharNormalizer(object):
    _norm = ''
    _sources = []

    @classmethod
    def find(cls, char):
        code = char
        for chcode in cls._sources:
            if chcode == code:
                return cls._norm
        return None


class MappingCharNormalizer(object):
    _entries = {}

    @classmethod
    def get(cls, char):
        return cls._entries.get(char)

SPACE = CodePoint.valueOf(0x0020) # SPACE

class SpaceChar(SingleCharNormalizer):
    _norm = SPACE
    _sources = [
        _norm,
        CodePoint.valueOf(0x00A0),  # NO BREAK SPACE
        CodePoint.valueOf(0x2002),  # EN SPACE
        CodePoint.valueOf(0x2003),  # EM SPACE
        CodePoint.valueOf(0x3000),  # IDEOGRAPHIC SPACE / 全角空白
        CodePoint.valueOf(0x2004),  # 1/3 SPACE
        CodePoint.valueOf(0x2005),  # 1/4 SPACE
        CodePoint.valueOf(0x2009),  # THIN SPACE
        CodePoint.valueOf(0x200B),  # NEGATIVE SPACE
    ]

    normalized = _norm


class HyphenChar(SingleCharNormalizer):
    _norm = CodePoint.valueOf(0x002D)  # ハイフンマイナス
    _sources = [
        _norm,
        CodePoint.valueOf(0x02D7),  # MODIFIER LETTER MINUS SIGN
        CodePoint.valueOf(0x058A),  # ARMENIAN HYPHEN
        CodePoint.valueOf(0x2010),  # ハイフン
        CodePoint.valueOf(0x2011),  # ノンブレーキングハイフン
        CodePoint.valueOf(0x2012),  # フィギュアダッシュ
        CodePoint.valueOf(0x2013),  # エヌダッシュ
        CodePoint.valueOf(0x2043),  # Hyphen bullet
        CodePoint.valueOf(0x207B),  # 上付きマイナス
        CodePoint.valueOf(0x208B),  # 下付きマイナス
        CodePoint.valueOf(0x2212)   # 負符号
    ]


class ChoonChar(SingleCharNormalizer):
    _norm = CodePoint.valueOf(0x30FC) # Katakana-Hiragana Prolonged Sound Mark
    _sources = [
        _norm,
        CodePoint.valueOf(0x2014),  # エムダッシュ
        CodePoint.valueOf(0x2015),  # ホリゾンタルバー
        CodePoint.valueOf(0x2500),  # 横細罫線
        CodePoint.valueOf(0x2501),  # 横太罫線
        CodePoint.valueOf(0xFE63),  # SMALL HYPHEN-MINUS
        CodePoint.valueOf(0xFF0D),  # 全角ハイフンマイナス
        CodePoint.valueOf(0xFF70)   # 半角長音記号
    ]


class TildeChar(SingleCharNormalizer):
    _norm = CodePoint.valueOf(0xFF5E)  # 全角チルダ
    _sources = [
        _norm,
        CodePoint.valueOf(0x007E),  # 半角チルダ
        CodePoint.valueOf(0x223C),  # チルダ演算子
        CodePoint.valueOf(0x223E),  # INVERTED LAZY S
        CodePoint.valueOf(0x301C),  # 波ダッシュ
        CodePoint.valueOf(0x3030)   # WAVY DASH
    ]


class ConvertRequired(MappingCharNormalizer):
    _entries = {}
    # ascii 全角 -> 半角    0xFF01 - 0xFF5E
    _entries.update({
        CodePoint.valueOf(0xFF01 + index) : CodePoint.valueOf(0x0021 + index)
        for index in range(0, 0x005D)
    })
    # カタカナ 半角 -> 全角
    _entries.update({
        u'ｱ' : u'ア',
        u'ｲ' : u'イ',
        u'ｳ' : u'ウ',
        u'ｴ' : u'エ',
        u'ｵ' : u'オ',
        u'ｶ' : u'カ',
        u'ｷ' : u'キ',
        u'ｸ' : u'ク',
        u'ｹ' : u'ケ',
        u'ｺ' : u'コ',
        u'ｻ' : u'サ',
        u'ｼ' : u'シ',
        u'ｽ' : u'ス',
        u'ｾ' : u'セ',
        u'ｿ' : u'ソ',
        u'ﾀ' : u'タ',
        u'ﾁ' : u'チ',
        u'ﾂ' : u'ツ',
        u'ﾃ' : u'テ',
        u'ﾄ' : u'ト',
        u'ﾅ' : u'ナ',
        u'ﾆ' : u'ニ',
        u'ﾇ' : u'ヌ',
        u'ﾈ' : u'ネ',
        u'ﾉ' : u'ノ',
        u'ﾊ' : u'ハ',
        u'ﾋ' : u'ヒ',
        u'ﾌ' : u'フ',
        u'ﾍ' : u'ヘ',
        u'ﾎ' : u'ホ',
        u'ﾏ' : u'マ',
        u'ﾐ' : u'ミ',
        u'ﾑ' : u'ム',
        u'ﾒ' : u'メ',
        u'ﾓ' : u'モ',
        u'ﾔ' : u'ヤ',
        u'ﾕ' : u'ユ',
        u'ﾖ' : u'ヨ',
        u'ﾗ' : u'ラ',
        u'ﾘ' : u'リ',
        u'ﾙ' : u'ル',
        u'ﾚ' : u'レ',
        u'ﾛ' : u'ロ',
        u'ﾜ' : u'ワ',
        u'ｦ' : u'ヲ',
        u'ﾝ' : u'ン',
        u'ｧ' : u'ァ',
        u'ｨ' : u'ィ',
        u'ｩ' : u'ゥ',
        u'ｪ' : u'ェ',
        u'ｫ' : u'ォ',
        u'ｯ' : u'ッ',
        u'ｬ' : u'ャ',
        u'ｭ' : u'ュ',
        u'ｮ' : u'ョ',
    })
    # 記号 半角 -> 全角
    _entries.update({
        u'｡' : u'。',
        u'､' : u'、',
        u'･' : u'・',
        u'=' : u'＝',
        u'｢' : u'「',
        u'｣' : u'」',
    })
    # 全角を維持するもの (既に定義されていたら置き換える)
    _entries.update({
        u'。' : u'。',
        u'、' : u'、',
        u'・' : u'・',
        u'＝' : u'＝',
        u'「' : u'「',
        u'」' : u'」',
    })
    # 全角記号
    _entries.update({
        CodePoint.valueOf(0x2018) : CodePoint.valueOf(0x0027),  # Left Single Quotation Mark
        CodePoint.valueOf(0x2019) : CodePoint.valueOf(0x0027),  # Right Single Quotation Mark
        CodePoint.valueOf(0x201C) : CodePoint.valueOf(0x0022),  # Left Double Quotation Mark
        CodePoint.valueOf(0x201D) : CodePoint.valueOf(0x0022),  # Right Double Quotation Mark
        CodePoint.valueOf(0xFE63) : CodePoint.valueOf(0x002D),  # Small Hyphen-Minus
        CodePoint.valueOf(0xFF0D) : CodePoint.valueOf(0x002D),  # Fullwidth Hyphen-Minus
        CodePoint.valueOf(0xFFE0) : CodePoint.valueOf(0x00A2),
        CodePoint.valueOf(0xFFE1) : CodePoint.valueOf(0x00A3),
        CodePoint.valueOf(0xFFE2) : CodePoint.valueOf(0x00AC),
        CodePoint.valueOf(0xFFE3) : CodePoint.valueOf(0x00AF),
        CodePoint.valueOf(0xFFE4) : CodePoint.valueOf(0x00A6),
        CodePoint.valueOf(0xFFE5) : CodePoint.valueOf(0x00A5)
    })


class SonantMark(SingleCharNormalizer, MappingCharNormalizer):
    """
    濁点
    """
    _norm = CodePoint.valueOf(0x309B)
    _sources = [
        CodePoint.valueOf(0x309B),
        CodePoint.valueOf(0x3099),
        CodePoint.valueOf(0xFF9E)
    ]

    _entries = {
        u'カ' : u'ガ',
        u'キ' : u'ギ',
        u'ク' : u'グ',
        u'ケ' : u'ゲ',
        u'コ' : u'ゴ',
        u'サ' : u'ザ',
        u'シ' : u'ジ',
        u'ス' : u'ズ',
        u'セ' : u'ゼ',
        u'ソ' : u'ゾ',
        u'タ' : u'ダ',
        u'チ' : u'ヂ',
        u'ツ' : u'ヅ',
        u'テ' : u'デ',
        u'ト' : u'ド',
        u'ハ' : u'バ',
        u'ヒ' : u'ビ',
        u'フ' : u'ブ',
        u'ヘ' : u'ベ',
        u'ホ' : u'ボ',
        u'ウ' : u'ヴ'
    }


class ConsonantMark(SingleCharNormalizer, MappingCharNormalizer):
    """
    半濁点
    """
    _norm = CodePoint.valueOf(0x309C)
    _sources = [
        _norm,
        CodePoint.valueOf(0x309A),
        CodePoint.valueOf(0xFF9F)
    ]
    _entries = {
        u'ハ' : u'パ',
        u'ヒ' : u'ピ',
        u'フ' : u'プ',
        u'ヘ' : u'ペ',
        u'ホ' : u'ポ'
    }


class Alphabet:
    lower = [six.unichr(i) for i in range(0x0061, 0x0061+26)]   # [a-z]
    upper = [six.unichr(i) for i in range(0x0041, 0x0041+26)]   # [A-Z]


class AbstractFilter(object):

    def apply(self, ch):
        raise NotImplementedError()

class PassThroughFilter(AbstractFilter):

    def apply(self, ch):
        return ch

class MappingFilter(AbstractFilter):
    def __init__(self, mapping = {}):
        self.mapping = mapping

    def apply(self, ch):
        return self.mapping.get(ch, ch)

class UpperFilter(MappingFilter):
    def __init__(self):
        super(UpperFilter, self).__init__(dict(zip(Alphabet.lower, Alphabet.upper)))


class LowerFilter(MappingFilter):
    def __init__(self):
        super(LowerFilter, self).__init__(dict(zip(Alphabet.upper, Alphabet.lower)))


class SentenceBulder(object):
    #from io import StringIO
    @classmethod
    def build(cls, document, filter_proc = PassThroughFilter()):
        #with StringIO(document) as input
        buffer = ""
        for index, ch in enumerate(document):
            offset = 0
            sp = SpaceChar.find(ch)
            if sp is not None:
                if len(buffer) > 0:
                    yield buffer
                    buffer = ""
                continue

            hc = HyphenChar.find(ch)
            if hc is not None:
                ch = hc
                #buffer += hc
                #continue

            cc = ChoonChar.find(ch)
            if cc is not None:
                if len(buffer) > 0 and buffer[-1] == cc:
                    continue
                ch = cc

            tc = TildeChar.find(ch)
            if tc is not None:
                continue

            sm = SonantMark.find(ch)
            if sm is not None:
                ch = sm
                if len(buffer) > 0:
                    norm = SonantMark.get(buffer[-1])
                    if norm is not None:
                        offset = -1
                        ch = norm

            cm = ConsonantMark.find(ch)
            if cm is not None:
                ch = cm
                if len(buffer) > 0:
                    norm = ConsonantMark.get(buffer[-1])
                    if norm is not None:
                        offset = -1
                        ch = norm

            cr = ConvertRequired.get(ch)
            if cr is not None:
                ch = cr

            ch = filter_proc.apply(ch)

            if offset < 0:
                buffer = buffer[0:offset] + ch
            else:
                buffer += ch

        if len(buffer) > 0:
            yield buffer


class Normalizer(object):

    def __init__(self, **kwargs):
        filter_proc = PassThroughFilter()
        upper = kwargs.get('upper', False)
        lower = kwargs.get('lower', False)
        if upper or lower:
            filter_proc = UpperFilter() if upper else LowerFilter()
        self._filter_proc = filter_proc

    def _concat(self, left, right):
        if unseparatedPair(left[-1], right[0]):
            return left + right
        return left + SPACE + right

    def normalize(self, document):
        return self.apply(document)

    def apply(self, document):
        sentence_iterator = SentenceBulder.build(document, self._filter_proc)
        try:
            result = next(sentence_iterator)
        except StopIteration:
            result = None

        if result is None:
            return ""
        for sentence in sentence_iterator:
            result = self._concat(result, sentence)
        return result


def normalize(document, **kwargs):
    return Normalizer(**kwargs).apply(document)


def test():
    import resource
    import functools

    assert "0123456789" == normalize("０１２３４５６７８９")
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" == normalize("ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ")
    assert "abcdefghijklmnopqrstuvwxyz" == normalize("ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ")
    assert "!\"#$%&'()*+,-./:;<>?@[¥]^_`{|}" == normalize("！”＃＄％＆’（）＊＋，−．／：；＜＞？＠［￥］＾＿｀｛｜｝")
    assert "＝。、・「」" == normalize("＝。、・「」")
    assert "ハンカク" == normalize("ﾊﾝｶｸ")
    assert "o-o" == normalize("o₋o")
    assert "majikaー" == normalize("majika━")
    assert "わい" == normalize("わ〰い")
    assert "スーパー" == normalize("スーパーーーー")
    assert "!#" == normalize("!#")
    assert "ゼンカクスペース" == normalize("ゼンカク　スペース")
    assert "おお" == normalize("お             お")
    assert "おお" == normalize("      おお")
    assert "おお" == normalize("おお      ")
    assert "検索エンジン自作入門を買いました!!!" == \
        normalize("検索 エンジン 自作 入門 を 買い ました!!!")
    assert "アルゴリズムC" == normalize("アルゴリズム C")
    assert "PRML副読本" == normalize("　　　ＰＲＭＬ　　副　読　本　　　")
    assert "Coding the Matrix" == normalize("Coding the Matrix")
    assert "南アルプスの天然水Sparking Lemonレモン一絞り" == \
        normalize("南アルプスの　天然水　Ｓｐａｒｋｉｎｇ　Ｌｅｍｏｎ　レモン一絞り")
    assert "南アルプスの天然水-Sparking*Lemon+レモン一絞り" == \
        normalize("南アルプスの　天然水-　Ｓｐａｒｋｉｎｇ*　Ｌｅｍｏｎ+　レモン一絞り")
    print("use: %d" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def _file_context(iofile, mode, default, **kwargs):
    from contextlib import contextmanager

    @contextmanager
    def context():
        f = codecs.open(iofile, mode, **kwargs) if iofile and iofile not in ['-'] else default
        yield f
        f is default or f.close()
    return context()



def main(argv):
    import argparse
    def get_args(argv):
        parser = argparse.ArgumentParser(
            description='normalizer for ipadic_neologd.')
        parser.add_argument(
            "-i", "--in",
            dest="infile",
            type=str,
            default='-',
            help="input file.")
        parser.add_argument(
            "-o", "--out",
            dest="outfile",
            type=str,
            default='-',
            help="output file.")
        parser.add_argument(
            "--upper",
            dest="use_upper",
            action="store_true",
            default=False,
            help="[a-z] to [A-Z].")
        parser.add_argument(
            "--lower",
            dest="use_lower",
            action="store_true",
            default=False,
            help="[A-Z] to [a-z].")
        return parser.parse_args(argv)
    opts = get_args(argv[1:])
    normalizer = Normalizer(lower = opts.use_lower,
                            upper = opts.use_upper)
    with _file_context(opts.infile, 'r', sys.stdin, encoding='utf-8') as infile:
        with _file_context(
                opts.outfile,
                'w',
                sys.stdout,
                encoding='utf-8') as outfile:
            for line in infile:
                outfile.write(normalizer.apply(line))

if __name__ == '__main__':
    main(sys.argv)
