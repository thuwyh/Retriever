from abc import ABC
import jieba

class Tokenizer(ABC):

    def tokenize(self, text:str) -> list[str]:
        pass

class SpaceTokenizer(Tokenizer):

    def tokenize(self, text: str) -> list[str]:
        return text.split()

class JiebaTokenizer(Tokenizer):

    def tokenize(self, text: str) -> list[str]:
        return list(jieba.cut(text))
    
SPACE_TOKENIZER = SpaceTokenizer()
JIEBA_TOKENIZER = JiebaTokenizer()