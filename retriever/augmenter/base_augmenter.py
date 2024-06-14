from abc import ABC
from typing import List

from retriever.types import Document
from retriever.embedder.base_embedder import BaseEmbedder
from retriever.tokenizer import Tokenizer

class BaseAugmenter(ABC):

    def augment(self, 
                documents: List[Document],
                tokenizer: Tokenizer,
                embedder: BaseEmbedder = None,
                parallel: int = 5,
                **kwargs
        ) -> List[float]:
        pass