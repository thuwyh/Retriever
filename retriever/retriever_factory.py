from typing import List
from retriever.retriever import Retriever
from retriever.tokenizer import Tokenizer, SPACE_TOKENIZER
from retriever.embedder import BaseEmbedder
from retriever.reranker import BaseReranker
from retriever.document import Document
from retriever.doc_processor import process_batch

class RetrieverFactory:

    @staticmethod
    def from_raw_text_content(
        texts:List[str],
        metas: List[object]=None,
        embedder:BaseEmbedder=None,
        tokenizer:Tokenizer=SPACE_TOKENIZER,
        reranker:BaseReranker=None
    ) -> Retriever:
        if metas is None:
            documents = process_batch(texts, [None]*len(texts), embedder, tokenizer)
        else:
            assert len(texts)==len(metas)
            documents = process_batch(texts, metas, embedder, tokenizer)
        return Retriever(
            documents=documents,
            embedder=embedder,
            tokenizer=tokenizer,
            reranker=reranker
        )