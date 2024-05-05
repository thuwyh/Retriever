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
        embedder:BaseEmbedder=None,
        tokenizer:Tokenizer=SPACE_TOKENIZER,
        reranker:BaseReranker=None
    ) -> Retriever:
        
        documents = process_batch(texts, embedder, tokenizer)
        return Retriever(
            documents=documents,
            embedder=embedder,
            tokenizer=tokenizer,
            reranker=reranker
        )