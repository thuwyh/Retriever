from typing import List
import pickle
from retriever.augmenter.base_augmenter import BaseAugmenter
from retriever.retriever import Retriever
from retriever.tokenizer import Tokenizer, SPACE_TOKENIZER
from retriever.embedder import BaseEmbedder
from retriever.reranker import BaseReranker
# from retriever.types import Document
from retriever.doc_processor import process_batch
from retriever.types import Document


def save_documents(documents: List[Document], output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(documents, f)

def load_documents(dump_path):
    with open(dump_path, 'rb') as f:
        data = pickle.load(f)
        return data
class RetrieverFactory:

    @staticmethod
    def from_raw_text_content(
        texts:List[str],
        metas: List[object]=None,
        embedder:BaseEmbedder=None,
        tokenizer:Tokenizer=SPACE_TOKENIZER,
        reranker:BaseReranker=None,
        augmenter: BaseAugmenter = None,
        documents_save_path: str = None
    ) -> Retriever:
        if metas is None:
            documents = process_batch(texts, [None]*len(texts), embedder, tokenizer)
        else:
            assert len(texts)==len(metas)
            documents = process_batch(texts, metas, embedder, tokenizer)
        if documents_save_path is not None:
            save_documents(documents, documents_save_path)
        return Retriever(
            documents=documents,
            embedder=embedder,
            tokenizer=tokenizer,
            reranker=reranker,
            augmenter=augmenter
        )
    
    @staticmethod
    def from_pickle_dump(
        embedder:BaseEmbedder=None,
        tokenizer:Tokenizer=SPACE_TOKENIZER,
        reranker:BaseReranker=None,
        augmenter: BaseAugmenter = None,
        documents_save_path: str = None
    ) -> Retriever:
        documents = load_documents(documents_save_path)
        return Retriever(
            documents=documents,
            embedder=embedder,
            tokenizer=tokenizer,
            reranker=reranker,
            augmenter=augmenter
        )