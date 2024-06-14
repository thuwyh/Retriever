from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid

from retriever.reranker.base_reranker import BaseReranker

class Source(str, Enum):
    BM25="bm25"
    ANN="ann"
    AUX_BM25="aux_bm25"
    AUX_ANN = "aux_ann"

class BaseField(BaseModel):
    should_index: bool

class VectorField(BaseModel):
    vector: List[float]

class MultiVectorField(BaseModel):
    vectors: List[List[float]]

class TokenField(BaseModel):
    tokens: List[str]

class Document(BaseModel):
    id: str = str(uuid.uuid4())
    vector: Optional[List[float]]=None
    vector_alias: Optional[List[List[float]]]=None
    tokens: Optional[List[str]]
    payload: str
    meta: Optional[object]
    
class RetrievedItem(BaseModel):
    doc: Document
    source: Source
    cosine_score: Optional[float]
    bm25_score: Optional[float]
    rerank_score: Optional[float]

class RetrievedResult(BaseModel):
    results: List[RetrievedItem]
    idx: set = Field(default_factory=set)

    def append(self, result:RetrievedItem):
        if result.doc.id in self.idx:
            return
        self.results.append(result)
        self.idx.add(result.doc.id)

    async def rerank(self, text_query, reranker:BaseReranker):
        texts = [x.doc.payload for x in self.results]
        rerank_results = await reranker.arerank(text_query, texts)
        for rr in rerank_results:
            self.results[rr.index].rerank_score = rr.score
        self.results = sorted(self.results, key=lambda x: x.rerank_score, reverse=True)