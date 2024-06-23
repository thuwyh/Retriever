from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid

from retriever.reranker.base_reranker import BaseReranker


class Source(str, Enum):
    BM25 = "bm25"
    ANN = "ann"
    AUX_BM25 = "aux_bm25"
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
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vector: Optional[List[float]] = None
    vector_alias: Optional[List[List[float]]] = None
    tokens: Optional[List[str]]
    payload: str
    meta: Optional[object] = None


class RetrievedItem(BaseModel):
    doc: Document
    source: Source
    cosine_score: Optional[float] = -1
    bm25_score: Optional[float] = -1
    rerank_score: Optional[float] = -1

    def to_json(
        self,
        corpus: Dict[str, Document],
        with_payload=True,
        with_meta=True,
        debug=False,
    ):
        if self.source in [Source.AUX_ANN, Source.AUX_BM25]:
            doc = corpus[self.doc.meta["original_docid"]]
        else:
            doc = self.doc
        ret = {
            "id": doc.id,
            "rerank_score": self.rerank_score
        }
        if debug:
            ret.update(
                {
                    "source": self.source.value,
                    "cosine_score": self.cosine_score,
                    "bm25_score": self.bm25_score
                }
            )
        if with_payload:
            ret["payload"] = doc.payload
        if with_meta:
            ret["meta"] = doc.meta
        return ret


class RetrievedResult(BaseModel):
    results: List[RetrievedItem] = Field(default_factory=list)
    idx: set = Field(default_factory=set)

    def append(self, result: RetrievedItem):
        # original doc and aux doc should have different id
        if result.doc.id in self.idx:
            return
        self.results.append(result)
        self.idx.add(result.doc.id)

    async def rerank(self, text_query, reranker: BaseReranker):
        texts = [x.doc.payload for x in self.results]
        rerank_results = await reranker.arerank(text_query, texts)
        for rr in rerank_results:
            self.results[rr.index].rerank_score = rr.score
        self.results = sorted(self.results, key=lambda x: x.rerank_score, reverse=True)

    def to_json(
        self,
        corpus: Dict[str, Document],
        top_n: int = 5,
        with_payload=True,
        with_meta=True,
    ):
        all_ids = set()
        ret = []
        for r in self.results:
            to_append = r.to_json(
                corpus, with_payload=with_payload, with_meta=with_meta
            )
            if to_append["id"] in all_ids:
                continue
            ret.append(to_append)
            all_ids.add(to_append["id"])
            if len(ret) == top_n:
                break
        return ret
