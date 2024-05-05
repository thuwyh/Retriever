from typing import List
import logging
import coloredlogs
import asyncio
from time import perf_counter
import numpy as np
from sanic import Sanic
from sanic.response import json as json_response
from sanic.views import HTTPMethodView
from sanic.worker.loader import AppLoader

from retriever.document import Document
from retriever.tokenizer import Tokenizer
from retriever.embedder import BaseEmbedder
from retriever.reranker import BaseReranker
from rank_bm25 import BM25Okapi
import hnswlib

class Retriever(HTTPMethodView):

    def __init__(self, 
                 documents:List[Document], 
                 embedder: BaseEmbedder,
                 tokenizer:Tokenizer,
                 reranker: BaseReranker=None,
                 log_level:int=logging.INFO) -> None:
        self.logger = logging.getLogger("Retriever")
        self.logger.setLevel(log_level)
        coloredlogs.install(level=log_level, logger=self.logger)

        self.tokenizer = tokenizer
        start = perf_counter()
        self.logger.info("Building BM25 index...")
        self.bm25_index = BM25Okapi([x.tokens for x in documents])
        self.logger.info(f"BM25 index built in {perf_counter()-start:.3f}s.")

        self.embedder = embedder
        self.logger.info("Building HNSW index...")
        self.ann = hnswlib.Index(space = 'cosine', dim = embedder.dim)
        self.ann.init_index(max_elements = len(documents), ef_construction = 200, M = 16)
        ids = np.arange(len(documents))
        self.ann.add_items(np.array([d.vector for d in documents], dtype=np.float32), ids)
        self.ann.set_ef(50)
        self.logger.info(f"HNSW index built in {perf_counter()-start:.3f}s.")

        self.reranker = reranker
        self.documents = documents

    async def aretrieve(self,
                        text_query:str=None, 
                        top_n:int=5,
                        with_payload=False):
        n = min(len(self.documents), top_n)
        tokenized_query = self.tokenizer.tokenize(text_query)
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        top_n_index = doc_scores.argsort()[-n:][::-1]

        query_vector = await self.embedder.aget_embedding(text_query, force_instruction='')
        query_vector = np.float32(np.array([query_vector]))
        labels, distances = self.ann.knn_query(query_vector, k = top_n)
        
        ret = []
        picked_ind = set()
        for rank, ind in enumerate(labels[0]):
            to_append = {
                'id': self.documents[ind].id,
                'index': int(ind),
                'cosine_score': float(distances[0][rank]),
                'bm25_score': doc_scores[ind],
                'source': 'ann'
            }
            picked_ind.add(ind)
            if with_payload:
                to_append['payload'] = self.documents[ind].payload
            ret.append(to_append)
        for ind in top_n_index:
            if ind in picked_ind:
                continue
            to_append = {
                'id': self.documents[ind].id,
                'index': int(ind),
                'cosine_score': -1,
                'bm25_score': doc_scores[ind],
                'source': 'bm25'
            }
            picked_ind.add(ind)
            if with_payload:
                to_append['payload'] = self.documents[ind].payload
            ret.append(to_append)
        if self.reranker is not None:
            # do rerank
            texts = [self.documents[x['index']].payload for x in ret]
            rerank_results = sorted(await self.reranker.arerank(text_query, texts), key=lambda x: x.score, reverse=True)
            ret = [ret[x.index] for x in rerank_results[:top_n]]
            for r, rerank_r in zip(ret, rerank_results):
                r['rerank_score'] = rerank_r.score
        return ret

    def retrieve(self, 
                 text_query:str=None, 
                 top_n:int=5,
                 with_payload=False):
        loop = asyncio.get_event_loop()
        ret = loop.run_until_complete(self.aretrieve(text_query, top_n, with_payload))
        return ret
    
    async def get(self, request):
        ret = await self.aretrieve(**request.json)
        return json_response(ret)
    
    def get_app(self):
        # web
        app = Sanic('Retriever')
        app.add_route(self.as_view(), '/retrieve')
        return app


    # def run(self, host:str, port:int, **kwargs): 
    #     self.app.add_route(self.__class__.as_view(), "/retrieve")
    #     self.app.run(host, port, **kwargs)