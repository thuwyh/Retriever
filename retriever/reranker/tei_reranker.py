from typing import List
from pathlib import Path
from diskcache import Cache
import aiohttp
import requests
import logging
import coloredlogs
from platformdirs import user_cache_dir

from retriever.reranker.base_reranker import BaseReranker, RerankResult
from retriever.utils import timer

logger = logging.getLogger("Retriever.TEIReranker")
logger.setLevel(logging.INFO)
coloredlogs.install(level=logging.INFO, logger=logger)

class TEIReranker(BaseReranker):

    def __init__(self, 
                 tei_addr:str, 
                 cache_name:str=None) -> None:
        super().__init__()
        self.tei_addr = tei_addr

        if cache_name is not None:
            cachedir = user_cache_dir("Retriever.TEIReranker", "thuwyh")
            cache_root = Path(cachedir) / cache_name
            self.cache = Cache(str(cache_root))
        else:
            self.cache = None

    @timer(logger=logger)
    async def arerank(self, query:str, texts: List[str]) -> List[RerankResult]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.tei_addr}/rerank", 
                json={'query':query, 
                      'texts': texts,
                      'truncate':True}) as resp:
                res = await resp.json()
                return [RerankResult(**x) for x in res]
            
    @timer(logger=logger)
    def rerank(self, query:str, texts: List[str]) -> List[RerankResult]:
        res = requests.post(f"{self.tei_addr}/rerank", 
                            json={'query':query, 
                      'texts': texts,
                      'truncate':True})
        return [RerankResult(**x) for x in res.json()]
    
if __name__ == '__main__':
    client = TEIReranker("http://localhost:8081")
    print(client.rerank("你好", ["你们好","我爱北京天安门"]))

    # import asyncio
    # print(asyncio.run(client.arerank("你好")))