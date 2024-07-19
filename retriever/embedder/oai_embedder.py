from typing import List
from pathlib import Path
from diskcache import Cache
from hashlib import md5
import aiohttp
import requests
from platformdirs import user_cache_dir

from retriever.embedder.base_embedder import BaseEmbedder

class OAIEmbedder(BaseEmbedder):

    def __init__(self, 
                 tei_addr:str, 
                 dim=768, 
                 normalize:bool=True, 
                 instruction:str=None,
                 cache_name:str=None) -> None:
        super().__init__()
        self.tei_addr = tei_addr
        self.instruction = instruction
        self.dim = dim
        self.normalize = normalize

        if cache_name is not None:
            cachedir = user_cache_dir("Retriever.OAIEmbedder", "thuwyh")
            cache_root = Path(cachedir) / cache_name
            self.cache = Cache(str(cache_root))
        else:
            self.cache = None

    async def aget_embedding(self, text: str, force_instruction:str=None) -> List[float]:
        instruction = force_instruction or self.instruction
        if instruction is not None:
            input_query = instruction + text
        else:
            input_query = text
        key = md5(input_query.encode()).hexdigest()
        if self.cache is not None:
            if key in self.cache:
                return self.cache[key]
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.tei_addr}/embed", json={'inputs':input_query, 'truncate':True}) as resp:
                res = await resp.json()
                if self.cache is not None:
                    self.cache[key] = res[0]
                return res[0]
            

    def get_embedding(self, text: str, force_instruction) -> List[float]:
        instruction = force_instruction or self.instruction
        if instruction is not None:
            input_query = instruction + text
        else:
            input_query = text
        key = md5(input_query.encode()).hexdigest()
        if self.cache is not None:
            if key in self.cache:
                return self.cache[key]
        res = requests.post(f"{self.tei_addr}/embed", json={'inputs':input_query, 'truncate':True})
        if self.cache is not None:
            self.cache[key] = res[0]
        return res.json()[0]
    
if __name__ == '__main__':
    client = TEIEmbedder("http://localhost:8080", dim=512, instruction="为这个句子生成表示以用于检索相关文章：")
    print(client.get_embedding("你好"))

    import asyncio
    print(asyncio.run(client.aget_embedding("你好")))