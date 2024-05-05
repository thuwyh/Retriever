from abc import ABC
from typing import List
from pydantic import BaseModel

class RerankResult(BaseModel):
    index: int
    score: float

class BaseReranker(ABC):

    def rerank(self, query:str, texts:List[str], **kwargs) -> List[RerankResult]:
        pass

    async def arerank(self, query:str, texts:List[str], **kwargs) -> List[RerankResult]:
        pass

