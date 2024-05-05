from abc import ABC
from typing import List

class BaseEmbedder(ABC):

    dim: int

    def get_embedding(self, text:str, **kwargs) -> List[float]:
        pass

    async def aget_embedding(self, text:str, **kwargs) -> List[float]:
        pass