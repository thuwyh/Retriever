from typing import List, Optional
from pydantic import BaseModel
import uuid

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
    payload: object
    meta: Optional[object]
    