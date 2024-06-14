from typing import List
from time import perf_counter
import logging
import asyncio
import coloredlogs
from retriever.augmenter.base_augmenter import BaseAugmenter
from retriever.retriever import Retriever
from retriever.tokenizer import Tokenizer, SPACE_TOKENIZER
from retriever.embedder import BaseEmbedder
from retriever.types import Document


logger = logging.getLogger("DocProcessor")
logger.setLevel(logging.INFO)
coloredlogs.install(level=logging.INFO, logger=logger)


async def aprepare_doc(
    text: str, embedder: BaseEmbedder, tokenizer: Tokenizer, sem: asyncio.Semaphore, meta:object=None
) -> Document:
    tokens = tokenizer.tokenize(text)
    doc = Document(tokens=tokens, payload=text, meta=meta)
    if embedder is not None:
        async with sem:
            embedding = await embedder.aget_embedding(text)
        doc.vector = embedding
    return doc


def process_batch(
    texts: List[str],
    metas: List[object],
    embedder: BaseEmbedder = None,
    tokenizer: Tokenizer = SPACE_TOKENIZER,
    parallel: int = 5,
) -> List[Document]:
    async def _process():
        sem = asyncio.Semaphore(parallel)
        tasks = []
        for t, m in zip(texts, metas):
            tasks.append(
                asyncio.create_task(
                    aprepare_doc(t, embedder, tokenizer, sem, meta=m)
                )
            )
        results = await asyncio.gather(*tasks)
        return results

    start = perf_counter()
    logger.info(f"Processing starts. Total tasks: {len(texts)}")
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(_process())
    logger.info(f"all tasks finished in {perf_counter()-start:.3f}s")
    return results
