import asyncio
import logging
from pathlib import Path
from time import perf_counter
from typing import List
from diskcache import Cache
import openai
import coloredlogs
from platformdirs import user_cache_dir
import json
from tqdm import tqdm

from retriever.augmenter.base_augmenter import BaseAugmenter
from retriever.types import Document
from retriever.embedder.base_embedder import BaseEmbedder
from retriever.tokenizer import Tokenizer

logger = logging.getLogger("OAID2QAugmenter")
logger.setLevel(logging.INFO)
coloredlogs.install(level=logging.INFO, logger=logger)

DEFAULT_TEMPLATE = """Given the following article:
{article}

Generate 5 search queries that can be answered according to the article.
Questions should be self-contained and do not use any referents.
Questions should be in the same language of the article.

Please output in the following JSON format:
{{
    "questions": [
        "question 1",
        "question 2",
        "question 3",
        "question 4",
        "question 5"
    ]
}}
Your output:
"""


class OpenAID2QAugmenter(BaseAugmenter):

    def __init__(
        self, api_key, base_url="https://api.openai.com/v1", template=DEFAULT_TEMPLATE
    ) -> None:
        self.oai_client = openai.AsyncClient(api_key=api_key, base_url=base_url)
        self.template = template
        cachedir = user_cache_dir("Retriever", "thuwyh")
        cache_root = Path(cachedir) / "oai_d2q"
        self.cache = Cache(str(cache_root))

    async def augment_single_article(
        self, doc: Document, tokenizer: Tokenizer, sem, embedder: BaseEmbedder = None,
        pbar:tqdm=None
    ) -> List[Document]:
        if doc.payload in self.cache:
            queries = self.cache[doc.payload]
        else:
            prompt = self.template.format(article=doc.payload)
            async with sem:
                resp = await self.oai_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-4o",
                    stream=False,
                    temperature=0.7,
                    response_format={"type": "json_object"},
                )
            try:
                d = json.loads(resp.choices[0].message.content.strip())
                queries = d["questions"]
                self.cache[doc.payload] = queries
            except:
                queries = []
        ret = []
        logger.info(f"{doc.payload[:20]}, {queries}")
        for q in queries:
            tokens = tokenizer.tokenize(q)
            new_doc = Document(tokens=tokens, payload=q, meta={"original_docid": doc.id})
            if embedder is not None:
                async with sem:
                    embedding = await embedder.aget_embedding(q)
            new_doc.vector = embedding
            ret.append(new_doc)
        if pbar is not None:
            pbar.update(1)
        return ret

    def augment(
        self,
        documents: List[Document],
        tokenizer: Tokenizer,
        embedder: BaseEmbedder = None,
        parallel: int = 5,
    ) -> List[Document]:
        async def _process():
            sem = asyncio.Semaphore(parallel)
            tasks = []
            pbar = tqdm(total=len(documents), desc="augmentation")
            for d in documents:
                tasks.append(
                    asyncio.create_task(
                        self.augment_single_article(
                            d, tokenizer, sem, embedder=embedder, pbar=pbar
                        )
                    )
                )
            results = await asyncio.gather(*tasks)
            return sum(results, [])

        start = perf_counter()
        logger.info(f"Augmenting starts. Total tasks: {len(documents)}")
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(_process())
        logger.info(f"{len(results)} aug docs generated. all tasks finished in {perf_counter()-start:.3f}s")
        return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    rewriter = OpenAID2QAugmenter(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    print(
        rewriter.rewrite(
            "基础配置有哪些",
            [
                (
                    "我想要了解百度云的mongodb",
                    "您好!非常感谢您对百度云的关注。关于您想了解的百度云的MongoDB服务，我们提供的是云数据库 DocDB for MongoDB.这款产品特别适用于终端设备产生的数据，它提供了实时数据的存储能力，并与云上专用计算引擎对接，便于进行大数据分析。",
                )
            ],
        )
    )
    print(
        rewriter.rewrite(
            "定价",
            [
                (
                    "我想要了解百度云的mongodb",
                    "您好!非常感谢您对百度云的关注。关于您想了解的百度云的MongoDB服务，我们提供的是云数据库 DocDB for MongoDB.这款产品特别适用于终端设备产生的数据，它提供了实时数据的存储能力，并与云上专用计算引擎对接，便于进行大数据分析。",
                ),
                ("基础配置有哪些", "基础配置有小型Mongodb， 中型Mongodb和大型Mongodb"),
            ],
        )
    )
