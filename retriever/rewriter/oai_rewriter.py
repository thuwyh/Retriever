import asyncio
import openai

DEFAULT_TEMPLATE = """Given a user query:
{query}

and the conversation history between the user and system:
{history}

please rewrite the user query, so that the rewritten query contains all the context and key info to retrieve relative information.
Example:
query: what is its population?
history: 
user: Where is the capital of China?
bot: Beijing

expected rewritten query: what is the population of Beijing

Reply only the rewritten query in the same language as original query.
Your output:
"""
                 
class OpenAIRewriter:

    def __init__(self, 
                 api_key, 
                 base_url="https://api.openai.com/v1",
                 template=DEFAULT_TEMPLATE) -> None:
        self.oai_client = openai.AsyncClient(
            api_key=api_key,
            base_url=base_url
        )
        self.template = template

    async def arewrite(self, query, history):
        if isinstance(history, list):
            history_text = ""
            for u, b in history:
                history_text += f"User: {u}\nBot: {b}\n"
        else:
            history_text = history

        prompt = self.template.format(
            query=query,
            history=history_text
        )
        resp = await self.oai_client.chat.completions.create(
            messages=[{'role':'user', 'content':prompt}],
            model="gpt-4o",
            stream=False,
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()


    def rewrite(self, query, history):
        try:
            loop = asyncio.get_event_loop()
            ret = loop.run_until_complete(self.arewrite(query, history))
        except:
            ret = asyncio.run(self.arewrite(query, history))
        return ret
    
if __name__ == '__main__':
    from dotenv import load_dotenv
    import os
    load_dotenv()
    rewriter = OpenAIRewriter(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    print(
        rewriter.rewrite(
            "基础配置有哪些",
            [("我想要了解百度云的mongodb","您好!非常感谢您对百度云的关注。关于您想了解的百度云的MongoDB服务，我们提供的是云数据库 DocDB for MongoDB.这款产品特别适用于终端设备产生的数据，它提供了实时数据的存储能力，并与云上专用计算引擎对接，便于进行大数据分析。")]
        )
    )
    print(
        rewriter.rewrite(
            "定价",
            [("我想要了解百度云的mongodb","您好!非常感谢您对百度云的关注。关于您想了解的百度云的MongoDB服务，我们提供的是云数据库 DocDB for MongoDB.这款产品特别适用于终端设备产生的数据，它提供了实时数据的存储能力，并与云上专用计算引擎对接，便于进行大数据分析。"),
             ("基础配置有哪些","基础配置有小型Mongodb， 中型Mongodb和大型Mongodb")]
        )
    )