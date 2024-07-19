from setuptools import setup, find_packages

setup(
    name='Retriever',
    version='0.1.0',
    url='https://github.com/thuwyh/Retriever',
    author='thuwyh',
    author_email='wuyhthu@gmail.com',
    description='A lightweight wrapper for high performance retrieval.',
    packages=find_packages(),    
    install_requires=[
        "openai",
        "pydantic",
        "diskcache",
        "coloredlogs",
        "platformdirs",
        "tqdm",
        "rank_bm25",
        "jieba",
        "hnswlib",
        "aiohttp"
    ],
)