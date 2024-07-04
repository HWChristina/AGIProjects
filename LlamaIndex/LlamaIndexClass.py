#基于LlamaIndex SDK实现的小型RAG系统
#加载库
import json
import os
from pydantic.v1 import BaseModel
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.feishu_docs import FeishuDocsReader
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.readers.file import FlatReader
from llama_index.core.node_parser import MarkdownNodeParser
from pathlib import Path
from llama_index.core import VectorStoreIndex,StorageContext
#自定义数据处理流程加载库
#from llama_index.vector_stores.chroma import ChromaVectorStore
#from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
#from llama_index.core import VectorStoreIndex
#from llama_index.readers.file import PyMuPDFReader
#import nest_asyncio
#检索后排序模型导入库
from llama_index.core.postprocessor import SentenceTransformerRerank
#Milvus数据库
#from llama_index.vector_stores.milvus import MilvusVectorStore    
#from pymilvus import connections
import chromadb
from chromadb.config import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore 
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import time

# 加载环境变量
_ = load_dotenv(find_dotenv())
client = OpenAI()
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"耗时{self.interval*100}ms")

# 加载环境变量
if os.environ.get('CUR_ENV_IS_STUDENT','false')=='true':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3']= sys.modules.pop('pysqlite3')
chroma_client = chromadb.EphemeralClient(settings= Settings(allow_reset=True))


    
def show_json(data):
    """用于展示json数据"""
    if isinstance(data, str):
        obj = json.loads(data)
        print(json.dumps(obj, indent=4))
    elif isinstance(data, dict) or isinstance(data, list):
        print(json.dumps(data, indent=4))
    elif issubclass(type(data), BaseModel):
        print(json.dumps(data.dict(), indent=4, ensure_ascii=False))

def show_list_obj(data):
    """用于展示一组对象"""
    if isinstance(data, list):
        for item in data:
            show_json(item)
    else:
        raise ValueError("Input is not a list")
    
'''-----------文档加载------------'''
#1.加载本地数据
""" reader = SimpleDirectoryReader(
    input_dir = "./data",
    recursive =False,  #是否递归遍历子目录
    required_exts = [".pdf"],  #只读取后缀为.pdf的文件
    #指定特定的文件加载器读取文件
    file_extractor = {".pdf": PyMuPDFReader()}
)
documents = reader.load_data() """
#print(documents[0].text)
#2.加载网页/服务器上的文档数据（需要API）(以飞书文档为例)
""" app_id = "cli_a6f43c5433be500c"
app_secret = "O895punGa9F6C29coN0oEbTUUyEhzX88"
# https://agiclass.feishu.cn/docx/FULadzkWmovlfkxSgLPcE4oWnPf
# 链接最后的 "FULadzkWmovlfkxSgLPcE4oWnPf" 为文档 ID 
doc_ids = ["FULadzkWmovlfkxSgLPcE4oWnPf"]
#定义飞书文档加载器
loader = FeishuDocsReader(app_id, app_secret)
#加载文档
documents = loader.load_data(document_ids=doc_ids)
#显示前1000字符
#print(documents[0].text[:1000]) """
'''-----------文本切分与解析(Chunking)------------'''
""" #文本切分
node_parser = TokenTextSplitter(
    chunk_size = 300,#100,  #每个节点/Chunk的大小
    chunk_overlap = 100#50  #相邻节点/Chunk的 overlap 大小
)
#切分文档
nodes = node_parser.get_nodes_from_documents(
    documents
    #是否显示进度
    #show_progress=False
) """

#文本解析
#解析md文档
""" md_docs = FlatReader().load_data(Path("./data/ChatALL.md"))
parser = MarkdownNodeParser()
nodes = parser.get_nodes_from_documents(md_docs)
show_json(nodes[2])
show_json(nodes[3]) """


'''-----------索引(Indexing)与检索(Retrieval)------------'''
#1.向量检索
#(1)直接利用SimpleVectorStore在内存中构建一个向量数据库并建立索引
#构建index
""" index = VectorStoreIndex(nodes)

#获取retriever
vector_retriever = index.as_retriever(
    similarity_top_k=2  #返回前两个结果
)

#检索
results = vector_retriever.retrieve("Llama2有多少参数")
show_list_obj(results)
 """
#(2)使用自定义的Vector Store(chromadb)

#chroma_client.reset()
chroma_collection = chroma_client.create_collection("ingestion_demo")
#创建Vector Store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
'''---------------直接利用pipeline实现多步骤操作----------------'''

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=300,chunk_overlap=100),#按照句子边界进行文本切分的工具
        TitleExtractor(),    #利用LLM对文本生成标题
        OpenAIEmbedding(),   #将文本向量化
    ],
    vector_store=vector_store,  #灌库
)
documents = SimpleDirectoryReader(
    "./data",
    required_exts = [".pdf"],
    file_extractor ={".pdf":PyMuPDFReader()}
).load_data()


#Storage Context是Vector Store的存储容器，用于存储文本等数据
#storage_context = StorageContext.from_defaults(vector_store=vector_store)

#创建index(在文档切分后构建索引)(pipeline:为灌库后的数据块创建索引)
#index  = VectorStoreIndex(nodes,storage_context = storage_context)

#创建index(pipeline:为灌库后的数据块创建索引)
index = VectorStoreIndex.from_vector_store(vector_store)

"""
#获取retriever
vector_retriever = index.as_retriever(
    similarity_top_k=5  #返回前多少个结果
)
#检索
nodes = vector_retriever.retrieve("Llama2能够商用么?")
for i ,node in enumerate(nodes):
    print(f"[{i}]{node.text}") 
"""


#检索结果输出
#results = vector_retriever.retrieve("Llama2有多少参数")

#show_list_obj(results[:1])

'''-----------本地保存Ingestionpipeline缓存(start)----------'''
""" #pipeline.persist("./pipeline_storage")

new_pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=300,chunk_overlap=100),#按照句子边界进行文本切分的工具
        TitleExtractor(),    #利用LLM对文本生成标题
        OpenAIEmbedding()    #将文本向量化
    ],
    vector_store=vector_store,  #灌库
)
#缓存
new_pipeline.load("./pipeline_storage") """
'''-----------本地保存Ingestionpipeline缓存(end)-----------'''


#计时
with Timer():
    #Ingest directly into a vector db
    pipeline.run(documents = documents)
    #缓存：
    #new_pipeline.run(documents = documents)
    

#获取retriever
vector_retriever = index.as_retriever(
    similarity_top_k=5  #返回前多少个结果
)

#检索
query = "Llama2能商用么?"
print(f"Retrieving results for query: {query}")
nodes = vector_retriever.retrieve(query)

#检索后排序：
postprocessor = SentenceTransformerRerank(
    model = "BAAI/bge-reranker-large",top_n = 2
)

nodes = postprocessor.postprocess_nodes(nodes,query_str = query)

# 输出结果
if nodes:
    for i, node in enumerate(nodes):
        print(f"[{i}] {node.text}")
else:
    print("No results found.")
    
    
    
"""-----------生成回复(QA&Chat)-----------
#单轮问答(query Engine)
#1.正常输出
qa_engine = index.as_query_engine()
response = qa_engine.query("Llama2有多少参数?") 
print(response)
#2.流式输出
qa_engine = index.as_query_engine(streaming=True)
response = qa_engine.query("Llama2有多少参数?")
response.print_response_stream()
#多轮对话(chat Engine)
#1.正常输出
chat_engine = index.as_chat_engine()
response = chat_engine.chat("Llama2有多少参数?")
print(response)
response = chat_engine.chat("how many at most?")
print(response)
#2.流式输出
chat_engine = index.as_chat_engine()
streaming_response = chat_engine.stream_chat("Llama 2有多少参数？")
for token in streaming_response.response_gen:
    print(token, end="")
"""

