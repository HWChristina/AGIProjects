#实现一个功能较为完整的RAG系统
import os
import json
from pathlib import Path
from pydantic.v1 import BaseModel

from llama_index.core import Settings #导入全局使用的语言模型库
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core import StorageContext
#混合检索导入库
from llama_index.core import SimpleKeywordTableIndex,VectorStoreIndex 
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import(
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
    QueryFusionRetriever     #确保导入QueryFusionRetriever
)
from typing import List
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine

from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import MarkdownNodeParser

from llama_index.core.postprocessor import SentenceTransformerRerank #重排序模型
from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.file import FlatReader
#自定义数据处理流程记载库
import chromadb
from chromadb.config import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
#定义多轮信息模版
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from llama_index.llms.openai import OpenAI #导入语言模型库
from llama_index.embeddings.openai import OpenAIEmbedding  #设定Embedding模型

# 加载环境变量
_ = load_dotenv(find_dotenv())
client = OpenAI()
#设置全局llm模型
Settings.llm = OpenAI(temperature=0, model="gpt-4o")

# 加载环境变量
if os.environ.get('CUR_ENV_IS_STUDENT','false')=='true':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3']= sys.modules.pop('pysqlite3')
chroma_client = chromadb.EphemeralClient(settings= Settings(allow_reset=True))

'''-----------文档加载------------'''
#1.加载本地数据
# 加载 pdf 文档
documents = SimpleDirectoryReader("./data", file_extractor={".pdf": PyMuPDFReader()}).load_data()

'''-----------文本切分与解析(Chunking)------------'''
#文本切分
node_parser = TokenTextSplitter(
    chunk_size = 300,#100,  #每个节点/Chunk的大小
    chunk_overlap = 100#50  #相邻节点/Chunk的 overlap 大小
)
#切分文档
nodes = node_parser.get_nodes_from_documents(
    documents
    #是否显示进度
    #show_progress=False
)
'''--------------灌库chromadb+检索----------------'''

# 全局设定Enbedding模型
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=512)


chroma_collection = chroma_client.create_collection("ingestion_demo")
#创建Vector Store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

#Storage Context是Vector Store的存储容器，用于存储文本等数据
storage_context = StorageContext.from_defaults(vector_store=vector_store)

'''-------------------创建index(在文档切分后构建索引)-----------------'''
#创建向量索引
vector_index  = VectorStoreIndex(nodes,storage_context = storage_context)
#创建关键字索引
keyword_index = SimpleKeywordTableIndex(nodes,storage_context=storage_context)

#获取retriever
class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""
    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes
#创建关键字检索器
vector_retriever=VectorIndexRetriever(index=vector_index,similarity_top_k=5)
#创建关键字检索器
keyword_retriever = KeywordTableSimpleRetriever(index = keyword_index)
custom_retriever = CustomRetriever(vector_retriever,keyword_retriever)

#使用QueryFusionRetriever
query_fusion_retriever = QueryFusionRetriever(
    retrievers=[custom_retriever],
    similarity_top_k = 5,
    num_queries=3,
    use_async=True,
)

#对检索后结果进行排序：
reranker = SentenceTransformerRerank(
    model = "BAAI/bge-reranker-large",
    top_n = 2
)
#创建单轮query engine
query_engine=RetrieverQueryEngine.from_args(
    query_fusion_retriever,
    node_postprocessors=[reranker]
)

#对话引擎
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine = query_engine, 
)
""" # 确保所有检索到的节点的 score 属性不为 None
def initialize_scores(nodes):
    for node in nodes:
        if node.score is None:
            node.score = 0.0 """
            
while True:
    question = input("User:")
    if question.strip()=="":
        break
    response = chat_engine.chat(question)
    print(f"AI:{response}")

""" #检索
query = "Llama2能商用么?"
print(f"Retrieving results for query: {query}")
#nodes = vector_retriever.retrieve(query)
retrieved_nodes = query_fusion_retriever.retrieve(query) """

#检查并初始化所有节点的score属性
""" for node_with_score in retrieved_nodes:
    if node_with_score.score is None:
        node_with_score.score = 0.0 """
#max_score = max(node_with_score.score if node_with_score.score is not None else 0, all_nodes[text].score if all_nodes[text].score is not None else 0)

#sorted_nodes = postprocessor.postprocess_nodes(retrieved_nodes,query_str = query)

""" #将检索结果作为上下文
context = " ".join([node.text for node in sorted_nodes]) """
    
""" #定义多轮消息
chat_text_qa_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="你必须根据用户提供的上下文回答问题。",
    ),
    ChatMessage(
        role=MessageRole.USER, 
        content=f"已知上下文：\n{context}\n\n问题:{query}"
    ),
]
#创建聊天模版
text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

#流式输出,使用生成模型生成答案
llm = OpenAI(temperature=0, model="gpt-3.5")
response = llm.complete(
    text_qa_template.format(
        question=query,
        context=context
    )
)
#输出结果
print(response.text) """