#实现基于langchain框架的ChatPDF后端代码
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_milvus.vectorstores import Milvus
from langchain_milvus import Milvus, Zilliz
from langchain_community.retrievers import MilvusRetriever
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain.chains import RetrievalQA
import json
from pydantic import BaseModel, Field
from pymilvus import connections
from pymilvus import Collection,DataType,FieldSchema,CollectionSchema, utility
#------加入用pipeline试调用promptTemplate,LLM,Outputparser----#
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable,RunnablePassthrough
#1.文档加载
loader = PyMuPDFLoader("./data/llama2-extracted.pdf")

pages = loader.load_and_split()

#print (pages[0].page_content)
#2.文档处理器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=100,
    length_function = len,
    add_start_index = True,
    )

texts = text_splitter.create_documents(
    [page.page_content for page in pages[:3]]
)
#提取文本内容
texts_content = [text.page_content for text in texts]
#texts_content = text_splitter.split_documents(pages)

#3.灌库
embeddings_model = OpenAIEmbeddings(model = "text-embedding-ada-002")

#获取嵌入,嵌入后由Milvus自动生成id(auto_id=True)[不可行，会出现id out of index的问题]
embeddings = embeddings_model.embed_documents(texts_content)

#手动创建id
ids = list(range(len(embeddings)))
insert_data = [ids,embeddings]
#4.向量数据库与向量检索
#利用docker环境使用Milvus
# 连接到Milvus
connections.connect("default", host="localhost", port="19530")
# 创建集合
collection_name = "document_collection"
# 检查集合是否存在并处理
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
# 定义字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, description="id",is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536) 
]
# 创建Schema时启用动态字段
schema = CollectionSchema(fields, "document collection")
# 创建Collection
collection = Collection(collection_name, schema)
# 插入向量
#collection.insert([embeddings])
# 插入向量
""" entities = [
    {"name": "id", "values": ids, "type":DataType.INT64},
    {"name": "embedding", "values": embeddings, "type": DataType.FLOAT_VECTOR}
] """
# 对于集合的每个字段，构造插入数据
collection.insert(insert_data)
id_text = {i: text for i, text in enumerate(texts_content)}
#创建索引
index_params = {
    "index_type": "IVF_FLAT",  # 可以选择其他索引类型，如 IVFFLAT, IVFSQ8, ANNOY
    "params": {"nlist": 128},
    "metric_type": "L2"
}
collection.create_index("embedding", index_params)
#加载集合
collection.load()
#检索top-3的结果
#创建检索器
db = Milvus.from_documents(
    documents=texts,
    embedding=embeddings_model,
    connection_args={"host": "localhost", "port": "19530"},
    #drop_old=True,
    )
query = "Llama2有多少参数?"
db.max_marginal_relevance_search(query,k=3)
milvus_retriever = db.as_retriever(search_kwargs={"k": 3})

def format_texts(texts):
    return "\n\n".join(text.page_content for text in texts)
template ="""
   Answer the question based only on the following context:
   {context}

   Question:{question} 
"""
prompt = ChatPromptTemplate.from_template(template)

#初始化模型
model = ChatOpenAI(temperature=0,model="gpt-4o")
#Chain
rag_chain=(
    {"question":RunnablePassthrough(),"context":milvus_retriever|format_texts}
    | prompt
    | model
    | StrOutputParser()
)

response = rag_chain.invoke(query)
print(f"AI:{response}")
