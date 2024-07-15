#实现基于langchain框架的ChatPDF后端代码
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_milvus.vectorstores import Milvus
#from langchain_community.retrievers import MilvusRetriever
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

#3.灌库
embeddings_model = OpenAIEmbeddings(model = "text-embedding-ada-002")

#获取嵌入,嵌入后由Milvus自动生成id(auto_id=True)
embeddings = embeddings_model.embed_documents(texts_content)

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
retriever = Milvus(collection_name,connection_args={"host": "localhost", "port": "19530"})


#retriever = db.as_retriever(search_kwargs={"k": 2})

#包装成Runnable对象

# 包装成Runnable对象
class MilvusRetriever(Runnable):
    def __init__(self, retriever):
        self.retriever = retriever

    def invoke(self, query, config):
        query_embedding = embeddings_model.embed_query(query)
        #执行搜索
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=5,
            output_fields=["id"]
        )
        relevant_texts = [id_text[id] for result in results for id in result.ids]
        
        return relevant_texts
        #return self.retriever.similarity_search(query)
    def __call__(self, query, config):
        return self.invoke(query, config)

milvus_retriever = MilvusRetriever(retriever=retriever)

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
    {"question":RunnablePassthrough(),"context":milvus_retriever}
    | prompt
    | model
    | StrOutputParser()
)
query = "Llama2有多少参数?"
response = rag_chain.invoke(query)
print(f"AI:{response}")
