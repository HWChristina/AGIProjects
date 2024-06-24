
from flask import Flask, request, jsonify, render_template
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

# 加载环境变量
_ = load_dotenv(find_dotenv())

client = OpenAI()

app = Flask(__name__, static_folder='static', template_folder='templates')

""" -----------文本提取和切割--------------- """
# PDF可按照页码进行提取
def extract_text_by_page(pdf_path, start_page, end_page):
    extract_text=[]
    for page_layout in extract_pages(pdf_path):
        page_number = page_layout.pageid
        if start_page <= page_number <= end_page:
            page_text=""
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    page_text += element.get_text()
            extract_text.append(page_text)
    return extract_text

# 段落切割
def split_into_paragraphs(texts):
    paragraphs = []
    for text in texts:
        paragraphs.extend(text.split('\n\n'))
    return paragraphs

""" -----------文本向量化--------------- """
# 嵌入生成
def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    return embeddings

""" -----------将向量放入向量数据库（更快检索到相似向量）------------ """
# 将向量插入到 Milvus
def insert_embeddings(paragraphs, embeddings):
    # 为每个段落生成唯一的 ID
    ids = list(range(len(paragraphs)))
    
    # 组合 ID 和 embeddings 成为二维列表
    data = [ids, embeddings]
    
    # 插入数据到 Milvus
    collection.insert(data)
    
    return ids

# 向量检索与重排序
def search_and_rerank(query, top_k=5):
    query_embedding = get_embeddings([query])[0]
    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    collection.load()
    
    results = collection.search([query_embedding], "embedding", search_params, limit=top_k)
    # 获取搜索结果
    candidate_paragraphs = [paragraphs[result.id] for result in results[0] if result.id < len(paragraphs)]
    # rerank(两两比较)
    rerank_scores = cross_encoder.predict([(query, para) for para in candidate_paragraphs])
    # 根据 rerank_scores 排序
    sorted_candidates = sorted(zip(candidate_paragraphs, rerank_scores), key=lambda x: x[1], reverse=True)
    return sorted_candidates

# 答案生成
def generate_answer(query, context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        temperature=0,
        max_tokens=512,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"}    
        ]      
    )
    return response.choices[0].message.content
"""----------前端增加代码----------"""
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    start_page = int(request.form.get('start_page', 1))
    end_page = int(request.form.get('end_page', 1))

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file.save(file.filename)
    pages_text = extract_text_by_page(file.filename, start_page, end_page)
    # 全局变量
    global paragraphs
    paragraphs = split_into_paragraphs(pages_text)
    paragraph_embeddings = get_embeddings(paragraphs)

    # Milvus 配置
    connections.connect("default", host="localhost", port="19530")

    collection_name = "pdf_collection"

    # 检查集合是否存在并处理
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    # 创建集合和索引
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, description="id", is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)  # 1536 为 text-embedding-ada-002 的向量维度
    ]
    schema = CollectionSchema(fields, "pdf_embeddings")
    global collection
    collection = Collection(collection_name, schema)

    # 插入向量数据并获取 ids
    ids = insert_embeddings(paragraphs, paragraph_embeddings)

    # 创建索引
    index_params = {
        "index_type": "IVF_FLAT",  # 可以选择其他索引类型，如 IVFFLAT, IVFSQ8, ANNOY
        "params": {"nlist": 128},
        "metric_type": "L2"
    }
    collection.create_index("embedding", index_params)

    # 加载集合
    collection.load()
    return jsonify({'message': 'PDF uploaded and embeddings stored'})

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data['question']
    top_results = search_and_rerank(question)
    context = top_results[0][0]
    answer = generate_answer(question, context)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)


