from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# 加载 .env 到环境变量
_ = load_dotenv(find_dotenv())
client = OpenAI()

app = Flask(__name__)
CORS(app)  # 允许所有域名的跨域请求

@app.route('/ask', methods=['POST'])
def ask():
    # 获取用户问题
    user_question = request.json['question']
    # 调用模型API获取答案
    answer = get_answer_from_model(user_question)

    return jsonify({'answer': answer})

def get_answer_from_model(question):
    # 调用模型API获取答案
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=question,
        max_tokens=512,
        stream=False
    )
    
    print("Response:", response)

    # 获取答案
    answer = response.choices[0].text.strip()
    return answer

if __name__ == '__main__':
    app.run(debug=True, port=5000)
