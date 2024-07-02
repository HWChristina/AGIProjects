#stream模式 作用：不会一次返回完整的JSON结构，需要拼接后再使用

from openai import OpenAI
from dotenv import load_dotenv,find_dotenv
from math import*
import sqlite3
import json
import requests

_ = load_dotenv(find_dotenv())

client = OpenAI()

def print_json(data):
    """
    打印参数。如果参数是有结构的(如字典或列表),则以格式化的JSON形式打印;
    否则，直接打印改值。
    """
    # 检查是否是pydantic模型
    if hasattr(data,'model_dump_json'):
        # 将pydantic模型转换为json
        data = json.loads(data.model_dump_json())
    # 检查是否是列表
    if(isinstance(data,(list))):
        for item in data:
            print_json(item)
    # 检查是否是字典
    elif(isinstance(data,(dict))):
        print(json.dumps(
            data,
            indent=4,
            ensure_ascii=False
        ))
    else:
        print(data)
        
def get_completion(message,model="gpt-3.5-turbo"):
 
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature = 0,
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "sum",
                    "description": "计算一组数的加和",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "messages": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                }
                            }
                        }
                    }
                }
            }
        ],
        #启动流式输出                
        stream=True,
    )
    return response

prompt = "1+2+3"
#prompt = "你是谁"

messages = [
    {"role": "system", "content": "你是一个小学数学老师，你要教学生加法"},
    {"role": "user", "content": prompt},
]

response = get_completion(messages)

function_name,args,text = "","",""

print("=======Streaming======")

#需要把stream里的token拼起来，才能得到完整的JSON结构
for msg in response:
    
    delta = msg.choices[0].delta
    if delta.tool_calls:
        if not function_name:
            function_name = delta.tool_calls[0].function.name
            print(function_name)
        args_delta = delta.tool_calls[0].function.arguments
        print(args_delta)  #打印每次得到的数据
        args = args + args_delta
    elif delta.content:
        #打印每次得到的数据
        text_delta = delta.content
        print(text_delta)
        #拼接
        text = text + delta.content

print("======done!=====")

if function_name or args:
    print(function_name)
    print_json(args)
if text:
    print(text)

        
