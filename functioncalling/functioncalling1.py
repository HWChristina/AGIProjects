#作业：利用fuction calling的方式实现第二课手机中流量包智能客服的例子。

#调用本地函数

from openai import OpenAI
from dotenv import load_dotenv,find_dotenv
from math import*
import json

_=load_dotenv(find_dotenv())

client = OpenAI()
#json格式打印
def print_json(data):
    """
    打印参数。如果参数是有结构的(如字典或列表),则以格式化的JSON形式打印;
    否则，直接打印改值。
    """
    if hasattr(data,'model_dump_json'):
        data = json.loads(data.model_dump_json())
    
    if(isinstance(data,(list))):
        for item in data:
            print_json(item)
    elif(isinstance(data,(dict))):
        print(json.dumps(
            data,
            indent=4,
            ensure_ascii=False
        ))
    else:
        print(data)
#
def get_completion(message,model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model = model,
        messages= messages,
        temperature=0.7,
        tools=[{   
        #用json描述函数，可以定义多个。由大模型决定调用谁。也可能都不调用
            "type":"function",
            "function":{
                "name":"sum",
                "description":"加法器，计算一组数的和",
                "parameters":{
                    "type":"object",
                    "properties":{
                        "numbers":{
                            "type":"array",
                            "items":{
                                "type":"number"
                            }
                        }
                    }
                }
            }
            
        }],
    )
    return response.choices[0].message

#prompt = "Tell me the sum of 1,2,3,4,5,6,7,8,9,10."
#prompt = "桌上有2个苹果,四个桃子和3本书,一共几个水果？"
#prompt = "1+2+3...+99+100"
prompt = "1024乘以1024是多少?"
#prompt = "太阳从哪边升起?"    #不需要算加法，会怎样？

messages=[
    {"role":"system","content":"你是一个数学家"},
    {"role":"user","content":prompt}
]
response = get_completion(messages,"gpt-4o")

#把大模型的回复加入到对话历史中。必不可少。
messages.append(response)

#如果返回的是函数调用结果，则打印出来
if(response.tool_calls is not None):
    #是否要调用sum
    tool_call = response.tool_calls[0]
    if(tool_call.function.name=="sum"):
        #调用sum    function calling机制过程：函数调用参数
        args = json.loads(tool_call.function.arguments)
        #调用sum函数
        result = sum(args["numbers"])
        
        #把函数调用结果加入到对话历史中
        messages.append(
            {
                "tool_call_id":tool_call.id,
                "role":"tool",
                "name":"sum",
                "content":str(result)
            }
        )
        
        #再次调用大模型
        print("=====最终GPT回复=====")
        print(get_completion(messages).content)
        
        
print("=====对话历史=====")
print_json(messages)