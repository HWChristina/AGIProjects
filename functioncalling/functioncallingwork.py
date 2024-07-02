#使用function calling的方式实现手机流量包智能客服例子

from openai import OpenAI
from dotenv import load_dotenv,find_dotenv
import sqlite3
import json
import requests

#获取环境变量
_ = load_dotenv(find_dotenv())
client = OpenAI()

def print_json(data):
    """
    打印参数。如果参数是有结构的(如字典、列表),则以格式化的JSON形式打印;
    否则，直接打印该值。
    """
    if hasattr(data,'model_dump_json'):
        data = json.loads(data.model_dump_json())
        
    if(isinstance(data,(list))):
        for item in data:
            print_json(item)
    elif(isinstance(data,(dict))):
        print(json.dumps(data,indent=4,ensure_ascii=False))
    else:
        print(data)
        

#描述数据库表的结构

database_schema_string = """
CREATE TABLE traffic_packages
(
    id INT PRIMARY KEY NOT NULL, -- 主键,不允许为空
    plan_name VARCHAR(255) NOT NULL, -- 流量套餐名称,不允许为空
    price DECIMAL(10,2) NOT NULL, --月费,以元为单位
    data_limit INT NOT NULL, --流量,以GB为单位
    student_only INTEGER -- 是否仅限学生使用,0表示否,1表示是
);
"""
#创建数据库连接
conn=sqlite3.connect(":memory:")
cursor = conn.cursor()

#创建表
cursor.execute(database_schema_string)

#插入5条明确的模拟记录
mock_data = [
    # 套餐名称，价格，流量限制，是否仅限学生使用
    (1,"经济套餐",50,10,0),
    (2,"畅游套餐",180,100,0),
    (3,"无限套餐",300,1000,0),
    (4,"校园套餐",150,200,1),
]

#插入数据
for record in mock_data:
    cursor.execute("INSERT INTO traffic_packages VALUES (?,?,?,?,?)",record)
    
#提交事务
conn.commit()

def ask_database(query):
    """
    根据数据库查询得到流量套餐信息
    """
    cursor.execute(query)
    records = cursor.fetchall()
    return records

#gpt-3.5-turbo
def get_sql_completion(messages,model = "gpt-4o"):
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        tools=[{
            "type": "function",
            "function": {
                "name": "ask_database",
                "description": "根据查阅数据库信息得到流量套餐信息",
                "parameters": {
                    "type":"object",
                    "properties":{
                        "query":{
                            "type":"string",
                            "description":f"""
                            SQL query extracting info to answer the user's question.
                            SQL should be written using this database schem:
                            {database_schema_string}
                            The query should be returned in plain text,not in JSON.
                            The query should only contain grammars supported by SQLite.
                            """,
                        }
                    },
                    "required":["query"]
                }
            }
        }],
    )
    return response.choices[0].message
           
#prompt = "流量最大的套餐是什么"
#prompt= "帮我推荐一个土豪套餐"
#prompt= "有没有便宜一点的套餐推荐"
#get_sql_completion("流量最大的套餐是什么")

#初始系统信息
messages=[
    {"role": "system", "content": "你是一个智能的流量套餐推荐系统，你只能回答关于流量套餐的问题，不能回答其他问题"}
]      

def process_user_input(user_input):
    #将用户输入添加到消息列表中
    messages.append({"role": "user", "content": user_input})
    #获取GPT回复
    response = get_sql_completion(messages)
    

    #查询数据库
    if response.content is None:
        response.content = ""
    
    messages.append(response)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call.function.name == "ask_database":
            arguments = tool_call.function.arguments
            args = json.loads(arguments)
            result = ask_database(args["query"])
            #print("====DB Record====")
            #print(result)
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "function",
                "name": "ask_database",
                "content": str(result)
            })
            response = get_sql_completion(messages)
           
           
    #print_json(messages)
    return response

""" 
#示例用户输入
user_input = "流量最大的套餐是什么"
response= process_user_input(user_input)

#继续处理更多用户的输入
more_user_input = "给我办一个"
response= process_user_input(more_user_input)  """
    
#处理多条用户输入
user_inputs = [
    "流量最大的套餐是什么",
    "太贵了，有没有便宜一点的套餐推荐",
    "帮我办一个"
]

for user_input in user_inputs:
    response = process_user_input(user_input)
    if response:
        print("Response:", response.content)
    print("\n")