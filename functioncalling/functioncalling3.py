#通过FunctionCalling查询数据库/查询多表

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

def get_sql_completion(messages,model = "gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature = 0,
        tools=[{#
            "type":"function",
            "function":{
                "name": "ask_database",
                "description":"Use this function to answer user questions about business.\
                            Output should be a fully formed SQL query.",
                "parameters":{
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
                    "required":["query"],
                }
            }
                
         }],
    )
    return response.choices[0].message

# 描述数据库表结构

database_schema_string = """
CREATE TABLE customers (
    id INT PRIMARY KEY NOT NULL, -- 主键，不允许为空
    customer_name VARCHAR(255) NOT NULL, -- 客户名，不允许为空
    email VARCHAR(255) UNIQUE, -- 邮箱，唯一
    register_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- 注册时间，默认为当前时间
);
CREATE TABLE products (
    id INT PRIMARY KEY NOT NULL, -- 主键，不允许为空
    product_name VARCHAR(255) NOT NULL, -- 产品名称，不允许为空
    price DECIMAL(10,2) NOT NULL -- 价格，不允许为空
);
CREATE TABLE orders (
    id INT PRIMARY KEY NOT NULL, -- 主键，不允许为空
    customer_id INT NOT NULL, -- 客户ID，不允许为空
    product_id INT NOT NULL, -- 产品ID，不允许为空
    price DECIMAL(10,2) NOT NULL, -- 价格，不允许为空
    status INT NOT NULL, -- 订单状态，整数类型，不允许为空。0代表待支付，1代表已支付，2代表已退款
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 创建时间，默认为当前时间
    pay_time TIMESTAMP -- 支付时间，可以为空
);
"""

#创建数据库表
statements = database_schema_string.strip().split(';')
#去除空语句
statements = [s.strip() for s in statements if s.strip()]


#创建数据库连接
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

#创建orders表
for statement in statements:
    cursor.execute(statement)


#插入5条明确的模拟记录
mock_data = [
   #id,customer_id,product_id,price,status,create_time,pay_time
    (1,1001,'TSHIRT_1',50.00,0,'2023-09-12 10:00:00',None),
    (2,1001,'TSHIRT_2',75.50,1,'2023-09-12 11:00:00','2023-08-16 12:00:00'),
    (3,1002,'SHOES_X1',25.25,2,'2023-10-17 12:30:00','2023-08-17 13:00:00'),
    (4,1003,'SHOES_X2',25.25,1,'2023-10-17 12:30:00','2023-08-17 13:00:00'),
    (5,1003, 'HAT_Z112', 60.75, 1,'2023-10-20 14:00:00', '2023-08-20 15:00:00'),
    (6,1002,'WATCH_X001',90.00,0,'2023-10-28 16:00:00',None)
]
#插入数据
for record in mock_data:
    cursor.execute('''
    INSERT INTO orders (id,customer_id,product_id,price,status,create_time,pay_time)
    VALUES (?,?,?,?,?,?,?)
    ''',record)

#提交事务
conn.commit()

def ask_database(query):
    #执行查询
    cursor.execute(query)
    #获取所有记录
    records=cursor.fetchall()
    return records

#prompt = "10月的销售额"
#prompt = "哪个用户的消费最高？消费多少？"
#prompt = "统计每月每件商品的销售额"
prompt = "这个星期消费最高的用户你谁？他买了哪些商品？每件商品买了几件？花费多少？"

messages=[
    #系统提示
    {"role":"system","content":"你是一个数据分析师，基于数据库的数据回答问题"},
    {"role":"user","content":prompt},
]
response = get_sql_completion(messages)
""" if response.content is None:
    response.content = ""

messages.append(response)

if response.tool_calls is not None:
    tool_call = response.tool_calls[0]
    if tool_call.function.name=="ask_database":
        arguments = tool_call.function.arguments
        args = json.loads(arguments)
        #print("====SQL====")
        #print(args["query"])
        result = ask_database(args["query"])
        #print("====DB Record====")
        #print(result)
        #添加函数调用
        messages.append({
            "tool_call_id":tool_call.id,
            "role":"function",
            "name":"ask_database",
            "content":str(result),
        })
        response = get_sql_completion(messages)
        #print("====Response===")
        #打印回复
        #print(response.content)
        #打印函数调用参数 """
print(response.tool_calls[0].function.arguments)


#查询数据库
""" if response.content is None:
    response.content = ""

messages.append(response)

if response.tool_calls is not None:
    tool_call = response.tool_calls[0]
    if tool_call.function.name=="ask_database":
        arguments = tool_call.function.arguments
        args = json.loads(arguments)
        print("====SQL====")
        print(args["query"])
        result = ask_database(args["query"])
        print("====DB Record====")
        print(result)
        
        messages.append({
            "tool_call_id":tool_call.id,
            "role":"function",
            "name":"ask_database",
            "content":str(result),
        })
        response = get_sql_completion(messages)
        print("====Response===")
        #打印回复
        print(response.content)

print("====对话历史====")
print_json(messages) """