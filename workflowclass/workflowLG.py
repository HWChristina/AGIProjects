#使用langgraph框架实现一个工作流
import json
import os
from langgraph.graph import StateGraph,START,END
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import Agently

# 加载环境变量
_ = load_dotenv(find_dotenv())
client = OpenAI()

"""---------------------使用LangGraph框架--------------------------"""
#构造模型请求的方法
def get_completion(
    prompt = str,
    system_message:str = "You are a helpful assistant.",
    model:str ="gpt-3.5-turbo",
    temperature:float = 0.3,
    json_mode:bool = False,
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content

#定义工作流内的传递的数据信息结构
from typing import TypedDict,Optional
class State(TypedDict):
    source_lang :str
    target_lang :str
    source_text :str
    country: Optional[str]=None
    translation_1: Optional[str]=None
    reflection:Optional[str]=None
    translation_2: Optional[str]=None
    
#创建了一个工作流对象
workflow = StateGraph(State)

#定义工作过程的执行函数
def initial_translation(state):
    source_lang = state.get("source_lang")
    target_lang = state.get("target_lang")
    source_text = state.get("source_text")
    
    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."

    prompt = f"""This is an{source_lang} to {target_lang} translation,please provide the {target_lang} translation for this text.\
Do not provide any explanations or text apart from the translation.
{source_lang}:{source_text}

{target_lang}:"""

    translation = get_completion(prompt, system_message=system_message)
    
    #print("[初次翻译结果]:\n",translation,"\n\n\n")
    return {"translation_1":translation}

def reflect_on_translation(state):
    source_lang = state.get("source_lang")
    target_lang = state.get("target_lang")
    source_text = state.get("source_text")
    country = state.get("country") or ""
    translation_1 = state.get("translation_1") 
    
    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

    addition_rule = (
        f"The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}."
        if country !=""
        else ""
    )
    
    prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
{addition_rule}

The source text and initial translation,delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else.""" 

    reflection = get_completion(prompt,system_message = system_message)
    
    #print("[翻译改进建议]:\n",reflection,"\n\n\n")
    
    return {"reflection":reflection}


"""
state = {
    "source_lang":xxx,
    "target_lang":xxx,
    "source_text":xxx,
    "country":xxx | None,
    "translation_1": xxx,
    "reflecion":xxx,
}
"""
def improve_translation(state):
    source_lang = state.get("source_lang")
    target_lang = state.get("target_lang")
    source_text = state.get("source_text")
    translation_1 = state.get("translation_1")
    reflection = state.get("reflection")
    
    system_message = f"You are an expert linguist,specializing in translation editing from {source_lang} to {target_lang}."

    prompt = f"""Your task is to carefully read,and edit, a translation from {source_lang} to {target_lang}, taking into 
account a list of expert suggestions and constructions and constructive criticisms.

The source text, the inital translation,and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translated and nothing else."""
    
    translation_2 = get_completion(prompt,system_message)
    #print ("[最终翻译结果]:\n",translation_2)
    
    return {"translation_2":translation_2 } 

#规划执行工作流
##节点(node)注册
workflow.add_node("initial_translation",initial_translation)
workflow.add_node("reflect_on_translation",reflect_on_translation)
workflow.add_node("improve_translation",improve_translation)

##连接节点
workflow.add_edge(START,"initial_translation")
workflow.add_edge("initial_translation","reflect_on_translation")
workflow.add_edge("reflect_on_translation","improve_translation")
workflow.add_edge("improve_translation",END)

#开始执行
app = workflow.compile()
result = app.invoke({
    "source_lang":"English",
    "target_lang":"中文",
    "source_text":
"""Ideas for extensions
Here are ideas we haven’t had time to experiment with but that we hope the open-source community will:

Try other LLMs. We prototyped this primarily using gpt-4-turbo. We would love for others to experiment with other LLMs as well as other hyperparameter choices and see if some do better than others for particular language pairs.
Glossary Creation. What’s the best way to efficiently build a glossary -- perhaps using an LLM -- of the most important terms that we want translated consistently? For example, many businesses use specialized terms that are not widely used on the internet and that LLMs thus don’t know about, and there are also many terms that can be translated in multiple ways. For example, ”open source” in Spanish can be “Código abierto” or “Fuente abierta”; both are fine, but it’d better to pick one and stick with it for a single document."""
})
#print(result)
#绘制流程图
from mermaid import Mermaid
mermaid_code = app.get_graph().draw_mermaid()
with open("workflow_diagram.md","w") as file:
    file.write(mermaid_code)
print("Mermaid diagram saved to workflow_diagram.md")
#Mermaid(app.get_graph().draw_mermaid())


