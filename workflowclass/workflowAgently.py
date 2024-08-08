#使用langgraph框架实现一个工作流
import json
from langgraph.graph import StateGraph,START,END
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import Agently
import os
openai_url = os.environ["OPENAI_BASE_URL"]
openai_api_key = os.environ["OPENAI_API_KEY"]
default_model = 'gpt-3.5-turbo'
# 加载环境变量
_ = load_dotenv(find_dotenv())
client = OpenAI()
"""---------------------使用Agently框架--------------------------"""
#agent工厂
agent_factory = (
    Agently.AgentFactory()
        .set_settings("current_model","OAIClient")
        .set_settings("model.OAIClient.url",openai_url)
        .set_settings("model.OAIClient.auth",{"api_key":openai_api_key})
        .set_settings("model.OAIClient.options",{"model":default_model})
)
workflow = Agently.Workflow()
#定义关键处理节点
@workflow.chunk()
def initial_translation(input,storage):
    """
    input:负责接收从上游传递过来的数据，可以接收多个端点，如果没有定义上游的端点，默认传到default端点
    storage:工作流全过程的全局数据存储器
    """
    source_lang = storage.get("source_lang")
    target_lang = storage.get("target_lang")
    source_text = storage.get("source_text")
    
    #创建一个翻译agent来执行任务(可以复用到improve_text)
    translate_agent = agent_factory.create_agent()
    translate_agent.set_agent_prompt(
        "role",
        f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
    )
    storage.set("translate_agent",translate_agent)
    
    prompt = f"""This is an{source_lang} to {target_lang} translation,please provide the {target_lang} translation for this text.\
Do not provide any explanations or text apart from the translation.
{source_lang}:{source_text}

{target_lang}:"""

    translation_1 = (
        translate_agent
            .input(prompt)
            .start()
    )
    storage.set("translation_1",translation_1)
    
    return {
        "stage":"initial translation",
        "result": translation_1
    }
    
#反思优化
@workflow.chunk()
def reflect_on_translation(inputs,storage):
    source_lang = storage.get("source_lang")
    target_lang = storage.get("target_lang")
    source_text = storage.get("source_text")
    country = storage.get("country") or ""
    translation_1 = storage.get("translation_1") 
    
    #创建一个反思agent来执行任务
    reflection_agent = agent_factory.create_agent()
    
    #给反思agent设置system信息
    reflection_agent.set_agent_prompt(
        "role",
        f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."
    )
    addition_rule = (
        "The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}."
        if country !=""
        else ""
    )
    #向反思agent发起反思任务
    reflection = (
        reflection_agent
            .input(
f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
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
            )
            .start()
    )
    #保存反思结果
    storage.set("reflection",reflection)
    return {
        "stage":"reflection",
        "result": reflection
    }

##二次翻译
@workflow.chunk()
def improve_translation(inputs,storage):
    source_lang = storage.get("source_lang")
    target_lang = storage.get("target_lang")
    source_text = storage.get("source_text")
    translation_1 = storage.get("translation_1")
    reflection = storage.get("reflection")
    
    #使用保存下来的翻译agent
    translate_agent = AgentFactory.get_agent("translate_agent")
    
    #直接发起二次翻译任务
    translation_2 = (
        translate_agent
            .input(
f"""Your task is to carefully read,and edit, a translation from {source_lang} to {target_lang}, taking into 
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
            )
            .start()
    )

    #保存第二次的翻译结果
    storage.set("translation_2",translation_2)
    
    return {
        "stage":"improve translation",
        "result":translation_2
        } 

#规划执行工作流
##节点(node)注册
(
    workflow.chunks["START"]
        .connect_to("initial_translation")
        .connect_to("reflect_on_translation")
        .connect_to("improve_translation")
        .connect_to("END")
)
#输出优化
@workflow.chunk()
def print_stage_result(inputs,storage):
    print(f"[{inputs['default']['stage']}]")
    print(inputs["default"]["result"])
    print("\n\n\n")

workflow.chunks['initial_translation'].connect_to(print_stage_result).connect_to("reflect_on_translation.wait")
workflow.chunks['reflect_on_translation'].connect_to(print_stage_result).connect_to("improve_translation.wait")
workflow.chunks['improve_translation'].connect_to(print_stage_result).connect_to("END.wait")


"""Ideas for extensions
Here are ideas we haven’t had time to experiment with but that we hope the open-source community will:

Try other LLMs. We prototyped this primarily using gpt-4-turbo. We would love for others to experiment with other LLMs as well as other hyperparameter choices and see if some do better than others for particular language pairs.
Glossary Creation. What’s the best way to efficiently build a glossary -- perhaps using an LLM -- of the most important terms that we want translated consistently? For example, many businesses use specialized terms that are not widely used on the internet and that LLMs thus don’t know about, and there are also many terms that can be translated in multiple ways. For example, ”open source” in Spanish can be “Código abierto” or “Fuente abierta”; both are fine, but it’d better to pick one and stick with it for a single document.
"""

#print(result)
#绘制流程图
from mermaid import Mermaid
mermaid_code = workflow.draw()
with open("workflow_diagram.md","w") as file:
    file.write(mermaid_code)
print("Mermaid diagram saved to workflow_diagram.md")
#Mermaid(app.get_graph().draw_mermaid())