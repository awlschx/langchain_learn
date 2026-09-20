import os
import sys
from typing import NotRequired

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore

#Windows 控制台默认 GBK，模型输出 emoji 时 pretty_print 会抛 UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8")

#从env文件中加载环境变量
load_dotenv(override=True)

model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

#长期记忆存储
store = InMemoryStore()


#自定义一个继承AgentState的类
class CustomState(AgentState):
    user_id : NotRequired[str]

@tool(parse_docstring=True)
def save_user_info(name : str , runtime : ToolRuntime) -> str:
    """将客户信息保存在长期记忆中

    Args:
        name: 用户名
    """
    namespace = ("users",)
    key = runtime.state["user_id"]
    value = {"name": name}

    runtime.store.put(namespace,key,value)
    return "saved"

@tool(parse_docstring=True)
def get_user_info(runtime : ToolRuntime) -> str:
    """从长期记忆中读取客户的信息

    Args:
        runtime: 工具的运行时
    """
    namespace = ("users",)
    key = runtime.state["user_id"]

    item = runtime.store.get(namespace,key)
    return str(item.value) if item else "unknown"

agent = create_agent(
    model=model,
    tools=[save_user_info,get_user_info],
    store=store,
    state_schema=CustomState,
    system_prompt="当用户告知姓名时，调用save_user_info保存"
)

print("=" * 30,'-> 第一个会话(线程) <-',"=" * 30)
response1 = agent.invoke({
    "messages":[HumanMessage("你好，很高兴认识你，我叫小明")],
    "user_id":"user-1"

})
for msg in response1["messages"]:
    msg.pretty_print()

print("=" * 30,'-> 第二个会话(线程) <-',"=" * 30)
response2 = agent.invoke({
    "messages":[HumanMessage("我是谁")],
    "user_id":"user-1"
})
for msg in response2["messages"]:
    msg.pretty_print()
