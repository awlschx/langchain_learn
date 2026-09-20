from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()

# model = init_chat_model("deepseek-chat")
# print(type(model))
# for chunk in model.stream("你是谁"):
#     print(chunk.content, end="", flush=True)
# print()

# 创建智能体
agent = create_agent(model="deepseek-chat")
response = agent.invoke(
    {
        "messages":[
            SystemMessage("你是原神的妮露"),
            HumanMessage("你是谁"),
            HumanMessage("你来谈谈什么是爱")
        ]
    }
)
print(response)
for message in response["messages"]:
    message.pretty_print()
# print(model.invoke(
#     [
#     #     模型身份
#         {"role":"system","content":"你是原神里的妮露，而且你要在每次回答的后面加上 ciallo~~~"},
#         {"role":"user","content":"你是谁？"}
#
#     ]
# ))