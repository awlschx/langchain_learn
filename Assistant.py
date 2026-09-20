import TextLoader
from jedi.plugins.stdlib import collections_namedtuple
from pymilvus import MilvusClient
from langchain.embeddings import init_embeddings
import os
from dotenv import load_dotenv
#1,基本配置

MILVUS_URI = "http://localhost:19530"#milvus服务的连接地址
DB_NAME = "rag_tutorial"#自定义数据库名称
COLLECTION_NAME = "docs"#向量
KNOWLEWDGE_FILE = "../knowledge.txt"#文件路径


#BGE-M3在siliconFlow/Milvus文档中都是1024维
EMBED_MODEL_NAME = "Pro/BAAI/bge-m3"#嵌入模型
EMBED_DIM = 1024#向量维度

#初始化客户端
client = MilvusClient(MILVUS_URI)

#查库，没有就创建
existed_databases = client.list_databases()
if DB_NAME not in existed_databases:
    client.create_database(db_name=DB_NAME)

#切换数据库
client.use_database(DB_NAME)

#如果已经存在collection，删除在建
if client.has_collection(collection_name=COLLECTION_NAME):
    client.drop_collection(collection_name=COLLECTION_NAME)

#创建
client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=EMBED_DIM,
    metric_type="COSINE"
)

#初始化embedding
embed_model = init_embeddings(
    model="openai:"+EMBED_MODEL_NAME,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("")
)

#读取文档并切分

#加载
loader = TextLoader(file_path=KNOWLEWDGE_FILE,encoding="utf-8")
documents = loader.load()

#切分
RecursiveCharacterTextSplitter(
    #切割长度
    chunk_size=200,
    #重叠部分
    chunk_overlap=20,
    seperator= [
        ""
    ]
)
chunks = splitter.split_document(documents)


#生成向量并写入milvus
text = [
    chunk.page_content for chunk in chunks
]

#向量化
vectors = embed_model.embed_documents(text)

data = [
    {
        "id" : i,
        "vector" : vectors[i],
        "text" : chunks[i].page_content,
        "source" : KNOWLEWDGE_FILE,
        "chunk_id" : i
    }

    for i in range (len(chunks))
]

insert_res = client.upsert(
    collection_name=COLLECTION_NAME,
    data=data,
)


#flush磁盘
client.flush(collections_name=COLLECTION_NAME)

#获取集合的统计信息
stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
print(stats)

#查询当前的collection有多少条数据

client.query(
    collection_name=COLLECTION_NAME,
    filter="id>=0",
    output_fields=["id","chunk_id"]
)

#创建agent
#加载环境变量
load_dotenv(override=True)

#初始化model
model = init_chat_model(
    model="deepseek",
    model_provider="deepseek",
    api_key=os.getenv(DEEPSEEK_API_KEY),
    base_url=os.getenv("")
)


agent = create_agent(
    model=model,
    tools=[],
    system_prompt=""
)

#定义函数实现检索
def retrieve(query : str,limit :int =3):
    #向量化
    query_vecter = embed_model.embed_query(query)

#检索
query = ""

#将问题向量化
query_vector = embed_model.embed_query(str(query))

#从库里检索
client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    limit=3,
    output_fields=["text","chunk_id","source"]

)


#生成回答
def generate_answer(query : str):
    #检索到的数据
    hits = retrieve(str,limit=5)

    #格式化
    context_blocks = []
    fot i,hit in enumerate(hits,1):
    text = hit["entity"]["text"]
    source = hit["entity"].get("source","unknown")
    chunk_id = hit["entity"].get("chunk_id","unknown")
    score = hit["distance"]#越高越相似

    print(f"[{i}] chunk_id={chunk_id} score={score:.4f} source={source}")
    print(text)
    print()


    #拼接
    context_blocks.append(
        f"[片段{i} | chunk_id={chunk_id} | source={source}]\n{text}"
    )

    context = "\n\n".join(context_blocks)


    #构造prompt
    user_prompt = f"""
问题：
{query}

上下文：
{context}
"""


    #调用agent
    result = agent.invoke({
        "messages" : [{"role":"user","content":user_prompt}],
    })

    final_msg=result["messages"][-1]

    print("========")
    final_msg.pretty_print()

