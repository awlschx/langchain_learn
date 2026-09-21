import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient

# Windows 控制台默认 GBK,模型输出 emoji 时 pretty_print 会抛 UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8")
# 报错信息同样可能含中文,一并改成 UTF-8,否则控制台会显示乱码
sys.stderr.reconfigure(encoding="utf-8")

# 环境变量必须在读取 os.getenv 之前加载
load_dotenv(override=True)

# ============ 1.基本配置 ============
MILVUS_URI = "http://localhost:19530"  # milvus服务的连接地址
DB_NAME = "rag_tutorial"  # 自定义数据库名称
COLLECTION_NAME = "docs"  # 向量集合名
KNOWLEDGE_FILE = "../knowledge.txt"  # 知识库文件路径(相对于运行目录)

# BGE-M3在siliconFlow/Milvus文档中都是1024维
EMBED_MODEL_NAME = "Pro/BAAI/bge-m3"  # 嵌入模型
EMBED_DIM = 1024  # 向量维度
# 硅基流动的 OpenAI 兼容接口地址
EMBED_BASE_URL = "https://api.siliconflow.cn/v1"
# 嵌入走硅基流动,必须用硅基流动自己的 key,DeepSeek 的 key 在这里无效
EMBED_API_KEY = os.getenv("SILICONFLOW_API_KEY")
CHAT_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not EMBED_API_KEY:
    raise SystemExit("缺少 SILICONFLOW_API_KEY,请在项目根目录的 .env 中填入硅基流动的密钥")
if not CHAT_API_KEY:
    raise SystemExit("缺少 DEEPSEEK_API_KEY,请在项目根目录的 .env 中填入 DeepSeek 的密钥")

CHAT_MODEL_NAME = "deepseek-chat"  # 对话模型
TOP_K = 3  # 每次检索取回的片段数

# ============ 2.初始化 embedding 与对话模型 ============
embed_model = init_embeddings(
    # "openai:" 前缀表示用 OpenAI 兼容协议调用,实际请求发往 EMBED_BASE_URL
    model="openai:" + EMBED_MODEL_NAME,
    api_key=EMBED_API_KEY,
    base_url=EMBED_BASE_URL,
    # 硅基流动等第三方接口只接受原始文本,不接受 token id 数组
    check_embedding_ctx_length=False,
)

model = init_chat_model(
    model=CHAT_MODEL_NAME,
    model_provider="deepseek",
    api_key=CHAT_API_KEY,
)

agent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "你是一个知识库问答助手。请只依据用户提供的【上下文】回答问题,"
        "并在回答末尾用 [片段N] 标注你引用的来源片段。"
        "如果上下文中没有相关信息,直接回答“知识库中没有找到相关内容”,不要编造。"
    ),
)


# ============ 3.读取文档、切分、向量化并写入 Milvus ============
def build_knowledge_base() -> None:
    client = MilvusClient(MILVUS_URI)

    # 查库,没有就创建
    if DB_NAME not in client.list_databases():
        client.create_database(db_name=DB_NAME)
    client.use_database(DB_NAME)

    # 如果已经存在 collection,删除再建(避免重复入库)
    if client.has_collection(collection_name=COLLECTION_NAME):
        client.drop_collection(collection_name=COLLECTION_NAME)

    # 快速建集合:text/source/chunk_id 会被存进动态字段
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=EMBED_DIM,
        metric_type="COSINE",
    )

    # 加载文档
    loader = TextLoader(file_path=KNOWLEDGE_FILE, encoding="utf-8")
    documents = loader.load()

    # 切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # 切割长度
        chunk_overlap=20,  # 相邻片段的重叠部分
        separators=["\n\n", "\n", "。", "!", "?", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"知识库共切分出 {len(chunks)} 个片段")

    # 向量化
    vectors = embed_model.embed_documents([chunk.page_content for chunk in chunks])

    data = [
        {
            "id": i,
            "vector": vectors[i],
            "text": chunks[i].page_content,
            "source": KNOWLEDGE_FILE,
            "chunk_id": i,
        }
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=COLLECTION_NAME, data=data)

    # 落盘,否则刚写入的数据查不到
    client.flush(collection_name=COLLECTION_NAME)

    # 集合的统计信息
    stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
    print(f"集合统计:{stats}")


# ============ 4.检索 ============
def retrieve(query: str, limit: int = TOP_K) -> list:
    client = MilvusClient(MILVUS_URI)
    client.use_database(DB_NAME)

    # 将问题向量化
    query_vector = embed_model.embed_query(query)

    # 从库里检索,COSINE 距离下 distance 越大越相似
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=limit,
        output_fields=["text", "chunk_id", "source"],
    )
    return results[0]


# ============ 5.生成回答 ============
def generate_answer(query: str) -> str:
    hits = retrieve(query)

    if not hits:
        print("没有检索到任何内容,请先确认知识库已入库")
        return ""

    # 格式化检索结果
    context_blocks = []
    for i, hit in enumerate(hits, 1):
        text = hit["entity"]["text"]
        source = hit["entity"].get("source", "unknown")
        chunk_id = hit["entity"].get("chunk_id", "unknown")
        score = hit["distance"]  # 越高越相似

        print(f"[{i}] chunk_id={chunk_id} score={score:.4f} source={source}")
        print(text)
        print()

        # 拼接
        context_blocks.append(
            f"[片段{i} | chunk_id={chunk_id} | source={source}]\n{text}"
        )

    context = "\n\n".join(context_blocks)

    # 构造 prompt
    user_prompt = f"""问题：
{query}

上下文：
{context}
"""

    # 调用 agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_prompt}]},
    )
    final_msg = result["messages"][-1]

    print("=" * 30, "回答", "=" * 30)
    final_msg.pretty_print()
    return final_msg.content


def main() -> None:
    build_knowledge_base()

    while True:
        query = input("\n请输入问题(直接回车退出):").strip()
        if not query:
            break
        generate_answer(query)


if __name__ == "__main__":
    main()
