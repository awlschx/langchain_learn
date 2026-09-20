# lc-cource

LangChain 学习项目。核心文件 `Assistant.py` 实现了一个最小的 **RAG(检索增强生成)问答助手**:
把本地知识库文档切分、向量化后存入 Milvus,用户提问时检索最相关的片段,拼接成上下文交给大模型生成回答。

## 工作流程

```
knowledge.txt
   │  读取 (TextLoader)
   ▼
文本切分 (chunk_size=200, chunk_overlap=20)
   │
   ▼
向量化 (BGE-M3, 1024 维)
   │
   ▼
写入 Milvus  ──► 集合 rag_tutorial.docs
                    (字段: id / vector / text / source / chunk_id)
   │
   ▼
用户提问 → 问题向量化 → 向量检索 Top-3 → 拼接上下文 → Agent 生成回答
```

## 环境要求

| 依赖 | 说明 |
| --- | --- |
| Python | >= 3.13 |
| Milvus | 需本地运行,默认连接 `http://localhost:19530` |
| SiliconFlow API Key | 用于调用 `Pro/BAAI/bge-m3` 嵌入模型 |
| DeepSeek API Key | 用于对话模型 |

Milvus 最简启动方式(Docker):

```bash
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
```

## 快速开始

1. **安装依赖**(项目使用 uv 管理):

```bash
uv sync
```

2. **配置环境变量**——在项目根目录创建 `.env`:

```
DEEPSEEK_API_KEY=你的密钥
```

3. **准备知识库文件**——`Assistant.py` 中读取的路径是 `../knowledge.txt`,
   即项目上一级目录:`D:\python_code\knowledge.txt`。文件内容为 UTF-8 编码的纯文本。

4. **运行**:

```bash
uv run Assistant.py
```

运行时会重建集合(已存在的 collection 会被删除后重新创建),打印集合统计信息,然后进入检索问答。

## 配置项

`Assistant.py` 顶部的常量按需修改:

| 常量 | 默认值 | 说明 |
| --- | --- | --- |
| `MILVUS_URI` | `http://localhost:19530` | Milvus 服务地址 |
| `DB_NAME` | `rag_tutorial` | 数据库名,不存在时自动创建 |
| `COLLECTION_NAME` | `docs` | 向量集合名 |
| `KNOWLEWDGE_FILE` | `../knowledge.txt` | 知识库文件路径 |
| `EMBED_MODEL_NAME` | `Pro/BAAI/bge-m3` | 嵌入模型 |
| `EMBED_DIM` | `1024` | 向量维度,需与模型一致 |

## 其他文件

- `main.py`、`chapter1.py` —— 课程章节练习代码。
- `pyproject.toml` / `uv.lock` —— 依赖声明与锁定文件。
