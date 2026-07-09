from dotenv import load_dotenv
load_dotenv()

import os
import re
import random
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dashscope import Generation


# ---------- RAG 相关导入 ----------
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

#uvicorn main:app --reload --host 127.0.0.1 --port 8000

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise RuntimeError("请设置环境变量 DASHSCOPE_API_KEY")

# ---------- 1. 全局 RAG 向量库初始化 ----------
vectorstore = None
retriever = None  # 混合检索器

def init_rag():
    global vectorstore, retriever
    file_path = "knowledge.txt"
    if not os.path.exists(file_path):
        print("警告：未找到 knowledge.txt")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 按 "题目：" 分割成独立块
    raw_blocks = text.split("题目: ")
    raw_blocks = ["题目: " + b.strip() for b in raw_blocks if b.strip()]

    docs = []
    for block in raw_blocks:
        # 如果单块超过 1000 字符，才进一步切分
        if len(block) <= 1200:
            docs.append(block)
        else:
            # 用换行和句号切分，保证块内不跨题目
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", "。", "，", " ", ""],
            )
            docs.extend(splitter.split_text(block))

    # 去重（避免重复块）
    seen = set()
    unique_docs = []
    for doc in docs:
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY
    )

    vectorstore = Chroma.from_texts(
        texts=unique_docs,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="interview_knowledge"
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 2. BM25 检索器（基于相同文档）
    bm25_retriever = BM25Retriever.from_texts(unique_docs)
    bm25_retriever.k = 5

    # 3. 混合检索器（向量权重 0.5，BM25 权重 0.5）
    retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    print(f"✅ RAG 初始化成功，共切分为 {len(unique_docs)} 个知识块","混合检索已启动")
# 启动时立即初始化 RAG
init_rag()

# ---------- 2. 检索函数(查询改+bm25关键词检索)----------
# 查询改写
def expand_query(query: str) -> list:
    """用 LLM 生成 2 个查询变体（不消耗 Embedding，只消耗普通 LLM token）"""
    prompt = f"""请将以下技术面试问题改写成 2 个不同角度、但语义相同的问题，用中文分号；分隔。
原问题：{query}
改写："""
    try:
        response = Generation.call(
            model='qwen-turbo',  # 用小模型省钱
            messages=[{"role": "user", "content": prompt}],
            api_key=DASHSCOPE_API_KEY,
            temperature=0.3,
            result_format='message'
        )
        if response.status_code == 200:
            text = response.output.choices[0].message.content
            # 分割成列表，去掉空字符串
            variants = [q.strip() for q in text.split("；") if q.strip()]
            # 最多取 2 个变体，加上原问题
            print(variants)
            return [query] + variants[:2]
    except Exception as e:
        print(f"查询改写失败: {e}")
    return [query]  # 失败时回退到原问题


# ---------- 叠加后的检索函数 ----------
def retrieve_knowledge(query: str) -> str:
    """方案一（查询改写） + 方案三（混合检索）叠加"""
    global retriever
    if retriever is None:
        return ""
    try:
        # 1. 生成多个查询变体（方案一）
        queries = expand_query(query)
        print(f"🔍 混合检索查询: {queries}")  # 调试时可看到日志

        # 2. 对每个查询执行混合检索（方案三）
        all_docs = []
        seen_contents = set()
        for q in queries:
            docs = retriever.invoke(q)  # 这里用的是混合检索器
            for doc in docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    all_docs.append(doc)

        # 3. 如果结果太多，取前 3 个（已经去重）
        if all_docs:
            return "\n\n".join([doc.page_content for doc in all_docs[:3]])
        return ""
    except Exception as e:
        print(f"检索出错: {e}")
        return ""

# ---------- 3. 题目选择器（2） ----------
DEFAULT_TOPICS = [
    "TCP三次握手", "HTTP状态码", "进程与线程的区别",
    "数据库索引原理", "什么是RESTful API"
]

def pick_topic(topic: str = None) -> str:
    if topic:
        return topic
    file_path = "knowledge.txt"
    topics = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("题目:"):
                    name = line.replace("题目:", "").strip()
                    if name:
                        topics.append(name)
    if topics:
        return random.choice(topics)
    return random.choice(DEFAULT_TOPICS)

class ChatRequest(BaseModel):
    messages: list
    topic: str = None

# ---------- 4. 核心 API（改动最小，仅替换知识获取方式） ----------
@app.post("/api/chat")
async def chat(request: ChatRequest):
    # ---------- 智能主题推断 ----------
    topic = request.topic

    if not topic:
        # 换题后无 topic：用用户最后一条消息做 RAG 检索，自动匹配题目
        user_msgs = [m.get("content", "") for m in request.messages if m.get("role") == "user"]
        if user_msgs:
            query = user_msgs[-1]  # 取用户最后一条消息

            # 用 RAG 检索最相关的 1 个知识块
            if vectorstore:
                try:
                    results = vectorstore.similarity_search(query, k=1)
                    if results:
                        content = results[0].page_content
                        # 从知识块中提取 "题目:" 行
                        for line in content.split('\n'):
                            if line.startswith("题目:"):
                                topic = line.replace("题目:", "").strip()
                                break
                except Exception as e:
                    print(f"RAG 检索出错: {e}")

            # 如果 RAG 没匹配到，保底使用用户输入作为主题
            if not topic:
                topic = query
        else:
            # 极少数情况：没有用户消息，回退到随机
            topic = pick_topic(None)

    # 如果 topic 还是空的（兜底），才走随机
    if not topic:
        topic = pick_topic(None)
    # ---------- 核心改动：用 RAG 检索替代关键词匹配 ----------
    # 直接用用户当前的 topic 作为查询词去向量库搜
    retrieved_knowledge = retrieve_knowledge(topic)
    knowledge_text = retrieved_knowledge if retrieved_knowledge else "（暂无相关参考资料，请面试官根据自身专业知识提问）"

    system_prompt = f"""你是一位资深技术面试官，正在对候选人进行八股文面试。

【重要】当前面试题目由系统指定，为：{topic}
你必须严格围绕这个题目进行提问，不得在回复中更换或提出其他题目。如果候选人的回答偏离主题，请引导回到本题。

知识库（）：
{knowledge_text}

规则：
1. 根据候选人的回答进行追问，每次只问一个问题。
2. 如果候选人回答正确，可继续深入追问；回答错误则指出并提示。
3. 当候选人明确表示“结束面试”或“总结”时，评估本轮对话。如果回答少于3个问题，回复“回答太少无法评估，请继续作答”；否则输出 JSON 总结：{{"offer_probability": 0~100, "comment": "调侃评语"}}。
4. 除结束时的 JSON 外，其他回复必须为自然对话语句，不包含代码或格式。
5. 如果候选人提出的问题明显不属于当前面试题目（比如你正在问 Java，他却问 TCP），你可以主动切换到该话题，但必须告知候选人。切换后，在回复末尾单独一行输出 `NEW_TOPIC: 新题目名称`，新题目名称应从候选人的问题中提取。
注意：结束时的 JSON 必须单独一行，且严格符合格式。"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in request.messages:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        user_msg_count = sum(1 for msg in request.messages if msg.get("role") == "user")
        if user_msg_count <= 1 and request.messages[-1].get("content") == "结束面试":
            return {"reply": "🤔 你还没回答任何问题呢！至少先聊聊技术再结束吧～"}

        response = Generation.call(
            model='qwen-turbo',
            messages=messages,
            result_format='message',
            api_key=DASHSCOPE_API_KEY,
            temperature=0.7,
            top_p=0.8,
        )
        if response.status_code == 200:
            reply = response.output.choices[0].message.content

            # 解析 NEW_TOPIC（保留你原来的换题逻辑）
            new_topic = None
            if "NEW_TOPIC:" in reply:
                lines = reply.split('\n')
                for line in lines:
                    if line.strip().startswith("NEW_TOPIC:"):
                        new_topic = line.replace("NEW_TOPIC:", "").strip()
                        break
                if new_topic:
                    reply = '\n'.join([line for line in lines if not line.strip().startswith("NEW_TOPIC:")])

            final_topic = new_topic if new_topic else topic
            return {"reply": reply.strip(), "topic": final_topic}
        else:
            return {"error": f"通义千问 API 错误: {response.code} - {response.message}"}
    except Exception as e:
        return {"error": str(e)}

# ---------- 5. 挂载静态文件 ----------
app.mount("/", StaticFiles(directory="static", html=True), name="static")