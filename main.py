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

# ------------------ 动态加载知识库 ------------------
def load_knowledge_by_topic(topic: str) -> str:
    file_path = "knowledge.txt"
    if not os.path.exists(file_path):
        return ""

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("题目:")
    blocks = ["题目:" + b.strip() for b in blocks if b.strip()]
    if not blocks:
        return ""

    matched_blocks = []
    if topic:
        keywords = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', topic)
        keywords = [kw for kw in keywords if len(kw) > 1 and kw not in ["什么", "如何", "为什么", "请问"]]
        for block in blocks:
            for kw in keywords:
                if kw in block:
                    matched_blocks.append(block)
                    break

    if matched_blocks:
        seen = set()
        unique = []
        for b in matched_blocks:
            if b not in seen:
                seen.add(b)
                unique.append(b)
            if len(unique) >= 2:
                break
        return "\n\n".join(unique)

    # 匹配失败，返回空字符串（不再返回保底内容）
    return ""

# ------------------ 题目选择器 ------------------
DEFAULT_TOPICS = [
    "TCP三次握手",
    "HTTP状态码",
    "进程与线程的区别",
    "数据库索引原理",
    "什么是RESTful API"
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

# ------------------ 请求模型 ------------------
class ChatRequest(BaseModel):
    messages: list
    topic: str = None

# ------------------ 核心 API ------------------
@app.post("/api/chat")
async def chat(request: ChatRequest):
    topic = pick_topic(request.topic)
    dynamic_knowledge = load_knowledge_by_topic(topic)
    knowledge_text = dynamic_knowledge if dynamic_knowledge else "（暂无相关参考资料，请面试官根据自身专业知识提问）"

    system_prompt = f"""你是一位资深技术面试官，正在对候选人进行八股文面试。

【重要】当前面试题目由系统指定，为：{topic}
你必须严格围绕这个题目进行提问，不得在回复中更换或提出其他题目。如果候选人的回答偏离主题，请引导回到本题。

知识库（仅供参考）：
{knowledge_text}

规则：
1. 根据候选人的回答进行追问，每次只问一个问题。
2. 如果候选人回答正确，可继续深入追问；回答错误则指出并提示。
3. 当候选人明确表示“结束面试”或“总结”时，评估本轮对话。如果回答少于3个问题，回复“回答太少无法评估，请继续作答”；否则输出 JSON 总结：{{"offer_probability": 0~100, "comment": "调侃评语"}}。
4. 除结束时的 JSON 外，其他回复必须为自然对话语句，不包含代码或格式。
5如果候选人说“我想切换到[某个主题]”或“我们聊聊[某个主题]”，表示他希望更换面试题目。你可以同意切换，并在回复中明确指出“新的面试题目为：[主题]”，并在末尾单独一行输出 NEW_TOPIC: [主题]，以便系统识别。新题目名称必须是知识库中存在的题目或合理的八股文主题。
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
            # ---------- 新增：解析 NEW_TOPIC ----------
            new_topic = None
            if "NEW_TOPIC:" in reply:
                lines = reply.split('\n')
                for line in lines:
                    if line.strip().startswith("NEW_TOPIC:"):
                        new_topic = line.replace("NEW_TOPIC:", "").strip()
                        break
                # 从回复中移除该行（避免显示给用户）
                if new_topic:
                    reply = '\n'.join([line for line in lines if not line.strip().startswith("NEW_TOPIC:")])

            final_topic = new_topic if new_topic else topic
            return {"reply": reply.strip(), "topic": final_topic}
        else:
            return {"error": f"通义千问 API 错误: {response.code} - {response.message}"}
    except Exception as e:
        return {"error": str(e)}

# ------------------ 挂载静态文件 ------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")