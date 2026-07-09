# 🎯 面试狙击手 — AI 驱动的智能面试官

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.14-orange.svg)](https://python.langchain.com/)
[![Railway](https://img.shields.io/badge/Railway-Deployed-purple.svg)](https://railway.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

基于 **RAG（检索增强生成）** 技术构建的智能面试练习系统。支持与 AI 面试官进行八股文对话式面试，系统会根据你的回答自动追问、智能换题，并在面试结束后生成详细的分析报告。

**🔗 在线体验：** [https://jujiproject.up.railway.app](https://jujiproject.up.railway.app)

---

## 📸 功能截图

| 聊天界面 | Offer 概率卡片 | 面试报告 |
| :---: | :---: | :---: |
| ![聊天界面](https://via.placeholder.com/200x400?text=聊天界面) | ![Offer卡片](https://via.placeholder.com/200x400?text=Offer卡片) | ![面试报告](https://via.placeholder.com/200x400?text=面试报告) |

> ⚠️ 截图占位，建议替换为你自己的实际截图。

---

## ✨ 核心功能

| 功能 | 说明 |
| :--- | :--- |
| **智能换题** | 用户输入任意关键词，系统自动匹配最相关的面试题目 |
| **混合检索** | 同时使用向量检索（语义）和 BM25（关键词），召回率更高 |
| **查询改写** | 自动将用户问题扩展为多个变体，提升检索精度 |
| **面试报告** | 结束面试后自动生成详细报告：回答质量评价、知识点覆盖、薄弱环节分析、学习建议 |
| **Offer 概率卡片** | 面试结束后生成调侃式大厂 Offer 概率卡片，支持截图分享 |
| **移动端适配** | 在手机上可流畅使用，键盘弹起时聊天区域自动滚动 |

---

## 🛠️ 技术栈

### 后端
- **[FastAPI](https://fastapi.tiangolo.com/)** — 高性能异步 Web 框架
- **[LangChain](https://python.langchain.com/)** — 大模型应用开发框架
- **[ChromaDB](https://www.trychroma.com/)** — 向量数据库
- **[DashScope](https://dashscope.aliyun.com/)** — 通义千问 API（Embedding + 生成 + Rerank）
- **[BM25](https://github.com/dorianbrown/rank_bm25)** — 关键词检索算法

### 前端
- 原生 HTML / CSS / JavaScript
- 无框架依赖，轻量快速
- 响应式设计，适配移动端

### 部署
- **[Railway](https://railway.app/)** — 云端部署 + 自动 CI/CD
- **GitHub** — 代码托管
- **Volume** — 向量数据库持久化存储

---

## 📂 项目结构
interview-sniper/
├── main.py # FastAPI 后端主文件
├── requirements.txt # Python 依赖
├── knowledge.txt # 知识库（可自定义扩充）
├── static/
│ └── index.html # 前端聊天界面
├── chroma_db/ # 向量数据库（本地/Volume）
├── .env # 环境变量（本地开发）
├── .gitignore # Git 忽略文件
└── README.md # 项目说明

text

---

## 🚀 本地运行

### 1. 克隆仓库

```bash
git clone https://github.com/liyulong2345/jujiproject.git
cd jujiproject
2. 创建虚拟环境（推荐 Python 3.11+）
bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
# 或
venv\Scripts\activate         # Windows
3. 安装依赖
bash
pip install -r requirements.txt
4. 配置环境变量
在项目根目录创建 .env 文件：

env
DASHSCOPE_API_KEY=sk-你的通义千问API密钥
你可以从 阿里云百炼控制台 获取 API Key。

5. 启动服务
bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
6. 访问应用
打开浏览器访问 http://127.0.0.1:8000

🧪 如何使用
发送第一条消息 → AI 自动从知识库中随机选一道题目

自由提问 → 系统会根据你的问题自动切换匹配的题目（如问“内存泄漏”会切到 JVM 相关题）

AI 追问 → 面试官会不断深入追问，考察你的掌握程度

结束面试 → 回答至少 3 个问题后点击“结束面试”，系统生成 Offer 概率卡片 + 详细报告

🌐 部署到 Railway
Fork / Push 代码到 GitHub

在 Railway 中连接 GitHub 仓库

设置环境变量：DASHSCOPE_API_KEY

挂载 Volume（路径 /app/chroma_db，大小 1 GB）

设置启动命令：uvicorn main:app --host 0.0.0.0 --port $PORT

自动部署完成，生成公网域名

📊 知识库自定义
编辑 knowledge.txt，按以下格式添加题目：

txt
题目: TCP三次握手
关键点: SYN, ACK, 序号, 状态转换
描述: 客户端发送SYN包，服务器回复SYN+ACK，客户端再发ACK。
系统会在启动时自动切分、向量化并建立索引。

🤝 贡献
欢迎提交 Issue 和 Pull Request！

Fork 本仓库

创建你的功能分支 (git checkout -b feature/AmazingFeature)

提交你的改动 (git commit -m 'Add some AmazingFeature')

推送到分支 (git push origin feature/AmazingFeature)

打开一个 Pull Request

📄 许可证
本项目采用 MIT License 开源协议。

🙏 致谢
阿里云 DashScope 提供大模型服务

LangChain 提供 RAG 框架支持

Railway 提供便捷的云部署服务

📬 联系我
GitHub: @liyulong2345

项目链接: https://github.com/liyulong2345/jujiproject

如果这个项目对你有帮助，欢迎 Star ⭐ 支持一下！
