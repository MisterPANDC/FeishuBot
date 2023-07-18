# A Feishu chatbot program based on ChatGLM with vector database

## Getting started
### step1: 启动基于本地知识库的ChatGLM，搭建api

手动构建知识库
```
python kb_setting.py
```

在`configs/model_config.py`中设置基本参数

其余参数设置公网端口，本地知识库id等，可以视为参数传入`chatbot.py`
可以直接运行shell脚本
```
bash bot_run_config.sh
```

也可以传参直接运行`chatbot.py`
```
python chatbot.py
```
### step2: 连接飞书机器人接口


## 基于本地知识库的 LLM 
参考 [langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM)

参考 [langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM) 来实现带有向量数据库的chatGLM api
其中在 `FeishuBot/Feishu_connection_src/chatglm_server/chatglm_server.py`中也实现了基本功能的chatGLM api

---
相比于项目[langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM)
本项目做出的更改如下:
1. 更改ChatMessage的封装，参照`chatglm_server.py`，基于chatGLM以及本地知识库返回一个dict
2. 去除上传本地知识文件等函数，转而使用程序`kb_setting.py`在后端手动处理知识库


---
将映射至公网端口，然后设置飞书接口即可使用

### 具体设置细节
KB_ROOT_PATH
VECTOR_SEARCH_TOP_K
vspath
knowledge_base_id

## 飞书接口
参考项目 [Feishu-ChatGLM](https://github.com/ConnectAI-E/Feishu-ChatGLM)

将api映射至端口后，更改设置
其中设置包括：
1. config.yml


最后运行以下程序,启动与飞书机器人的连接
```
FeishuBot/Feishu_connection_src/main.py
```
