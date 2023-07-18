from fastapi import Body, FastAPI, File, Form, Query, UploadFile, WebSocket, Request

import nltk #nltk什么作用还需要进一步研究
import uvicorn, json, datetime
import torch
# configs

from configs.model_config import (KB_ROOT_PATH, EMBEDDING_DEVICE,
                                  EMBEDDING_MODEL, NLTK_DATA_PATH,
                                  VECTOR_SEARCH_TOP_K, LLM_HISTORY_LEN, OPEN_CROSS_DOMAIN)

#modules to implement knowledge based chat model
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint
from chains.local_doc_qa import LocalDocQA
from kb_setting import create_path

def torch_gc():
    if torch.cuda.is_available():
        for device in CUDA_DEVICES:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


local_doc_id = "lab"#当前只测试lab即可，后面会作为可以传入的参数

_, _, _, vs_path = create_path(local_doc_id)

#这里改用fastapi中的Request
@app.post("/")
async def local_doc_chat(request: Request):
    json_post_raw = await request.json()
    print(type(json_post_raw))
    print(json_post_raw)
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')#为什么request中会有这些东西
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')

    if not os.path.exists(vs_path):
        return "no vs path"
    else:
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=prompt, vs_path=vs_path, chat_history=history, streaming=True
        ):
            pass
        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]

        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        answer = {
            "response": resp["result"],
            "history": history,
            "status": 200,
            "time": time
        }
        log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
        print(log)
        torch_gc()
        return answer

if __name__ == "__main__":
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # 初始化消息
    args = None
    args = parser.parse_args()
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    local_doc_qa = LocalDocQA()
    llm_model_ins = shared.loaderLLM()
    local_doc_qa.init_cfg(
    llm_model=llm_model_ins,
    embedding_model=EMBEDDING_MODEL,
    embedding_device=EMBEDDING_DEVICE,
    top_k=VECTOR_SEARCH_TOP_K,
    )
    uvicorn.run(app, args.host, args.port, workers=1)
