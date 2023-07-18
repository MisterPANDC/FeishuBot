from fastapi import Body, FastAPI, File, Form, Query, UploadFile, WebSocket, Request

import nltk #nltk什么作用还需要进一步研究
import uvicorn, json, datetime
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

local_doc_qa = LocalDocQA()
local_doc_qa.init_cfg(
    llm_model=llm_model_ins,
    embedding_model=EMBEDDING_MODEL,
    embedding_device=EMBEDDING_DEVICE,
    top_k=VECTOR_SEARCH_TOP_K,
    )
local_doc_id = "lab"#当前只测试lab即可，后面会作为可以传入的参数

_, _, _, vs_path = create_path(local_doc_id)

async def local_doc_chat(
        knowledge_base_id: str = Body(..., description="Knowledge Base Name", example="kb1"),
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    vs_path = get_vs_path(knowledge_base_id)
    if not os.path.exists(vs_path):
        return 0
    else:
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=question, vs_path=vs_path, chat_history=history, streaming=True
        ):
            pass
        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]

        return ChatMessage(
            question=question,
            response=resp["result"],
            history=history,
            source_documents=source_documents,
        )

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
    api_start(args.host, args.port, ssl_keyfile=args.ssl_keyfile, ssl_certfile=args.ssl_certfile)
