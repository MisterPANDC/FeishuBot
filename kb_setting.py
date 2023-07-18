import os
from chains.local_doc_qa import LocalDocQA
from configs.model_config import (KB_ROOT_PATH, EMBEDDING_DEVICE,
                                  EMBEDDING_MODEL, NLTK_DATA_PATH,
                                  VECTOR_SEARCH_TOP_K, LLM_HISTORY_LEN, OPEN_CROSS_DOMAIN)

import models.shared as shared

local_doc_qa = LocalDocQA()
llm_model_ins = shared.loaderLLM()
local_doc_qa.init_cfg(
    llm_model=llm_model_ins,
    embedding_model=EMBEDDING_MODEL,
    embedding_device=EMBEDDING_DEVICE,
    top_k=VECTOR_SEARCH_TOP_K,
)
local_doc_id = "lab"#当前只测试lab即可，后面会作为可以传入的参数

def create_path(local_doc_id, file_name = None):
    #local_doc_id指定一个知识库 (sample)
    #知识库储存在根路径下 kb_path返回id对应的知识库的路径 (knowledge_base/samples)
    kb_path = os.path.join(KB_ROOT_PATH, local_doc_id)
    #在知识库中会额外储存一份原文件，路径如下 (knowledge_base/samples/content)
    doc_path = os.path.join(kb_path, "content")
    """
    先手动构造知识库 文件直接放入content文件夹中 直接用embedding来处理即可
    """
    if file_name != None:
        file_path = os.path.join(doc_path, file_name)
    else:
        file_path = None

    vs_path = os.path.join(kb_path, "vector_store")#指定id下具体的向量数据库(knowledge_base/samples/vector_store)
    return kb_path, doc_path, file_path, vs_path

def create_db(file_path, vs_path):
    local_doc_qa.init_knowledge_vector_store([file_path], vs_path) #file_path转为一个list 可以传入多个文件

if __name__ == "__main__":
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    _, _, file_path, vs_path = create_path(local_doc_id, 'test.txt')
    create_db(file_path, vs_path)