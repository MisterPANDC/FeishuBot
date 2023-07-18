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
