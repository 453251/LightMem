import json

# 1. 读 search results
with open("NaiveRAG_MobileBench_40_0_1.json", "r", encoding="utf-8") as f:
    retrievals = json.load(f)

sub = retrievals[68:72]  # 或者其他你怀疑的范围

# 2. 复用 memory_evaluation.py 里的逻辑
from memories.datasets.base import QuestionAnswerPair
from inference_utils.operators import QuestionAnsweringOperator
from memory_evaluation import answer_questions
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量

for item in sub:
    item["qa_pair"] = QuestionAnswerPair(**item["qa_pair"])

responses = answer_questions(
    sub,
    qa_model="gpt-4o-mini",
    qa_batch_size=4,  # 或改成 1 看表现有什么区别
    interface_kwargs={
        "api_keys": [os.environ["OPENAI_API_KEY"]],
        "base_urls": [os.environ.get("OPENAI_API_BASE")],
    },
)
print(len(responses))
