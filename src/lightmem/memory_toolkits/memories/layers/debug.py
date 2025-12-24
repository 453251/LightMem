# debug.py

import json
from mem0 import Memory

from .naive_rag import NaiveRAGConfig, NaiveRAGLayer  # 你自己项目里的路径按需改

from dotenv import load_dotenv
load_dotenv()

CONFIG_PATH = "/disk/disk_4T_2/chenyijun/LightMem/src/lightmem/memory_toolkits/memories/configs/NaiveRAG.json"


def main():
    # 1. 读配置 & 初始化 Memory（跟 NaiveRAG 里方式保持一致）
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    # 这里如果你是通过 NaiveRAGLayer 包了一层，就用那一层来拿 memory
    # 否则可以直接用 mem0.Memory.from_config(cfg_dict)
    naive_cfg = NaiveRAGConfig(**cfg_dict)
    layer = NaiveRAGLayer(naive_cfg)
    mem = layer.memory  # 这就是 mem0.memory.main.Memory 实例

    # 2. 直接走底层 vector_store.list，不走 get_all（绕开 _build_filters_and_metadata）
    raw = mem.vector_store.list(filters={}, limit=10000)

    if isinstance(raw, (list, tuple)):
        vectors = raw[0]
    else:
        vectors = raw

    print("向量库总条数:", len(vectors))

    for v in vectors[:20]:
        payload = v.payload or {}
        print(
            "id =", v.id,
            "| user_id =", payload.get("user_id"),
            "| agent_id =", payload.get("agent_id"),
            "| run_id =", payload.get("run_id"),
            "| memory =", (payload.get("data") or "")[:80].replace("\n", " "),
        )


if __name__ == "__main__":
    main()
