from __future__ import annotations
import os
import json
import pickle
import time
from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field, model_validator

from .base import BaseMemoryLayer


class FullContextConfig(BaseModel):
    """配置：Full-Text Memory 层（简单全量上下文存储与检索）。

    该层以最小依赖的方式，将消息完整文本存储，并提供：
    - 追加单条/多条消息
    - 简单检索（基于包含/关键词分词匹配）
    - 删除与更新
    - 保存与加载
    """

    user_id: str = Field(..., description="用户唯一 ID")
    save_dir: str = Field(
        default="full_context",
        description="保存内存的目录（包含 config.json 与 .pkl 序列化数据）",
    )
    retrieve_mode: Literal['match', 'full'] = Field(
        default="full",
        description="默认检索模式：match 基于包含关键词匹配，full 返回全部",
    )
    case_sensitive: bool = Field(
        default=False, description="检索是否区分大小写"
    )
    return_all: bool = Field(
        default=True,
        description="在 full 模式下，是否返回全部消息（忽略 k 限制）",
    )
    max_messages: int = Field(
        default=100000,
        description="最多保存的消息条数，超过后从最早开始丢弃",
        gt=0,
    )

    # 没用，兼容通用脚本字段
    llm_backend: Literal["openai", "ollama"] = Field(
        default="openai",
        description="LLM backend provider (kept for consistency).",
    )

    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model name (kept for consistency).",
    )

    @model_validator(mode="after")
    def _validate_save_dir(self) -> FullContextConfig:
        if os.path.isfile(self.save_dir):
            raise AssertionError(
                f"Provided path ({self.save_dir}) should be a directory, not a file"
            )
        return self


class FullContextLayer(BaseMemoryLayer):
    """Full-Text Memory 层：存储完整消息并提供基础检索。"""

    layer_type: str = "full_context"

    def __init__(self, config: FullContextConfig) -> None:
        self.config = config
        # 内存结构：{id: {"content": str, "role": str, "timestamp": float, "metadata": {...}}}
        self._memories: Dict[str, Dict[str, Any]] = {}
        self._ordered_ids: List[str] = []

    # 基础工具
    def _gen_id(self) -> str:
        # 使用时间戳+计数保证基本唯一性
        nid = f"{int(time.time()*1000)}-{len(self._ordered_ids)}"
        return nid

    def _ensure_capacity(self) -> None:
        # 超出上限时，从最早的消息开始移除
        while len(self._ordered_ids) > self.config.max_messages:
            oldest_id = self._ordered_ids.pop(0)
            self._memories.pop(oldest_id, None)

    # 接口实现
    def add_message(self, message: Dict[str, str], **kwargs) -> None:
        if "role" not in message or "content" not in message:
            raise KeyError("message must contain 'role' and 'content'")
        ts = kwargs.get("timestamp", time.time())
        mid = self._gen_id()
        payload = {
            "id": mid,
            "role": message["role"],
            "content": message["content"],
            "timestamp": ts,
            "metadata": {k: v for k, v in kwargs.items() if k != "timestamp"},
        }
        self._memories[mid] = payload
        self._ordered_ids.append(mid)
        self._ensure_capacity()

    def add_messages(self, messages: List[Dict[str, str]], **kwargs) -> None:
        for m in messages:
            self.add_message(m, **kwargs)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        **kwargs,
    ) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        """
        检索模式：
        - mode = "match": 基于包含/分词匹配；按时间逆序返回最多 k 条。
        - mode = "full": 无视查询条件，直接按时间逆序返回全部或最多 k 条。

        额外参数：
        - case_sensitive: bool（仅 match 模式使用），默认取 config.case_sensitive。
        - return_all: bool（仅 full 模式使用），为 True 时忽略 k，返回全部。
        """

        results: List[Dict[str, Union[str, Dict[str, Any]]]] = []
        mode: Literal['match', 'full'] = kwargs.get(
            "mode", self.config.retrieve_mode)

        if mode == "full":
            return_all: bool = kwargs.get("return_all", False)
            count = 0
            for mid in reversed(self._ordered_ids):
                mem = self._memories[mid]
                results.append(
                    {
                        "content": mem["content"],
                        "metadata": {
                            "id": mem["id"],
                            "role": mem["role"],
                            "timestamp": mem["timestamp"],
                            **mem.get("metadata", {}),
                        },
                    }
                )
                count += 1
                if not return_all and count >= k:
                    break
            return results

        # 默认/匹配模式
        if not query:
            return []

        case_sensitive = kwargs.get("case_sensitive", self.config.case_sensitive)
        q = query if case_sensitive else query.lower()

        def match(text: str) -> bool:
            if not case_sensitive:
                text = text.lower()
            if q in text:
                return True
            words = [w for w in q.split() if w]
            return any(w in text for w in words) if words else False

        # 逆序扫描，优先最新内容
        for mid in reversed(self._ordered_ids):
            mem = self._memories[mid]
            if match(mem["content"]):
                results.append(
                    {
                        "content": mem["content"],
                        "metadata": {
                            "id": mem["id"],
                            "role": mem["role"],
                            "timestamp": mem["timestamp"],
                            **mem.get("metadata", {}),
                        },
                    }
                )
                if len(results) >= k:
                    break
        return results

    def delete(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            self._memories.pop(memory_id)
            try:
                self._ordered_ids.remove(memory_id)
            except ValueError:
                pass
            return True
        return False

    def update(self, memory_id: str, **kwargs) -> bool:
        mem = self._memories.get(memory_id)
        if not mem:
            return False
        # 可更新的字段：content, role, timestamp, metadata(合并)
        if "content" in kwargs:
            mem["content"] = kwargs["content"]
        if "role" in kwargs:
            mem["role"] = kwargs["role"]
        if "timestamp" in kwargs:
            mem["timestamp"] = kwargs["timestamp"]
        other = {k: v for k, v in kwargs.items() if k not in {"content", "role", "timestamp"}}
        if other:
            meta = mem.get("metadata", {})
            meta.update(other)
            mem["metadata"] = meta
        return True

    def save_memory(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)

        # 写入 config.json
        cfg_path = os.path.join(self.config.save_dir, "config.json")
        cfg = {
            "layer_type": self.layer_type,
            "user_id": self.config.user_id,
            "case_sensitive": self.config.case_sensitive,
            "max_messages": self.config.max_messages,
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)

        # 写入内存快照 .pkl
        payload = {
            "ordered_ids": self._ordered_ids,
            "memories": self._memories,
        }
        pkl_path = os.path.join(self.config.save_dir, f"{self.config.user_id}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)

    def load_memory(self, user_id: Optional[str] = None) -> bool:
        uid = user_id or self.config.user_id
        cfg_path = os.path.join(self.config.save_dir, "config.json")
        pkl_path = os.path.join(self.config.save_dir, f"{uid}.pkl")
        if not (os.path.exists(cfg_path) and os.path.exists(pkl_path)):
            return False

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        if uid != cfg_dict.get("user_id"):
            raise ValueError(
                f"Config user_id ({cfg_dict.get('user_id')}) does not match requested ({uid})"
            )

        # 更新配置（除保存目录外）
        self.config = FullContextConfig(
            user_id=cfg_dict["user_id"],
            save_dir=self.config.save_dir,
            case_sensitive=cfg_dict.get("case_sensitive", False),
            max_messages=cfg_dict.get("max_messages", 100000),
        )

        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        self._ordered_ids = payload.get("ordered_ids", [])
        self._memories = payload.get("memories", {})
        # 保证容量
        self._ensure_capacity()
        return True
