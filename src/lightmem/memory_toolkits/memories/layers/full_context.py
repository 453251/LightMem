from __future__ import annotations
import os
import json
import pickle
import time
from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field, model_validator

from .base import BaseMemoryLayer
from token_monitor import get_tokenizer_for_model
from litellm import token_counter as litellm_token_counter

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

    context_window: int = Field(
        default=8192,
        description="上下文窗口大小（按 token 数近似计），超过后从最早消息开始丢弃",
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

        # token 相关：记录每条消息的 token 数以及总 token 数
        self._token_per_id: Dict[str, int] = {}
        self._total_tokens: int = 0

        # 初始化 tokenizer（与 token_monitor 共享逻辑）
        self._tokenizer = get_tokenizer_for_model(self.config.llm_model)

    # 基础工具
    def _gen_id(self) -> str:
        # 使用时间戳+计数保证基本唯一性
        nid = f"{int(time.time()*1000)}-{len(self._ordered_ids)}"
        return nid

    def _count_tokens(self, text: str) -> int:
        """计算文本的 token 数。

        这里直接复用 litellm 的 token_counter，并使用与 token_monitor 相同的
        get_tokenizer_for_model 返回的 custom_tokenizer，以保证计数一致性。
        """
        try:
            return litellm_token_counter(
                model=self.config.llm_model,
                custom_tokenizer=self._tokenizer,
                text=text,
            )
        except Exception:
            # 兜底：如果外部依赖不可用，退化为按字符数近似
            return len(text)

    def _recalculate_tokens(self) -> None:
        """根据当前内存内容重新计算每条消息和总 token 数。

        用于从旧版本数据加载，或在需要时整体校准。
        """
        self._token_per_id.clear()
        self._total_tokens = 0
        for mid in self._ordered_ids:
            mem = self._memories.get(mid)
            if not mem:
                continue
            content = mem.get("content", "") or ""
            tks = self._count_tokens(content)
            self._token_per_id[mid] = tks
            self._total_tokens += tks

    def _ensure_capacity(self) -> None:
        """在写入后确保总 token 数不超过 context_window。

        超出上限时，从最早的消息开始移除，直到满足约束。
        """
        max_tokens = getattr(self.config, "context_window", None)
        if not max_tokens or max_tokens <= 0:
            return

        while self._ordered_ids and self._total_tokens > max_tokens:
            oldest_id = self._ordered_ids.pop(0)
            # 从 token 统计中减去
            tks = self._token_per_id.pop(oldest_id, 0)
            self._total_tokens -= tks
            # 删除实际内容
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
        参数：
        - query: 搜索关键字（若为空字符串且 mode 未显式指定，则退化为 full）
        - k: 覆盖默认配置的返回条数；当 k = -1 时，返回全部。
        """

        results: List[Dict[str, Union[str, Dict[str, Any]]]] = []

        count = 0
        # 逆序扫描，优先最新内容
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
            if k != -1 and count >= k:
                break
        return results

    def delete(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            # 先更新 token 统计
            tks = self._token_per_id.pop(memory_id, 0)
            self._total_tokens -= tks

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

        # content 更新需要同步更新 token 统计
        if "content" in kwargs:
            new_content = kwargs["content"]
            old_tokens = self._token_per_id.get(memory_id, 0)
            new_tokens = self._count_tokens(new_content)

            # 更新内容
            mem["content"] = new_content

            # 更新 token 统计
            self._token_per_id[memory_id] = new_tokens
            self._total_tokens += new_tokens - old_tokens

            # 更新后确保容量
            self._ensure_capacity()

        if "role" in kwargs:
            mem["role"] = kwargs["role"]
        if "timestamp" in kwargs:
            mem["timestamp"] = kwargs["timestamp"]

        other = {
            k: v
            for k, v in kwargs.items()
            if k not in {"content", "role", "timestamp"}
        }
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
            "context_window": self.config.context_window,
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
            context_window=cfg_dict.get("context_window", 8196),
        )

        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        self._ordered_ids = payload.get("ordered_ids", [])
        self._memories = payload.get("memories", {})
        
        # 重新计算 token 统计并保证容量
        self._recalculate_tokens()
        self._ensure_capacity()
        return True
