from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional, Literal, Union

from pydantic import BaseModel, Field, model_validator

from .base import BaseMemoryLayer

from mem0.memory.main import Memory  # type: ignore

logger = logging.getLogger(__name__)


class NaiveRAGConfig(BaseModel):
    """NaiveRAG 的默认配置（对齐通用 memory_construction / memory_search 的入参约定）"""

    # ===== 通用脚本会强制覆盖/注入的字段 =====
    user_id: str = Field(..., description="The user id of the memory system.")

    # 通用脚本会设置：config['save_dir'] = f"{layer_type}/{user_id}"
    save_dir: str = Field(
        default="vector_store/naive_rag",
        description="The directory to persist vector store and config.",
    )

    # ===== 与 memzero 对齐的字段名（便于共用 config.json）=====
    retriever_name_or_path: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name/path (HF or OpenAI embedding model).",
    )

    embedding_model_dims: int = Field(
        default=384,
        description="Embedding dimension.",
    )

    use_gpu: str = Field(
        default="cpu",
        description="Device for embedding model, e.g. 'cpu' or 'cuda'.",
    )

    llm_backend: Literal["openai", "ollama"] = Field(
        default="openai",
        description="LLM backend provider (kept for consistency).",
    )

    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model name (kept for consistency).",
    )

    # ===== 向量库/嵌入 provider =====
    vector_store_provider: Literal["qdrant", "chroma"] = Field(
        default="qdrant",
        description="Vector store provider for mem0.",
    )

    # 如果不填，则默认用 user_id
    collection_name: Optional[str] = Field(
        default=None,
        description="Vector store collection name; defaults to user_id.",
    )

    embedder_provider: Literal["huggingface", "openai"] = Field(
        default="huggingface",
        description="Embedder provider.",
    )

    # ✅关键：qdrant 是否开启 on-disk 持久化
    # mem0 文档里 qdrant 有 on_disk 选项，默认可能是 False
    qdrant_on_disk: bool = Field(
        default=True,
        description="Enable Qdrant persistent storage (on_disk).",
    )

    # ✅可选：add 阶段每写入 N 次就做一次快速自检（避免跑完才发现 0 points）
    add_self_check_every: int = Field(
        default=200,
        description="Do a lightweight get_all check every N adds (0 disables).",
    )

    @model_validator(mode="after")
    def _validate_and_fill(self) -> "NaiveRAGConfig":
        if os.path.isfile(self.save_dir):
            raise AssertionError(
                f"Provided path ({self.save_dir}) should be a directory, not a file"
            )
        if not self.collection_name:
            self.collection_name = self.user_id
        return self


class NaiveRAGLayer(BaseMemoryLayer):
    """Naive RAG 层：用 mem0 的 Memory 作为简单写入与检索后端。"""

    layer_type: str = "NaiveRAG"

    def __init__(self, config: NaiveRAGConfig) -> None:
        self.config = config
        self.memory_config = self._build_memory_config()

        # 计数器：用于 add 自检
        self._add_counter = 0

        try:
            self.memory_layer = Memory.from_config(self.memory_config)  # type: ignore
            logger.info(
                f"NaiveRAGLayer initialized for user={self.config.user_id}, "
                f"vs_provider={self.config.vector_store_provider}, save_dir={self.config.save_dir}, "
                f"collection={self.config.collection_name}, on_disk={self.config.qdrant_on_disk}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize NaiveRAG(mem0): {e}")
            raise RuntimeError(f"Failed to initialize NaiveRAG(mem0): {e}") from e

    def _build_memory_config(self) -> Dict[str, Any]:
        """Build mem0 configuration dict."""
        # embedder
        if self.config.embedder_provider == "huggingface":
            embedder_cfg: Dict[str, Any] = {
                "provider": "huggingface",
                "config": {
                    "model": self.config.retriever_name_or_path,
                    "embedding_dims": self.config.embedding_model_dims,
                    "model_kwargs": {"device": self.config.use_gpu},
                },
            }
        elif self.config.embedder_provider == "openai":
            embedder_cfg = {
                "provider": "openai",
                "config": {
                    "model": self.config.retriever_name_or_path,
                    "embedding_dims": self.config.embedding_model_dims,
                },
            }
        else:
            raise ValueError(f"Unsupported embedder_provider: {self.config.embedder_provider}")

        vector_store_cfg: Dict[str, Any] = {
            "collection_name": self.config.collection_name,
            "embedding_model_dims": self.config.embedding_model_dims,
            "path": self.config.save_dir,
        }

        # ✅关键：qdrant 打开 on_disk，否则经常出现“文件建了但 points=0”
        if self.config.vector_store_provider == "qdrant":
            vector_store_cfg["on_disk"] = self.config.qdrant_on_disk

        return {
            "llm": {
                "provider": self.config.llm_backend,
                "config": {
                    "model": self.config.llm_model,
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "openai_base_url": os.environ.get("OPENAI_API_BASE"),
                },
            },
            "vector_store": {
                "provider": self.config.vector_store_provider,
                "config": vector_store_cfg,
            },
            "embedder": embedder_cfg,
        }

    # ==================== 持久化相关 ====================

    def _save_config(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)
        config_path = os.path.join(self.config.save_dir, "config.json")

        # 只保存可复现/可 load 的字段（不写 api_key）
        config_dict = {
            "layer_type": self.layer_type,
            "user_id": self.config.user_id,
            "save_dir": self.config.save_dir,
            "retriever_name_or_path": self.config.retriever_name_or_path,
            "embedding_model_dims": self.config.embedding_model_dims,
            "use_gpu": self.config.use_gpu,
            "llm_backend": self.config.llm_backend,
            "llm_model": self.config.llm_model,
            "vector_store_provider": self.config.vector_store_provider,
            "collection_name": self.config.collection_name,
            "embedder_provider": self.config.embedder_provider,
            "qdrant_on_disk": self.config.qdrant_on_disk,
            "add_self_check_every": self.config.add_self_check_every,
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)

    def save_memory(self) -> None:
        """
        qdrant/chroma 多数情况下是即写即存。
        但我们至少写 config.json，保证 search 阶段可以重建。
        同时 best-effort 触发 persist/close。
        """
        try:
            os.makedirs(self.config.save_dir, exist_ok=True)
            self._save_config()

            # best-effort: 触发底层落盘（不同版本实现不同）
            vs = getattr(self.memory_layer, "vector_store", None)
            if vs is not None:
                if hasattr(vs, "persist"):
                    try:
                        vs.persist()
                    except Exception:
                        pass
                client = getattr(vs, "client", None)
                if client is not None and hasattr(client, "close"):
                    try:
                        client.close()
                    except Exception:
                        pass

            logger.info(
                f"NaiveRAG saved config for user {self.config.user_id} at {self.config.save_dir}"
            )
        except Exception as e:
            logger.error(f"Error saving NaiveRAG config for user {self.config.user_id}: {e}")
            raise RuntimeError(f"Error saving NaiveRAG config for user {self.config.user_id}: {e}") from e

    def load_memory(self, user_id: Optional[str] = None) -> bool:
        """
        通用脚本依赖这个返回值：
        - memory_construction：未 rerun 时，如果 load_memory=True 就跳过构建
        - memory_search：strict=True 时，如果 False 会抛错
        """
        if user_id is None:
            user_id = self.config.user_id

        # 优先用 config.json 重建（如果存在）
        config_path = os.path.join(self.config.save_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)

                # 以传入 user_id 为准（通用脚本会这样调用）
                cfg["user_id"] = user_id
                if not cfg.get("collection_name"):
                    cfg["collection_name"] = user_id

                self.config = NaiveRAGConfig(**cfg)
                self.memory_config = self._build_memory_config()
                self.memory_layer = Memory.from_config(self.memory_config)  # type: ignore
            except Exception as e:
                logger.warning(f"[NaiveRAG] Failed to rebuild from config.json: {e}")

        # 最后：用 get_all 判断是否真有点
        has_any = self._has_any_memory(user_id=user_id)
        logger.info(f"[NaiveRAG] load_memory(user_id={user_id}) -> {'FOUND' if has_any else 'EMPTY'}")
        return has_any

    def _has_any_memory(self, user_id: str) -> bool:
        """
        兼容 mem0 不同版本的 get_all 返回结构。
        """
        try:
            existing = self.memory_layer.get_all(user_id=user_id, limit=1)  # type: ignore
        except TypeError:
            # 有些版本改成 filters
            existing = self.memory_layer.get_all(filters={"AND": [{"user_id": user_id}]}, limit=1)  # type: ignore
        except Exception as e:
            logger.warning(f"[NaiveRAG] get_all failed for user {user_id}: {e}")
            return False

        if isinstance(existing, dict):
            results = existing.get("results") or existing.get("memories") or existing.get("data") or []
            return bool(results)
        if isinstance(existing, list):
            return len(existing) > 0
        return False

    # ==================== 写入 ====================

    def add_message(self, message: Dict[str, str], **kwargs) -> None:
        """
        memory_construction 调用：
            layer.add_message({"role": message.role, "content": message.content}, timestamp=...)
        """
        content = message.get("content")
        if content is None:
            raise KeyError("message must contain 'content'")

        # 保留原始 role 进 metadata，便于排查
        raw_role = message.get("role", "user")

        metadata: Dict[str, Any] = {"raw_role": raw_role}
        if "timestamp" in kwargs and kwargs["timestamp"] is not None:
            metadata["timestamp"] = kwargs["timestamp"]

        # ✅关键策略：为了避免 mem0 direct-import 过滤导致“0 points”，统一用 role="user" 写入
        # NaiveRAG baseline 只需要把文本进向量库，不需要严格区分 user/assistant。
        stored_messages = [{"role": "user", "content": content}]

        self.memory_layer.add(
            messages=stored_messages,
            user_id=self.config.user_id,
            infer=False,
            metadata=metadata or None,
        )

        # add 自检：每 N 条做一次轻量检查，早点发现 0 points
        self._add_counter += 1
        if self.config.add_self_check_every and self._add_counter % self.config.add_self_check_every == 0:
            ok = self._has_any_memory(user_id=self.config.user_id)
            if not ok:
                logger.warning(
                    f"[NaiveRAG] self-check failed after {self._add_counter} adds: "
                    f"get_all still empty. save_dir={self.config.save_dir}, collection={self.config.collection_name}"
                )

    def add_messages(self, messages: List[Dict[str, str]], **kwargs) -> None:
        for m in messages:
            self.add_message(m, **kwargs)

    # ==================== 检索 ====================

    def retrieve(
        self, query: str, k: int = 10, **kwargs
    ) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        res = self.memory_layer.search(
            query=query,
            user_id=self.config.user_id,
            limit=k,
        )

        if isinstance(res, dict):
            results = res.get("results") or res.get("memories") or res.get("data") or []
        elif isinstance(res, list):
            results = res
        else:
            results = []

        outputs: List[Dict[str, Union[str, Dict[str, Any]]]] = []
        for item in results:
            content = item.get("memory", "")
            metadata = {kk: vv for kk, vv in item.items() if kk != "memory"}

            out: Dict[str, Union[str, Dict[str, Any]]] = {
                "content": content,
                "metadata": metadata,
            }

            used_content = {
                "memory": content,
                "score": metadata.get("score"),
                "created_at": metadata.get("created_at"),
                "updated_at": metadata.get("updated_at"),
            }
            out["used_content"] = "\n".join(
                f"{kk}: {vv}" for kk, vv in used_content.items() if vv is not None
            )

            outputs.append(out)

        return outputs

    # ==================== 其它接口 ====================

    def delete(self, memory_id: str) -> bool:
        try:
            self.memory_layer.delete(memory_id)  # type: ignore
            return True
        except Exception as e:
            logger.error(f"[NaiveRAG] delete error: {e}")
            return False

    def delete_all(self) -> bool:
        try:
            self.memory_layer.delete_all(user_id=self.config.user_id)  # type: ignore
            return True
        except Exception as e:
            logger.error(f"[NaiveRAG] delete_all error: {e}")
            return False

    def update(self, memory_id: str, **kwargs) -> bool:
        data = kwargs.get("data") or kwargs.get("content")
        if data is None:
            logger.error("[NaiveRAG] update requires 'data' or 'content'")
            return False
        try:
            self.memory_layer.update(memory_id, data)  # type: ignore
            return True
        except Exception as e:
            logger.error(f"[NaiveRAG] update error: {e}")
            return False
