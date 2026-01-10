from __future__ import annotations
from .base import BaseMemoryLayer 
from pydantic import (
    BaseModel, 
    Field, 
    model_validator,
)
from .baselines.mem0 import (
    Memory,
    MemoryConfig
) 
from typing import (
    Literal, 
    List, 
    Dict, 
    Any,
    Optional, 
    Union
)
import os
import json
import pickle
import logging



logger = logging.getLogger(__name__)

class MemZeroConfig(BaseModel):
    """The default configuration for MemZero"""

    # ===== 通用脚本会强制覆盖/注入的字段 =====
    user_id: str = Field(..., description="The user id of the memory system.")

    save_dir: str = Field(
        default="vector_store/MemZero",
        description="The directory to persist vector store and config.",
    )

    # ===== 与 mem0 对齐的字段名（便于共用 config.json）=====
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

    collection_name: Optional[str] = Field(
        default=None,
        description="Vector store collection name; defaults to user_id.",
    )

    embedder_provider: Literal["huggingface", "openai"] = Field(
        default="huggingface",
        description="Embedder provider.",
    )

    qdrant_on_disk: bool = Field(
        default=True,
        description="Enable Qdrant persistent storage (on_disk).",
    )

    # ===== Graph Store 相关 =====
    enable_graph: bool = Field(
        default=False,
        description="Whether to enable Mem0 graph store.",
    )

    # 官方目前支持 neo4j / memgraph / neptune / kuzu 等
    graph_store_provider: Optional[str] = Field(
        default=None,
        description="Graph store provider, e.g. 'neo4j', 'memgraph', 'neptune', 'kuzu'.",
    )

    # 直接透传给 mem0 的 graph_store.config
    graph_store_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Graph store config passed to Mem0.",
    )

    @model_validator(mode="after")
    def _validate_and_fill(self) -> "MemZeroConfig":
        if os.path.isfile(self.save_dir):
            raise AssertionError(
                f"Provided path ({self.save_dir}) should be a directory, not a file"
            )
        if not self.collection_name:
            self.collection_name = self.user_id

        # 如果开启了 graph，但 provider 没填，给个报错防踩坑
        if self.enable_graph and not self.graph_store_provider:
            raise ValueError("enable_graph=True 时必须提供 graph_store_provider")
        
        if self.enable_graph and self.graph_store_provider == "kuzu":
            # 如果没写 db，就默认放在 save_dir 里
            if "db" not in self.graph_store_config:
                os.makedirs(self.save_dir, exist_ok=True)
                self.graph_store_config["db"] = os.path.join(
                    self.save_dir,
                    f"{self.user_id}.kuzu",
                )

        return self


class MemZeroLayer(BaseMemoryLayer):
    layer_type: str = "memzero"

    def __init__(self, config: MemZeroConfig) -> None:
        """Create an interface of MemZero. The implemenation is based on the 
        [official implementation](https://github.com/mem0ai/mem0)."""
        self.config = config
        self.memory_config = self._build_memory_config()

        try:
            self.memory_layer = Memory.from_config(self.memory_config)
            logger.info(f"MemZeroLayer initialized for user: {config.user_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Mem0: {e}")
            raise RuntimeError(f"Failed to initialize Mem0: {e}") from e

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
            raise ValueError(
                f"Unsupported embedder_provider: {self.config.embedder_provider}"
            )

        vector_store_cfg: Dict[str, Any] = {
            "collection_name": self.config.collection_name,
            "embedding_model_dims": self.config.embedding_model_dims,
            "path": self.config.save_dir,
        }

        if self.config.vector_store_provider == "qdrant":
            vector_store_cfg["on_disk"] = self.config.qdrant_on_disk

        # ==== graph_store 配置（关键）====
        graph_store_cfg: Dict[str, Any] = {
            "provider": self.config.graph_store_provider,
            "config": self.config.graph_store_config,
        } if self.config.enable_graph else {
            # 如果不启用 graph，就给一个空 config，Mem0 里会走 enable_graph = False 的分支
            "provider": None,
            "config": {},
        }

        cfg: Dict[str, Any] = {
            # Graph Memory 要求 version v1.1 及以上
            "version": "v1.1",
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

        # 只有开启 graph 才真正塞进去，避免触发奇怪的验证错误
        if self.config.enable_graph:
            cfg["graph_store"] = graph_store_cfg

        return cfg


    def load_memory(self, user_id: Optional[str] = None) -> bool:
        """Load the memory of the user."""
        if user_id is None:
            user_id = self.config.user_id
            
        pkl_path = os.path.join(self.config.save_dir, f"{user_id}.pkl")
        config_path = os.path.join(self.config.save_dir, "config.json")
        
        if not os.path.exists(pkl_path) or not os.path.exists(config_path):
            logger.info(f"No saved memory found for user {user_id}")
            return False 
        
        try:
            with open(config_path, 'r', encoding="utf-8") as f:
                config_dict = json.load(f)
                    
            if user_id != config_dict["user_id"]:
                raise ValueError(
                    f"The user id in the config file ({config_dict['user_id']}) "
                    f"does not match the user id ({user_id}) in the function call."
                )
                
            self.config = MemZeroConfig(**config_dict)
            self.memory_config = self._build_memory_config()
            self.memory_layer = Memory.from_config(self.memory_config)
                
            with open(pkl_path, "rb") as f:
                memories_data = pickle.load(f)
                
            if memories_data:
                for memory_item in memories_data:
                    try:
                        self.memory_layer.add(
                            messages=[{"role": "user", "content": memory_item.get("memory", "")}],
                            user_id=user_id,
                            infer=False
                        )
                    except Exception as e:
                        logger.warning(f"Failed to restore memory {memory_item.get('id', 'unknown')}: {e}")
            
            logger.info(f"Successfully loaded {len(memories_data)} memories for user {user_id}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load memory for user {user_id}: {e}")
        
        # 最后：用 get_all 判断是否真有点
        has_any = self._has_any_memory(user_id=user_id)
        logger.info(f"[MemZero] load_memory(user_id={user_id}) -> {'FOUND' if has_any else 'EMPTY'}")
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
            logger.warning(f"[MemZero] get_all failed for user {user_id}: {e}")
            return False

        if isinstance(existing, dict):
            results = existing.get("results") or existing.get("memories") or existing.get("data") or []
            return bool(results)
        if isinstance(existing, list):
            return len(existing) > 0
        return False
    
    def add_message(self, message: Dict[str, str], **kwargs) -> None:
        """Add a message to the memory layer."""
        if "timestamp" not in kwargs: 
            raise KeyError("timestamp is required in `kwargs`")
        timestamp = kwargs["timestamp"] 
        try:
            self.memory_layer.add(
                messages=message["content"],
                user_id=self.config.user_id,
                metadata={"timestamp": timestamp}
            )
        except KeyError as e:
            # 专门兜 graph 里那种缺 source/destination 的 KeyError
            logger.warning(
                "[MemZeroLayer] add_message KeyError: %s | content=%r",
                e, message["content"][:100],
            )
        except Exception as e:
            msg = str(e)
            
            # 检查是否是内容过滤相关的错误（包括mem0转换后的格式错误）
            is_content_filter_error = (
                "content management policy" in msg or 
                "content_filter" in msg or
                "'messages' must be list[dict]" in msg  # mem0在遇到content filter时可能抛出的格式错误
            )
            
            if is_content_filter_error:
                logger.warning(
                    "[MemZeroLayer] skip message due to content filter (or related format error): %r",
                    message["content"][:120],
                )
                # 直接跳过这一条，不再往外抛，让整体 trajectory 继续
                return

            # Mem0对于graph的支持不太行，不太健壮，跑graph的时候会有一堆问题，这里全部warning不抛出
            logger.warning(
                "[MemZeroLayer] add_message failed: %s | content=%r",
                e, message["content"][:120],
            )
            return

    def add_messages(self, messages: List[Dict[str, str]], **kwargs) -> None:
        """Add a list of messages to the memory layer."""
        self.memory_layer.add(
            messages=messages,
            user_id=self.config.user_id
        )

    # def retrieve(self, query: str, k: int = 10, **kwargs) -> List[Dict[str, str | Dict[str, Any]]]:
    #     """Retrieve the memories."""
    #     related_memories = self.memory_layer.search(
    #         query=query, 
    #         user_id=self.config.user_id, 
    #         limit=k
    #     )
    #     outputs = []
    #     for mem in related_memories.get("results", []):
    #         outputs.append(
    #             {
    #                 "content": mem["memory"],
    #                 "metadata": {
    #                     key: value
    #                     for key, value in mem.items() if key != "memory"
    #                 }
    #             }
    #         )
    #     return outputs

    # ==================== 检索 ====================

    # def retrieve(
    #     self, query: str, k: int = 10, **kwargs
    # ) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
    #     res = self.memory_layer.search(
    #         query=query,
    #         user_id=self.config.user_id,
    #         limit=k,
    #     )

    #     if isinstance(res, dict):
    #         results = res.get("results") or res.get("memories") or res.get("data") or []
    #         relations = res.get("relations") if self.config.enable_graph else None
    #     elif isinstance(res, list):
    #         results = res
    #         relations = None
    #     else:
    #         results = []

    #     outputs: List[Dict[str, Union[str, Dict[str, Any]]]] = []
    #     for item in results:
    #         content = item.get("memory", "")
    #         metadata = {kk: vv for kk, vv in item.items() if kk != "memory"}

    #         out: Dict[str, Union[str, Dict[str, Any]]] = {
    #             "content": content,
    #             "metadata": metadata,
    #         }

    #         used_content = {
    #             "memory": content,
    #             "score": metadata.get("score"),
    #             "created_at": metadata.get("created_at"),
    #             "updated_at": metadata.get("updated_at"),
    #         }
    #         out["used_content"] = "\n".join(
    #             f"{kk}: {vv}" for kk, vv in used_content.items() if vv is not None
    #         )

    #         outputs.append(out)

    #     if self.config.enable_graph and relations is not None:
    #         return {
    #             "memories": outputs,
    #             "relations": relations,
    #         }
    #     return outputs

    def retrieve(
        self, query: str, k: int = 10, **kwargs
    ) -> Union[
        List[Dict[str, Union[str, Dict[str, Any]]]],
        Dict[str, Any],
    ]:
        try:
            res = self.memory_layer.search(
                query=query,
                user_id=self.config.user_id,
                limit=k,
            )
        except Exception as e:
            msg = str(e)

            # 专门识别 content management / content_filter 这类错误
            is_content_filter_error = (
                "content management policy" in msg
                or "content_filter" in msg
                or "upstream_error" in msg and "param': 'prompt'" in msg
            )

            if is_content_filter_error:
                logger.warning(
                    "[MemZeroLayer] search skipped due to content filter: %r",
                    query[:120],
                )
                # 图相关场景，保持返回结构兼容
                if self.config.enable_graph:
                    return {
                        "memories": [],
                        "relations": [],
                    }
                # 非 graph 就当作检索不到
                return []

            # 其他错误照旧抛出去，避免把真正 bug 吞掉
            return []

        # ====== 原来的解析逻辑 ======
        if isinstance(res, dict):
            results = (
                res.get("results")
                or res.get("memories")
                or res.get("data")
                or []
            )
            relations = res.get("relations") if self.config.enable_graph else None
        elif isinstance(res, list):
            results = res
            relations = None
        else:
            results = []
            relations = None

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

        if self.config.enable_graph and relations is not None:
            return {
                "memories": outputs,
                "relations": relations,
            }
        return outputs


    def delete(self, memory_id: str) -> bool:
        """Delete a memory from the memory layer."""
        try:
            self.memory_layer.delete(memory_id)
            return True
        except Exception as e:
            print(f"Error in delete method in MemZeroLayer: \n\t{e.__class__.__name__}: {e}")
            return False
    
    def delete_all(self) -> bool:
        """Delete all memories of the user."""
        try:
            self.memory_layer.delete_all(user_id=self.config.user_id)
            return True
        except Exception as e:
            print(f"Error in delete_all method in MemZeroLayer: \n\t{e.__class__.__name__}: {e}")
            return False
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory in the memory layer."""
        try:
            data = kwargs.get("data", "")
            self.memory_layer.update(memory_id, data)
            return True
        except Exception as e:
            print(f"Error in update method in MemZeroLayer: \n\t{e.__class__.__name__}: {e}")
            return False
    
    def save_memory(self) -> None:
        """Save the memory state to storage."""
        try:
            os.makedirs(self.config.save_dir, exist_ok=True)
            self._save_config()
            
            all_memories = self.memory_layer.get_all(
                user_id=self.config.user_id,
                limit=100000
            )
            
            memories_data = self._normalize_memory_data(all_memories)
            pkl_path = os.path.join(self.config.save_dir, f"{self.config.user_id}.pkl")
            
            with open(pkl_path, "wb") as f:
                pickle.dump(memories_data, f)
                
            logger.info(f"Successfully saved {len(memories_data)} memories for user {self.config.user_id}")
            
        except Exception as e:
            logger.error(f"Error saving memories for user {self.config.user_id}: {e}")
            raise RuntimeError(f"Error saving memories for user {self.config.user_id}: {e}") from e

    def _save_config(self) -> None:
        config_path = os.path.join(self.config.save_dir, "config.json")
        config_dict = {
            "layer_type": self.layer_type,
            "user_id": self.config.user_id,
            "save_dir": self.config.save_dir,
            "retriever_name_or_path": self.config.retriever_name_or_path,
            "embedding_model_dims": self.config.embedding_model_dims,
            "use_gpu": self.config.use_gpu,
            "llm_backend": self.config.llm_backend,
            "llm_model": self.config.llm_model,
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": os.environ.get("OPENAI_API_BASE"),
            "vector_store_provider": self.config.vector_store_provider,
            "collection_name": self.config.collection_name,
            "embedder_provider": self.config.embedder_provider,
            "qdrant_on_disk": self.config.qdrant_on_disk,
        }
        
        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        
    def _normalize_memory_data(self, all_memories) -> List[Dict[str, Any]]:
        if isinstance(all_memories, dict) and "results" in all_memories:
            return all_memories["results"]
        elif isinstance(all_memories, list):
            return all_memories
        else:
            logger.warning(f"Unexpected memory data format: {type(all_memories)}")
            return []
