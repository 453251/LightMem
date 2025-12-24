from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Any, List

from .base import (
    MemoryDataset,
    Trajectory,
    Session,
    QuestionAnswerPair,
    Message,
)


def _parse_session_datetime(dt_str: str) -> datetime:
    """Parse LOCOMO-style datetime string like '1:56 pm on 8 May, 2023'."""
    # Example: "1:56 pm on 8 May, 2023"
    return datetime.strptime(dt_str, "%I:%M %p on %d %B, %Y")


class LOCOMO(MemoryDataset):
    """Dataset wrapper for LOCOMO-style long-term multi-session dialogs."""

    @classmethod
    def read_raw_data(cls, path: str) -> LOCOMO:
        """
        读取 locomo10.json 这类数据集，并构造
        - trajectories: List[Trajectory]
        - question_answer_pair_lists: List[List[QuestionAnswerPair]]
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trajectories: List[Trajectory] = []
        qa_lists: List[List[QuestionAnswerPair]] = []

        for sample_idx, sample in enumerate(data):
            conversation = sample.get("conversation", {})
            speaker_a = conversation.get("speaker_a", "SpeakerA")
            speaker_b = conversation.get("speaker_b", "SpeakerB")

            # ===== 构造 sessions =====
            sessions: List[Session] = []
            session_time_map: Dict[int, datetime] = {}

            # 先把所有 session_X_date_time 解析出来
            for key, value in conversation.items():
                # 只处理类似 "session_1_date_time" 的键
                if key.startswith("session_") and key.endswith("_date_time"):
                    # "session_1_date_time" -> "1"
                    parts = key.split("_")
                    if len(parts) < 3:
                        continue
                    try:
                        idx = int(parts[1])
                    except ValueError:
                        continue

                    dt_str = value
                    session_dt = _parse_session_datetime(dt_str)
                    session_time_map[idx] = session_dt

            # 按 session 序号排序构造 Session 对象
            for s_idx in sorted(session_time_map.keys()):
                msg_key = f"session_{s_idx}"
                raw_msgs = conversation.get(msg_key, [])
                if not raw_msgs:
                    continue

                session_dt = session_time_map[s_idx]
                messages: List[Message] = []

                for msg in raw_msgs:
                    speaker = msg.get("speaker", "")
                    text = msg.get("text", "")
                    if "blip_caption" in msg and msg["blip_caption"]:
                        text = f"{text} (image description: {msg['blip_caption']})"

                    # 简单规则：speaker_a 作为 "user"，speaker_b 作为 "assistant"
                    if speaker == speaker_a:
                        role = "user"
                    elif speaker == speaker_b:
                        role = "assistant"
                    else:
                        # 不认识的说话人，兜底成 "user"
                        role = "user"

                    messages.append(
                        Message(
                            role=role,
                            content=text,
                            timestamp=session_dt,
                        )
                    )

                sessions.append(
                    Session(
                        messages=messages,
                        timestamp=session_dt,
                        metadata={
                            "id": f"locomo_session_{sample_idx}_{s_idx}",
                            "speaker_a": speaker_a,
                            "speaker_b": speaker_b,
                        },
                    )
                )

            # 构造 Trajectory
            trajectory = Trajectory(
                sessions=sessions,
                metadata={
                    "id": f"locomo_{sample_idx}",
                },
            )
            trajectories.append(trajectory)

            # ===== 构造 question_answer_pair 列表 =====
            # 对 LOCOMO 来说，问题可以认为是在对完整对话之后提问，
            # 因此时间戳就统一放在最后一个 session 的时间上。
            if sessions:
                question_ts = sessions[-1].timestamp
            else:
                # 万一没有 session，就用当前时间兜底
                question_ts = datetime.now()

            qapairs: List[QuestionAnswerPair] = []
            for q_idx, qa in enumerate(sample.get("qa", [])):
                # 有些条目只有 adversarial_answer，没有 answer
                answer = qa.get("answer")
                if answer is None:
                    answer = qa.get("adversarial_answer")

                if isinstance(answer, int):
                    answer = str(answer)

                if answer is not None:
                    answer_list = (answer,)
                else:
                    # 实在没有就给个空 tuple；评测端如果用到可以自己处理
                    answer_list = tuple()

                category = qa.get("category")
                metadata: Dict[str, Any] = {
                    "id": f"locomo_q_{sample_idx}_{q_idx}",
                    # 和 LongMemEval 一样，给一个 question_type 方便做统计
                    "question_type": f"category_{category}",
                    "category": category,
                    "evidence": qa.get("evidence", []),
                }
                if "adversarial_answer" in qa:
                    metadata["adversarial_answer"] = qa["adversarial_answer"]

                qapairs.append(
                    QuestionAnswerPair(
                        role="user",
                        question=qa["question"],
                        answer_list=answer_list,
                        timestamp=question_ts,
                        metadata=metadata,
                    )
                )

            qa_lists.append(qapairs)

        return cls(
            trajectories=trajectories,
            question_answer_pair_lists=qa_lists,
        )

    def _generate_metadata(self) -> Dict[str, Any]:
        """生成数据集的一些统计信息，风格仿照 LongMemEval。"""
        dataset_metadata: Dict[str, Any] = {
            "name": "LOCOMO",
            # 这里的 paper / codebase_url 你可以换成真实信息
            "paper": "LOCOMO: Long Conversation Memory Benchmark",
            "codebase_url": "",
            "total_sessions": 0,
            "total_messages": 0,
            "total_questions": 0,
            "size": len(self),
        }

        question_type_stats: Dict[str, int] = {}

        for trajectory, qa_list in self:
            dataset_metadata["total_sessions"] += len(trajectory)
            dataset_metadata["total_messages"] += sum(len(session) for session in trajectory)
            dataset_metadata["total_questions"] += len(qa_list)

            for qa in qa_list:
                q_type = qa.metadata.get("question_type", "unknown")
                question_type_stats[q_type] = question_type_stats.get(q_type, 0) + 1

        dataset_metadata["question_type_stats"] = question_type_stats

        if len(self) > 0 and dataset_metadata["total_sessions"] > 0:
            dataset_metadata["avg_session_per_trajectory"] = (
                dataset_metadata["total_sessions"] / len(self)
            )
            dataset_metadata["avg_message_per_session"] = (
                dataset_metadata["total_messages"] / dataset_metadata["total_sessions"]
            )
            dataset_metadata["avg_question_per_trajectory"] = (
                dataset_metadata["total_questions"] / len(self)
            )
        else:
            dataset_metadata["avg_session_per_trajectory"] = 0.0
            dataset_metadata["avg_message_per_session"] = 0.0
            dataset_metadata["avg_question_per_trajectory"] = 0.0

        return dataset_metadata
