# mobilebench_preprocessor.py

from typing import Dict, Any
from memories.datasets.base import Message, Session

"""MobileBench 数据集上的预处理函数"""
def style_message_for_MobileBench(
    message: Message,
    session: Session,
) -> Dict[str, Any]:
    # 获取 role
    role = message.role

    # 从 metadata 里取出 name
    name = message.metadata.get("name", "unknown")

    # 决定 Human / System，这里只是一个示例策略：
    #   - 如果 role 为 "user" -> Human Message
    #   - 否则 -> System Message（比如 Calendar / Search / Agent 等）
    if role.lower() == "user":
        header = "Human Message"
    else:
        header = "System Message"
    
    # 由于这个函数只适用于 LoCoMo 数据集，所以我们只考虑role全为user，不考虑system message的处理
    # 还得把时间戳加上
    message_timestamp = message.timestamp
    # 最终内容
    content = (
        f"===== {header} =====\n"
        f"Name: {name}\n"
        f"Time: {message_timestamp}\n\n"
        f"{message.content}"
    )

    return {
        "role": role,
        "content": content,
    }