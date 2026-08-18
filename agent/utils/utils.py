import os
import json
import re
from typing import Dict, Any

# 本地存储
class LocalStorage:
    # 存储文件路径
    agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(agent_dir, "data")
    storage_path = os.path.join(config_dir, "mnma_storage.json")

    # 检查并确保存储文件存在
    @classmethod
    def ensure_storage_file(cls):
        # 确保配置目录存在
        if not os.path.exists(cls.config_dir):
            os.makedirs(cls.config_dir)

        # 确保存储文件存在
        if not os.path.exists(cls.storage_path):
            with open(cls.storage_path, "w") as f:
                json.dump({}, f)

    # 读取存储数据
    @classmethod
    def read(cls) -> dict:
        cls.ensure_storage_file()
        try:
            with open(cls.storage_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # 存储文件格式错误时重置为空对象
            with open(cls.storage_path, "w") as f:
                json.dump({}, f)
            return {}

    # 获取存储值
    @classmethod
    def get(cls, task: str, key: str) -> str | bool | int | None:
        storage = cls.read()
        task_storage = storage.get(task)
        if task_storage is None:
            return None
        return task_storage.get(key)

    # 写入存储数据到文件
    @classmethod
    def write(cls, storage: dict) -> bool:
        try:
            with open(cls.storage_path, "w") as f:
                json.dump(storage, f)
            return True
        except Exception as e:
            print(f"存储数据时出错: {e}")
            return False

    # 设置存储值
    @classmethod
    def set(cls, task: str, key: str, value: str | bool | int) -> bool:
        storage = cls.read()
        if task not in storage:
            storage[task] = {}
        storage[task][key] = value

        return cls.write(storage)