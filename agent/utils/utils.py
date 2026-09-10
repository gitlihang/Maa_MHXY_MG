# -*- coding: utf-8 -*-
from maa.custom_action import CustomAction

from typing import Dict, Any, Optional
from contextlib import contextmanager

import json
import os
import re
import threading
import time
import tempfile

# ===== 多实例数据隔离 =====
# 文档: docs/3.3-ProjectInterfaceV2协议.md -> "Agent 子进程环境变量"
#
# 注意：PI_CONTROLLER 只是 interface.json 里"当前选中的那条 controller 数组元素"，
# 对本项目的 Adb 控制器而言它形如 {"name":"ADB","type":"Adb"}，**不包含运行期选中的
# 模拟器地址**，因此无法区分多开实例（旧实现会把所有实例都归到 ctrl::ADB 上而互相串数据）。
# 真正可靠的实例标识是 MaaFramework 的控制器 uuid（context.tasker.controller.uuid），
# 每个模拟器/设备各不相同，所以优先使用它，PI_CONTROLLER 仅作兜底。
DEFAULT_INSTANCE_ID = "default"

# 只提示一次，避免刷屏
_FALLBACK_WARNED = False
_LOCK_WARNED = False

# 抢不到存储锁时最多等待的秒数（超时后退化为无锁执行，保证功能可用）
LOCK_TIMEOUT_SECONDS = 10.0


def _warn_lock_timeout() -> None:
    global _LOCK_WARNED
    if _LOCK_WARNED:
        return
    _LOCK_WARNED = True
    try:
        from .logger import logger

        logger.warning(
            f"等待存储锁超过 {LOCK_TIMEOUT_SECONDS:g}s，本次写入未加锁，"
            "多实例并发时可能覆盖彼此的数据。"
        )
    except Exception:
        print(f"[LocalStorage] 等待存储锁超过 {LOCK_TIMEOUT_SECONDS:g}s，本次写入未加锁")


def _warn_fallback(message: str) -> None:
    global _FALLBACK_WARNED
    if _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED = True
    try:
        # 延迟导入，避免 utils 包初始化期的循环导入
        from .logger import logger

        logger.warning(message)
    except Exception:
        print(f"[LocalStorage] {message}")


def _controller_instance_id(context: Any) -> str:
    """从 MaaFramework 控制器取设备 uuid，这是区分多开实例的最可靠标识。"""
    if context is None:
        return ""
    try:
        device_uuid = context.tasker.controller.uuid
    except Exception:
        return ""
    if not device_uuid:
        return ""
    return f"controller::{device_uuid}"


def _pi_controller_instance_id() -> str:
    """兜底：解析 PI_CONTROLLER 环境变量（Adb 控制器通常取不到地址，故区分能力有限）。"""
    raw = os.environ.get("PI_CONTROLLER", "")
    if not raw:
        return ""

    try:
        ctrl = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""

    if not isinstance(ctrl, dict):
        return ""

    # 兼容顶层 address 与 adb.address 两种布局
    address = ctrl.get("address")
    adb = ctrl.get("adb")
    if isinstance(adb, dict):
        address = adb.get("address") or address
    if address:
        return f"adb::{address}"

    name = ctrl.get("name")
    if name:
        return f"ctrl::{name}"

    return ""


def get_instance_id(context: Any = None) -> str:
    """获取当前实例的隔离标识。

    优先级：
    1. context.tasker.controller.uuid —— 每个模拟器/设备唯一，多开时可真正隔离；
    2. PI_CONTROLLER 环境变量（address / name）；
    3. DEFAULT_INSTANCE_ID。

    注意：不使用 hash() 作为 key，因其每次进程运行结果不稳定，无法跨实例稳定区分多开。
    """
    instance_id = _controller_instance_id(context)
    if instance_id:
        return instance_id

    instance_id = _pi_controller_instance_id()
    if instance_id:
        if instance_id.startswith("ctrl::"):
            _warn_fallback(
                "PI_CONTROLLER 不含设备地址，多开实例会共用存储作用域 "
                f"({instance_id})；请在自定义动作中传入 context 以启用 controller.uuid 隔离。"
            )
        return instance_id

    _warn_fallback(
        "未取得实例标识（未传入 context 且 PI_CONTROLLER 未注入），"
        f"所有实例将共用 {DEFAULT_INSTANCE_ID} 存储作用域。"
    )
    return DEFAULT_INSTANCE_ID


# ===== 跨进程互斥 =====
# 多开时每个实例是一个独立的 agent 进程，而存储文件是同一个，
# 因此"读取-修改-写回"必须整体互斥，否则会互相覆盖丢更新。
_THREAD_LOCK = threading.RLock()


@contextmanager
def _storage_lock(lock_path: str):
    """同进程用线程锁串行化，跨进程用文件锁（Windows: msvcrt，POSIX: fcntl）。

    加锁失败时退化为无锁执行，保证功能可用。
    """
    with _THREAD_LOCK:
        handle = None
        locked = False
        try:
            try:
                handle = open(lock_path, "a+b")
            except OSError:
                handle = None

            if handle is not None:
                try:
                    if os.name == "nt":
                        import msvcrt

                        handle.seek(0, os.SEEK_END)
                        if handle.tell() == 0:
                            handle.write(b"\0")
                            handle.flush()

                        # 不能用 msvcrt.LK_LOCK：它每秒才重试一次、10 秒后直接抛
                        # OSError，多开并发抢锁时容易超时退化成无锁而丢数据。
                        # 改用非阻塞加锁 + 5ms 自旋，等待更平滑。
                        deadline = time.time() + LOCK_TIMEOUT_SECONDS
                        while True:
                            try:
                                handle.seek(0)
                                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                                locked = True
                                break
                            except OSError:
                                if time.time() >= deadline:
                                    break
                                time.sleep(0.005)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                        locked = True
                except Exception:
                    locked = False

                if not locked:
                    _warn_lock_timeout()

            yield
        finally:
            if handle is not None:
                try:
                    if locked:
                        if os.name == "nt":
                            import msvcrt

                            handle.seek(0)
                            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                        else:
                            import fcntl

                            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                try:
                    handle.close()
                except OSError:
                    pass


# 本地存储
class LocalStorage:
    # 存储文件路径
    agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(agent_dir, "data")
    storage_path = os.path.join(config_dir, "mnma_storage.json")
    lock_path = os.path.join(config_dir, "mnma_storage.lock")

    # 检查并确保存储文件存在
    @classmethod
    def ensure_storage_file(cls):
        # 确保配置目录存在
        if not os.path.exists(cls.config_dir):
            os.makedirs(cls.config_dir, exist_ok=True)

        # 确保存储文件存在
        if not os.path.exists(cls.storage_path):
            try:
                with open(cls.storage_path, "w", encoding="utf-8") as f:
                    json.dump({}, f)
            except OSError:
                pass

    # 读取存储数据
    @classmethod
    def read(cls) -> dict:
        cls.ensure_storage_file()
        for _ in range(3):
            try:
                with open(cls.storage_path, "r", encoding="utf-8") as f:
                    raw = f.read()
            except (FileNotFoundError, OSError):
                return {}

            if not raw.strip():
                return {}

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # 可能刚好撞上另一个进程的写入窗口，稍候重试
                time.sleep(0.05)
                continue

            return data if isinstance(data, dict) else {}

        # 重试后仍是坏数据：备份走人，绝不直接清空（旧实现会把其它实例的数据一起抹掉）
        cls._quarantine_corrupt_file()
        return {}

    @classmethod
    def _quarantine_corrupt_file(cls) -> None:
        backup = f"{cls.storage_path}.corrupt.{int(time.time())}"
        try:
            os.replace(cls.storage_path, backup)
        except OSError:
            return

        try:
            from .logger import logger

            logger.warning(f"存储文件内容损坏，已备份到 {backup} 并重建")
        except Exception:
            print(f"[LocalStorage] 存储文件内容损坏，已备份到 {backup} 并重建")

        try:
            with open(cls.storage_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
        except OSError:
            pass

    @staticmethod
    def _scoped_key(task: str, context: Any = None) -> str:
        """为 task 分组键加上实例前缀，实现多实例下的数据隔离。

        例如：node_success -> controller::<设备uuid>::node_success
        不同实例写同一个 mnma_storage.json 时，彼此的状态互不干扰。
        """
        return f"{get_instance_id(context)}::{task}"

    @classmethod
    def _load_bucket(cls, storage: dict, task: str, context: Any = None) -> tuple:
        """取出 task 对应的数据桶，兼容隔离改造前遗留的无前缀数据。"""
        scoped = cls._scoped_key(task, context)
        bucket = storage.get(scoped)
        if isinstance(bucket, dict):
            return scoped, bucket

        # 兼容旧版（无实例前缀）的历史数据，读到后会在写回时迁移到新 key 下
        legacy = storage.get(task)
        if isinstance(legacy, dict):
            return scoped, dict(legacy)

        return scoped, {}

    # 获取存储值
    @classmethod
    def get(cls, task: str, key: str, default: Any = None, context: Any = None) -> Any:
        storage = cls.read()
        _, bucket = cls._load_bucket(storage, task, context)
        return bucket.get(key, default)

    # 写入存储数据到文件（先写临时文件再原子替换，避免读到半截 JSON）
    @classmethod
    def write(cls, storage: dict) -> bool:
        tmp_path = None
        try:
            cls.ensure_storage_file()
            fd, tmp_path = tempfile.mkstemp(
                prefix=".mnma_storage_", suffix=".tmp", dir=cls.config_dir
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(storage, f, ensure_ascii=False, indent=4)
            # 刻意不做 fsync：实测约 57ms/次，会把临界区撑大十几倍，导致多开抢锁超时。
            # os.replace 已保证读方看到的是完整文件，断电最多丢失最后一次计数。

            # os.replace 是原子替换；Windows 上目标文件被瞬时占用时重试
            last_error = None
            for _ in range(5):
                try:
                    os.replace(tmp_path, cls.storage_path)
                    last_error = None
                    break
                except PermissionError as e:
                    last_error = e
                    time.sleep(0.05)
            if last_error is not None:
                raise last_error

            tmp_path = None
            return True
        except Exception as e:
            print(f"存储数据时出错: {e}")
            return False
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # 设置存储值
    @classmethod
    def set(cls, task: str, key: str, value: Any, context: Any = None) -> bool:
        cls.ensure_storage_file()
        with _storage_lock(cls.lock_path):
            storage = cls.read()
            scoped, bucket = cls._load_bucket(storage, task, context)
            bucket[key] = value
            storage[scoped] = bucket
            return cls.write(storage)

    # 原子自增：整个"读取-计算-写回"过程持锁，多实例并发也不会丢更新
    @classmethod
    def increment(
        cls,
        task: str,
        key: str,
        step: int = 1,
        default: int = 0,
        context: Any = None,
    ) -> Optional[int]:
        """把 task.key 原子地加上 step 并返回新值；写入失败时返回 None。"""
        cls.ensure_storage_file()
        with _storage_lock(cls.lock_path):
            storage = cls.read()
            scoped, bucket = cls._load_bucket(storage, task, context)

            current = bucket.get(key, default)
            # bool 是 int 的子类，需排除；非数值一律回退到默认值
            if isinstance(current, bool) or not isinstance(current, (int, float)):
                current = default

            new_value = current + step
            bucket[key] = new_value
            storage[scoped] = bucket

            if not cls.write(storage):
                return None
            return new_value
