# -*- coding: utf-8 -*-
"""手动记录 (地图名, 坐标)，并判断当前位置是否已经"稳定"。

底层复用 ``utils.LocalStorage``，存储结构：

    task  = "position_tracker"
    key   = f"{adb_serial}+{timestamp_s}"     # 秒级时间戳
            例如 "127.0.0.1:16416+1730000000"
    value = [地图名, x, y]

**完全手动**：record() 你调一次就多一条记录，没有后台定时器。

三种判定模式（``is_stable`` / coordinate 节点的 attach.mode）：

    duration（默认）: 当前坐标连续不变 >= sec 秒；不依赖节点调用频率。
    count           : 当前坐标连续相同 >= min_count 条。
    window          : 旧逻辑，sec 秒内 >= min_count 条且全部相同。

duration / count 模式下，相邻记录间隔超过 max_gap 秒会重置连续段，
避免上次停留留下的旧数据误触发。

典型用法::

    from utils import PositionTracker

    PositionTracker.record(dtm, zb, context=context)      # zb 是 "151,71" 这样的串
    PositionTracker.record(dtm, 151, 71, context=context) # 或者分开传

    PositionTracker.is_stable(mode="duration", sec=5, context=context)
    PositionTracker.is_stable(mode="count", min_count=3, context=context)
    PositionTracker.is_stable(mode="window", sec=10, min_count=6, context=context)

    # 旧接口仍可用
    if PositionTracker.same_within(10, context=context, min_count=3):
        ...
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .utils import LocalStorage, _storage_lock, get_instance_id

TASK_NAME = "position_tracker"

# 默认保留时长（秒）与最多条数
DEFAULT_KEEP_SECONDS = 600.0
DEFAULT_MAX_RECORDS = 200

# 判定 all_same 时默认至少需要多少条记录
DEFAULT_MIN_COUNT = 2

# duration 模式默认要求连续不变多少秒；count/duration 模式允许的相邻记录最大间隔
DEFAULT_STABLE_SECONDS = 5.0
DEFAULT_MAX_GAP_SECONDS = 5.0


# ===== 内部工具 =====

def _now_s() -> int:
    return int(time.time())


def _to_s(ts: Any) -> int:
    """统一成秒级 int。"""
    return int(float(ts))


def _serial(context: Any = None) -> str:
    """取当前实例的 adb 地址，例如 "127.0.0.1:16416"。"""
    iid = get_instance_id(context)
    for prefix in ("adb::", "controller::", "ctrl::"):
        if iid.startswith(prefix):
            return iid[len(prefix):]
    return iid


def _make_key(serial: str, ts_s: int) -> str:
    return f"{serial}+{ts_s}"


def _parse_key(key: str) -> Tuple[str, Optional[int]]:
    """拆 "127.0.0.1:16416+1730000000" -> ("127.0.0.1:16416", 1730000000)。"""
    if "+" not in key:
        return key, None
    serial, _, ts_str = key.rpartition("+")
    try:
        return serial, int(ts_str)
    except ValueError:
        pass
    try:
        return serial, int(float(ts_str))
    except ValueError:
        return serial, None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_COORD_RE = re.compile(r"[,，]")


def _parse_coord(raw: Any) -> Optional[Tuple[float, float]]:
    """把坐标串或序列解析成 (x, y) 浮点元组；解析不出返回 None。

    支持： "151,71" / "151，71" / "(151,71)" / "（151,71）"
           / (151, 71) / [151, 71]
    """
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        try:
            return float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            return None

    if not isinstance(raw, str):
        return None

    s = raw.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    elif s.startswith("（") and s.endswith("）"):
        s = s[1:-1].strip()

    parts = _COORD_RE.split(s, maxsplit=1)
    if len(parts) != 2:
        return None

    try:
        return float(parts[0].strip()), float(parts[1].strip())
    except ValueError:
        return None


def _norm_value(raw: Any) -> Optional[list]:
    """把存储里的原始 value 规范成 [map, x(float), y(float)]；不合法返回 None。"""
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    return [raw[0], _as_float(raw[1]), _as_float(raw[2])]


def _value_key(value: list) -> tuple:
    """转成可哈希、可比较的元组，用于去重和一致性判断。"""
    return tuple(value)


def _resolve_min_count(min_count: Any) -> int:
    """规范化 min_count，至少为 2（小于 2 无意义）。"""
    try:
        n = int(min_count)
    except (TypeError, ValueError):
        n = DEFAULT_MIN_COUNT
    return n if n >= 2 else DEFAULT_MIN_COUNT


# ===== 主类 =====

class PositionTracker:
    """手动记录 (地图名, 坐标)，并对比时间窗口内所有 value 是否一致。"""

    # ---------- 写（手动调用） ----------

    @classmethod
    def record(
        cls,
        map_name: Any,
        x: Any = None,
        y: Any = None,
        context: Any = None,
        timestamp: Any = None,
    ) -> Optional[str]:
        """手动记录一条位置，返回存储 key；写入失败返回 None。

        两种调用方式：

            record(地图名, "151,71")       # 坐标串自动拆成 x, y
            record(地图名, 151, 71)        # 分开传

        参数：
            map_name  —— 地图名，例如 dtm
            x         —— 坐标串 或 x 值
            y         —— y 值（当 x 是坐标串时可省略）
            timestamp —— 秒级时间戳；不传则取当前时间
                        同一秒内重复写会覆盖上一条（key 相同）

        存储 value = [地图名, x(float), y(float)]。
        若 x 是字符串但解析不出两个数，则原样存 [地图名, x, ""]。
        """
        if y is None:
            parsed = _parse_coord(x)
            if parsed is not None:
                px, py = parsed
            else:
                px, py = x, ""
        else:
            px, py = _as_float(x), _as_float(y)

        ts_s = _now_s() if timestamp is None else _to_s(timestamp)
        key = _make_key(_serial(context), ts_s)
        ok = LocalStorage.set(TASK_NAME, key, [map_name, px, py], context=context)
        return key if ok else None

    # ---------- 读 ----------

    @classmethod
    def history(
        cls,
        seconds: Optional[float] = None,
        context: Any = None,
    ) -> List[Dict[str, Any]]:
        """返回按时间升序排列的记录。

        每项形如::

            {
                "key":       "127.0.0.1:16416+1730000000",
                "serial":    "127.0.0.1:16416",
                "timestamp": 1730000000,   # 秒
                "age":       1.23,         # 距今多少秒
                "map":       "长安城",
                "x":         151.0,
                "y":         71.0,
                "value":     ["长安城", 151.0, 71.0],
            }

        seconds 不为 None 时只返回最近该秒数内的记录。
        """
        storage = LocalStorage.read()
        _, bucket = LocalStorage._load_bucket(storage, TASK_NAME, context)

        now = _now_s()
        cutoff = None if seconds is None else now - seconds

        items: List[Dict[str, Any]] = []
        for key, raw in bucket.items():
            serial, ts = _parse_key(key)
            if ts is None:
                continue
            if cutoff is not None and ts < cutoff:
                continue
            value = _norm_value(raw)
            if value is None:
                continue

            items.append({
                "key": key,
                "serial": serial,
                "timestamp": ts,
                "age": float(now - ts),
                "map": value[0],
                "x": value[1],
                "y": value[2],
                "value": value,
            })

        items.sort(key=lambda it: it["timestamp"])
        return items

    @classmethod
    def latest(cls, context: Any = None) -> Optional[Dict[str, Any]]:
        """取最新一条记录，没有则返回 None。"""
        items = cls.history(context=context)
        return items[-1] if items else None

    @classmethod
    def values(
        cls,
        seconds: Optional[float] = None,
        context: Any = None,
    ) -> List[list]:
        """直接返回窗口内所有 value，按时间升序。"""
        return [it["value"] for it in cls.history(seconds=seconds, context=context)]

    # ---------- 对比：只看 value 是否一致 ----------

    @classmethod
    def compare(
        cls,
        seconds: float,
        context: Any = None,
        min_count: int = DEFAULT_MIN_COUNT,
    ) -> Dict[str, Any]:
        """取出 seconds 秒内的所有记录，判断它们的 value 是否完全一致。

        不做距离 / 位移计算，只整体比较 value（即 [地图名, x, y]）。

        参数：
            seconds   —— 时间窗口（秒）
            min_count —— 判定 all_same 所需的**最少条数**（默认 2）
                         例如 min_count=5 表示：必须至少 5 条，且这 5 条全部相同，
                         all_same 才为 True；不足 5 条一律视为"数据不足"。

        返回值::

            {
                "has_data":  bool,   # 窗口内有没有记录（>=1）
                "enough":    bool,   # 条数是否达到 min_count
                "count":     int,    # 记录条数
                "min_count": int,    # 本次使用的门槛
                "all_same":  bool,   # 是否 >= min_count 条且全部一致
                "changed":   bool,   # 是否出现过变化（>=2 条且不同）
                "values":    [...],  # 所有 value，按时间升序
                "distinct":  [...],  # 去重后的 value，按首次出现顺序
                "first":     {...},  # 最早一条完整记录
                "last":      {...},  # 最新一条完整记录
            }

        判定规则（设 n = count, m = min_count, d = distinct 个数）：
            n == 0        -> has_data=False, enough=False, all_same=False, changed=False
            0 < n < m     -> has_data=True,  enough=False, all_same=False
                             changed = (n >= 2 and d >= 2)
            n >= m        -> enough=True
                             all_same = (d == 1)
                             changed  = (d >= 2)
        """
        m = _resolve_min_count(min_count)
        items = cls.history(seconds=seconds, context=context)

        result: Dict[str, Any] = {
            "has_data": False,
            "enough": False,
            "count": len(items),
            "min_count": m,
            "all_same": False,
            "changed": False,
            "values": [],
            "distinct": [],
            "first": None,
            "last": None,
        }

        if not items:
            return result

        result["has_data"] = True
        result["values"] = [it["value"] for it in items]
        result["first"] = items[0]
        result["last"] = items[-1]

        # 去重（保序）
        seen = set()
        distinct: List[list] = []
        for v in result["values"]:
            k = _value_key(v)
            if k not in seen:
                seen.add(k)
                distinct.append(v)
        result["distinct"] = distinct

        n = len(items)
        d = len(distinct)
        result["enough"] = n >= m

        # all_same 必须满足两个条件：够条数 + 全部相同
        if result["enough"] and d == 1:
            result["all_same"] = True

        # changed 只看"是否出现过不同值"，2 条就够判断，不要求 min_count
        if n >= 2 and d >= 2:
            result["changed"] = True

        return result

    @classmethod
    def same_within(
        cls,
        seconds: float,
        context: Any = None,
        min_count: int = DEFAULT_MIN_COUNT,
    ) -> bool:
        """seconds 秒内是否至少有 min_count 条、且这些 value 全部一致。

        数据不足（条数 < min_count）时返回 False —— 没到门槛就不算"一致"。
        """
        return cls.compare(seconds, context=context, min_count=min_count)["all_same"]

    @classmethod
    def changed_within(cls, seconds: float, context: Any = None) -> bool:
        """seconds 秒内 value 是否发生过变化（>=2 条且存在不同值）。

        只有 0 或 1 条时返回 False —— 谈不上"变过"。
        """
        return cls.compare(seconds, context=context)["changed"]

    # ---------- 连续稳定段 ----------

    @classmethod
    def latest_streak(
        cls,
        max_gap: float = DEFAULT_MAX_GAP_SECONDS,
        context: Any = None,
    ) -> Dict[str, Any]:
        """取最后一段连续相同的 value（相邻记录间隔超过 max_gap 秒就断开）。

        返回::

            {
                "value":      [...] or None,
                "count":      int,      # 连续段内记录条数
                "started_at": int,      # 段首秒级时间戳
                "last_at":    int,      # 段尾秒级时间戳
                "duration":   float,    # last_at - started_at
                "gap_broken": bool,     # 是否因间隔过大而提前断开
            }
        """
        result: Dict[str, Any] = {
            "value": None,
            "count": 0,
            "started_at": None,
            "last_at": None,
            "duration": 0.0,
            "gap_broken": False,
        }

        items = cls.history(context=context)
        if not items:
            return result

        try:
            gap_limit = float(max_gap)
        except (TypeError, ValueError):
            gap_limit = DEFAULT_MAX_GAP_SECONDS

        latest = items[-1]
        key = _value_key(latest["value"])
        end = len(items) - 1
        idx = end
        while idx > 0:
            prev, cur = items[idx - 1], items[idx]
            if _value_key(prev["value"]) != key:
                break
            if cur["timestamp"] - prev["timestamp"] > gap_limit:
                result["gap_broken"] = True
                break
            idx -= 1

        streak = items[idx:]
        result["value"] = latest["value"]
        result["count"] = len(streak)
        result["started_at"] = streak[0]["timestamp"]
        result["last_at"] = latest["timestamp"]
        result["duration"] = float(latest["timestamp"] - streak[0]["timestamp"])
        return result

    @classmethod
    def stable_for(
        cls,
        seconds: float,
        context: Any = None,
        max_gap: float = DEFAULT_MAX_GAP_SECONDS,
    ) -> bool:
        """当前坐标是否已经连续不变 >= seconds 秒（至少要有 2 条记录）。

        只看最后一段连续相同的值，因此不要求节点每秒都调用；
        但只要两次记录间隔超过 max_gap 秒，就当作重新出现，从头计时。
        """
        try:
            need = float(seconds)
        except (TypeError, ValueError):
            need = DEFAULT_STABLE_SECONDS
        if need <= 0:
            need = DEFAULT_STABLE_SECONDS

        streak = cls.latest_streak(max_gap=max_gap, context=context)
        return streak["count"] >= 2 and streak["duration"] >= need

    @classmethod
    def consecutive(
        cls,
        min_count: int = DEFAULT_MIN_COUNT,
        context: Any = None,
        max_gap: float = DEFAULT_MAX_GAP_SECONDS,
    ) -> bool:
        """当前坐标是否已经连续相同 >= min_count 条（间隔超 max_gap 秒会重置）。"""
        streak = cls.latest_streak(max_gap=max_gap, context=context)
        return streak["count"] >= _resolve_min_count(min_count)

    @classmethod
    def is_stable(
        cls,
        mode: str = "duration",
        sec: Any = None,
        min_count: Any = None,
        max_gap: Any = DEFAULT_MAX_GAP_SECONDS,
        context: Any = None,
    ) -> bool:
        """统一入口：按 mode 选择判定方式。

        mode:
            "duration"（默认）—— 连续不变 >= sec 秒
            "count"           —— 连续相同 >= min_count 条
            "window"          —— 旧逻辑：sec 秒内 >= min_count 条且全部相同
        """
        m = str(mode or "duration").strip().lower()
        if m in ("window", "legacy", "same_within", "time_window"):
            seconds = DEFAULT_STABLE_SECONDS if sec is None else sec
            count = DEFAULT_MIN_COUNT if min_count is None else min_count
            return cls.same_within(seconds, context=context, min_count=count)

        if m in ("count", "consecutive", "count_streak"):
            return cls.consecutive(min_count, context=context, max_gap=max_gap)

        return cls.stable_for(
            DEFAULT_STABLE_SECONDS if sec is None else sec,
            context=context,
            max_gap=max_gap,
        )

    # ---------- 维护 ----------

    @classmethod
    def prune(
        cls,
        keep_seconds: float = DEFAULT_KEEP_SECONDS,
        max_records: int = DEFAULT_MAX_RECORDS,
        context: Any = None,
    ) -> int:
        """删除过期 / 超量记录，返回删除条数。建议长循环里隔一会儿调一次。"""
        scoped = LocalStorage._scoped_key(TASK_NAME, context)
        now = _now_s()
        removed = 0

        with _storage_lock(LocalStorage.lock_path):
            storage = LocalStorage.read()
            bucket = storage.get(scoped)
            if not isinstance(bucket, dict) or not bucket:
                return 0

            # 1) 过期
            if keep_seconds:
                for key in list(bucket.keys()):
                    _, ts = _parse_key(key)
                    if ts is None or (now - ts) > keep_seconds:
                        del bucket[key]
                        removed += 1

            # 2) 超量：保留最新的 max_records 条
            if max_records and len(bucket) > max_records:
                ordered = sorted(bucket.keys(), key=lambda k: _parse_key(k)[1] or 0)
                for key in ordered[: len(bucket) - max_records]:
                    del bucket[key]
                    removed += 1

            if removed:
                storage[scoped] = bucket
                LocalStorage.write(storage)

        return removed

    @classmethod
    def clear(cls, context: Any = None) -> int:
        """清空本实例的全部位置记录，返回删除条数。"""
        scoped = LocalStorage._scoped_key(TASK_NAME, context)
        with _storage_lock(LocalStorage.lock_path):
            storage = LocalStorage.read()
            bucket = storage.get(scoped)
            if not isinstance(bucket, dict) or not bucket:
                return 0
            count = len(bucket)
            del storage[scoped]
            LocalStorage.write(storage)
            return count