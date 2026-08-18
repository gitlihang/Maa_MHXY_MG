from collections.abc import Sequence
import random
import time


class HumanSimulator:
    """人类操作生物学拟人化模拟器
    
    用于消除机械化脚本的特征：
    1. 坐标落点：采用二维正态（截断高斯）分布代替机械矩形中心；
    2. 操作耗时：采用正态/生理延迟代替固定 sleep 时间；
    3. 行为模式：注入随机发呆与注意力分散，打破规律性动作特征。
    """

    @staticmethod
    def get_gaussian_point(
        box: Sequence[int | float],
        margin_ratio: float = 0.05
    ) -> tuple[int, int]:
        """在给定的矩形区域 [x, y, w, h] 内生成符合二维截断高斯分布的随机落点。
        
        95% 以上落点集中在中心区域（3-sigma 原则），四周自然稀疏衰减，并自适应保护边界。
        
        Args:
            box: 矩形区域 [x, y, w, h] 或 (x, y, w, h)
            margin_ratio: 边界最小内缩比例（默认 5% 或 2px）
            
        Returns:
            tuple[int, int]: 拟人化落点坐标 (target_x, target_y)
        """
        if not box or len(box) < 4:
            raise ValueError(f"box 格式错误，期望长度 >= 4 的列表或元组，收到: {box}")

        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # 异常/退化矩形保护
        if w <= 0 or h <= 0:
            return x, y
        if w <= 3:
            target_x = x + w // 2
        else:
            center_x = x + w / 2.0
            sigma_x = max(w / 6.0, 1.0)
            margin_x = min(2, max(0, int(w * margin_ratio)))
            min_x = x + margin_x
            max_x = max(min_x, x + w - margin_x - 1)
            raw_x = random.gauss(center_x, sigma_x)
            target_x = round(max(min_x, min(max_x, raw_x)))

        if h <= 3:
            target_y = y + h // 2
        else:
            center_y = y + h / 2.0
            sigma_y = max(h / 6.0, 1.0)
            margin_y = min(2, max(0, int(h * margin_ratio)))
            min_y = y + margin_y
            max_y = max(min_y, y + h - margin_y - 1)
            raw_y = random.gauss(center_y, sigma_y)
            target_y = round(max(min_y, min(max_y, raw_y)))

        return target_x, target_y

    @staticmethod
    def get_point_offset(
        x: float,
        y: float,
        radius: float = 5.0
    ) -> tuple[int, int]:
        """针对固定坐标点生成带高斯微扰的拟人化落点，避免每次点击同一个绝对像素点。
        
        Args:
            x: 原始 X 坐标
            y: 原始 Y 坐标
            radius: 最大扰动半径范围（默认 5px）
            
        Returns:
            tuple[int, int]: 微扰后的落点坐标 (target_x, target_y)
        """
        sigma = max(radius / 3.0, 1.0)
        offset_x = random.gauss(0, sigma)
        offset_y = random.gauss(0, sigma)
        
        # 截断在最大半径以内
        offset_x = max(-radius, min(radius, offset_x))
        offset_y = max(-radius, min(radius, offset_y))
        
        return round(x + offset_x), round(y + offset_y)

    @staticmethod
    def sleep_human(
        base_seconds: float = 1.0,
        jitter_ratio: float = 0.25,
        min_delay: float = 0.05
    ) -> float:
        """模拟人类生理与操作延迟（带正态分布扰动）。
        
        Args:
            base_seconds: 基准等待秒数
            jitter_ratio: 扰动系数（标准差与均值的比例，默认 0.25）
            min_delay: 允许的最小延迟秒数（防止生成负数或过短延迟）
            
        Returns:
            float: 实际休眠的秒数
        """
        if base_seconds <= 0:
            return 0.0
        sigma = base_seconds * jitter_ratio
        actual_delay = max(min_delay, random.gauss(base_seconds, sigma))
        time.sleep(actual_delay)
        return actual_delay

    @staticmethod
    def sleep_ms(
        base_ms: float,
        jitter_ratio: float = 0.25,
        min_ms: float = 50.0
    ) -> float:
        """毫秒级别的拟人化延迟。
        
        Args:
            base_ms: 基准等待毫秒数
            jitter_ratio: 扰动系数
            min_ms: 允许的最小延迟毫秒数
            
        Returns:
            float: 实际休眠的秒数
        """
        return HumanSimulator.sleep_human(
            base_seconds=base_ms / 1000.0,
            jitter_ratio=jitter_ratio,
            min_delay=min_ms / 1000.0
        )

    @staticmethod
    def random_idle(
        chance: float = 0.05,
        min_sec: float = 3.0,
        max_sec: float = 8.0
    ) -> float:
        """模拟人类挂机/发呆/注意力分散机制（小概率触发）。
        
        Args:
            chance: 触发概率 (0.0 ~ 1.0，如 0.05 表示 5% 概率)
            min_sec: 发呆最短秒数
            max_sec: 发呆最长秒数
            
        Returns:
            float: 若触发则返回发呆时长（秒），否则返回 0.0
        """
        if random.random() < chance:
            idle_time = random.uniform(min_sec, max_sec)
            time.sleep(idle_time)
            return idle_time
        return 0.0

    @classmethod
    def human_click(
        cls,
        controller,
        target: Sequence[int | float],
        pre_delay: float = 0.2,
        post_delay: float = 0.4,
        idle_chance: float = 0.0
    ) -> tuple[int, int]:
        """高级封装：执行一次包含前后拟人延迟与高斯落点的完整点击。
        
        Args:
            controller: Maa 控制器对象 (包含 post_click 方法)
            target: 目标区域 [x, y, w, h] 或 坐标点 [x, y]
            pre_delay: 点击前的生理反应延迟基准时间（秒）
            post_delay: 点击后的操作确认延迟基准时间（秒）
            idle_chance: 点击后随机发呆概率 (0.0 ~ 1.0)
            
        Returns:
            tuple[int, int]: 点击的实际坐标
        """
        if len(target) >= 4:
            tx, ty = cls.get_gaussian_point(target[:4])
        elif len(target) >= 2:
            tx, ty = cls.get_point_offset(target[0], target[1])
        else:
            raise ValueError(f"target 坐标格式不正确: {target}")

        if pre_delay > 0:
            cls.sleep_human(pre_delay, 0.2)

        controller.post_click(tx, ty).wait()

        if post_delay > 0:
            cls.sleep_human(post_delay, 0.25)

        if idle_chance > 0:
            cls.random_idle(chance=idle_chance)

        return tx, ty


__all__ = ["HumanSimulator"]
