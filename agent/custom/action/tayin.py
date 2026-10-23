"""
MAA 自定义动作：拓印自动描摹（折线版）
- 使用固定 RGB 阈值提取笔画骨架
- 基于端点遍历排序轨迹点，并拆分为独立笔画
- 每个笔画用一次"连续折线滑动"(touch_down -> touch_move* -> touch_up) 执行，
  笔画内不抬手，仅笔画之间抬笔，避免逐条 post_swipe 造成的笔画断裂
- 限制笔画数 ≤ 10
"""

import time
import random
import math
import numpy as np
import cv2
from skimage.morphology import skeletonize

from maa.custom_action import CustomAction
from maa.context import Context
from utils import logger
from maa.agent.agent_server import AgentServer


@AgentServer.custom_action("tayin")
class tayin(CustomAction):
    """
    拓印自动描摹 梦幻西游拓印（固定RGB阈值，折线连续滑动）
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        logger.info("进入tayin（折线版）")

        # 1. 截图
        image_bgr = context.tasker.controller.post_screencap().wait().get()
        if image_bgr is None:
            logger.error("截图失败")
            return CustomAction.RunResult(success=False)

        # 转换为 RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 2. ROI 与颜色阈值
        roi = [515, 197, 320, 320]
        lower_color = [220, 172, 132]   # R, G, B 下界
        upper_color = [233, 203, 164]   # R, G, B 上界

        # 3. 提取轨迹：按笔画分组（每组为一个独立笔画，已按书写方向排好序）
        stroke_groups = self._extract_stroke_points(image_rgb, roi, lower_color, upper_color)
        if not stroke_groups:
            logger.warning("未检测到有效笔画")
            return CustomAction.RunResult(success=False)
        logger.info(f"提取到 {len(stroke_groups)} 个笔画")

        # 4. 生成每个笔画的折线关键点序列（每个笔画一次连续滑动，中间不抬手）
        max_strokes = 10
        strokes = self._generate_stroke_swipe_sequences(stroke_groups, max_strokes=max_strokes)
        if not strokes:
            logger.warning("无法生成滑动指令")
            return CustomAction.RunResult(success=False)
        logger.info(f"生成 {len(strokes)} 条笔画折线滑动")

        # 5. 执行滑动：笔画内不抬手，笔画间抬笔
        for idx, stroke in enumerate(strokes):
            if idx > 0:
                # 模拟"提笔换笔画"的停顿
                time.sleep(random.uniform(0.1, 0.3))
            # 每笔画总时长（毫秒），可自行调整
            duration = random.randint(600, 1000)
            self._execute_stroke_swipe(context, stroke, duration_ms=duration)

        logger.info("拓印描摹完成")
        return CustomAction.RunResult(success=True)

    # ------------------------------------------------------------------
    # 图像处理 / 轨迹排序 / 折线滑动生成
    # ------------------------------------------------------------------

    def _extract_stroke_points(self, image, roi, lower_color, upper_color):
        """
        从 ROI 中提取笔画骨架点，按连通分量拆分为独立笔画。
        返回：list[list[(x, y), ...]]，每个内层列表是一个笔画（全局坐标，已按书写方向排序）。
        """
        x, y, w, h = roi
        roi_img = image[y:y + h, x:x + w]

        # 阈值提取
        lower = np.array(lower_color, dtype=np.uint8)
        upper = np.array(upper_color, dtype=np.uint8)
        mask = cv2.inRange(roi_img, lower, upper)
        pixel_count = np.count_nonzero(mask)

        if pixel_count < 100:
            logger.warning(f"mask非零像素过少: {pixel_count}")
            return []

        # 闭运算填充断裂
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 骨架化
        skeleton_bool = skeletonize(closed > 0)
        skeleton = (skeleton_bool * 255).astype(np.uint8)

        ys, xs = np.where(skeleton > 0)
        if len(xs) == 0:
            return []

        points_roi = np.stack([xs, ys], axis=-1)

        # 拆分并排序笔画
        groups_roi = self._sort_strokes_to_groups(points_roi)

        # 还原全局坐标
        global_groups = []
        for group in groups_roi:
            global_groups.append([(int(px + x), int(py + y)) for px, py in group])
        return global_groups

    def _sort_strokes_to_groups(self, points: np.ndarray):
        """
        将骨架拆分为独立笔画（8邻域连通分量），每个笔画内部做端点遍历排序，
        并按笔画质心排序（先纵向、后横向），以贴近书写顺序。
        返回：list[list[(x, y)]]，每个内层列表是一个已排序笔画的像素点。
        """
        if len(points) == 0:
            return []

        points_list = points.tolist()
        points_set = set(map(tuple, points_list))

        # 构建8邻域邻接表
        neighbors = {}
        for p in points_list:
            nbs = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    q = (p[0] + dx, p[1] + dy)
                    if q in points_set:
                        nbs.append(q)
            neighbors[(p[0], p[1])] = nbs

        # 1. 拆分成独立的连通分量（不同笔画）
        visited_global = set()
        components = []
        for p in points_list:
            key = (p[0], p[1])
            if key in visited_global:
                continue
            comp = []
            stack = [key]
            visited_global.add(key)
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in neighbors[node]:
                    if nb not in visited_global:
                        visited_global.add(nb)
                        stack.append(nb)
            components.append(comp)

        # 2. 对每个连通分量内部进行端点DFS遍历
        ordered_components = []
        for comp in components:
            if not comp:
                continue
            # 找该分量的端点（邻居数为1）
            endpoints = [pt for pt in comp if len(neighbors[pt]) == 1]
            start = endpoints[0] if endpoints else comp[0]

            comp_visited = set()
            ordered_comp = []
            stack = [start]
            # 迭代DFS（防止递归爆栈）
            while stack:
                node = stack.pop()
                if node in comp_visited:
                    continue
                comp_visited.add(node)
                ordered_comp.append(node)
                for nb in reversed(neighbors[node]):
                    if nb not in comp_visited:
                        stack.append(nb)
            ordered_components.append(ordered_comp)

        # 3. 按质心排序（先 y 后 x），模拟从上到下、从左到右的书写顺序
        def centroid(comp):
            arr = np.array(comp, dtype=np.float64)
            return (arr[:, 1].mean(), arr[:, 0].mean())  # (y, x)

        ordered_components.sort(key=centroid)
        return ordered_components

    def _generate_stroke_swipe_sequences(self, stroke_groups, max_strokes=10):
        """
        对每个笔画做一个折线关键点序列（Douglas-Peucker 简化），
        供折线滑动使用；笔画数超过 max_strokes 时保留最长的笔画。
        返回：list[list[(x, y)]]。
        """
        max_points_per_stroke = 12
        # 过滤过短的碎笔画段，避免提笔过多、抖动
        min_points_per_stroke = 3

        result = []
        for stroke in stroke_groups:
            if len(stroke) < min_points_per_stroke:
                continue

            pts = np.array(stroke, dtype=np.float32)

            if len(pts) <= max_points_per_stroke + 1:
                sampled = pts.tolist()
            else:
                epsilon = 2.0
                while True:
                    pts_int = pts.reshape(-1, 1, 2).astype(np.int32)
                    approx = cv2.approxPolyDP(pts_int, epsilon, False)
                    simplified = approx.reshape(-1, 2).tolist()
                    if len(simplified) <= max_points_per_stroke:
                        sampled = simplified
                        break
                    epsilon *= 1.5
                    if epsilon > 50:
                        indices = np.linspace(0, len(pts) - 1, max_points_per_stroke, dtype=int)
                        sampled = pts[indices].tolist()
                        break

            result.append([(int(p[0]), int(p[1])) for p in sampled])

        # 按笔画长度（折线点数量 + 几何长度）降序，保留最重要的 max_strokes 条
        def stroke_len(s):
            total = 0.0
            for i in range(1, len(s)):
                total += math.hypot(s[i][0] - s[i - 1][0], s[i][1] - s[i - 1][1])
            return total

        result.sort(key=stroke_len, reverse=True)
        return result[:max_strokes]

    def _execute_stroke_swipe(self, context, stroke, duration_ms=500):
        """
        执行一个笔画的"连续折线滑动"：
        touch_down(起点) -> touch_move(途经点)... -> touch_up(抬起)，
        中间不抬手。
        各 touch_move 之间的间隔按总时长与线段长度比例分配，接近匀速描摹。
        """
        controller = context.tasker.controller
        if len(stroke) < 2:
            return

        # 计算每段长度与总长度
        segs = []
        for i in range(1, len(stroke)):
            dx = stroke[i][0] - stroke[i - 1][0]
            dy = stroke[i][1] - stroke[i - 1][1]
            segs.append(math.hypot(dx, dy))
        total_len = sum(segs)
        if total_len <= 0:
            return

        # 按下
        controller.post_touch_down(stroke[0][0], stroke[0][1]).wait()

        try:
            for i in range(1, len(stroke)):
                # 该段所占时长
                sleep_ms = duration_ms * (segs[i - 1] / total_len)
                time.sleep(sleep_ms / 1000.0)
                controller.post_touch_move(stroke[i][0], stroke[i][1]).wait()
        finally:
            # 抬起
            controller.post_touch_up().wait()


@AgentServer.custom_action("tayin_TZ")
class tayin_TZ(CustomAction):
    """
    拓印自动描摹 梦幻西游拓印（固定RGB阈值，滑动次数≤10）
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        
        context.tasker.post_stop().wait()
        logger.info("停止所有任务完成。")

        return CustomAction.RunResult(success=True)
