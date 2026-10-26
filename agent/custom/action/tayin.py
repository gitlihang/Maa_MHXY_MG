"""
MAA 自定义动作：拓印自动描摹（折线版）
- 使用固定 RGB 阈值提取笔画骨架
- 基于端点遍历排序轨迹点，并拆分为独立笔画
- 每个笔画用一次"连续折线滑动"(touch_down -> touch_move* -> touch_up) 执行，
  笔画内不抬手，仅笔画之间抬笔，避免逐条 post_swipe 造成的笔画断裂
- 限制笔画数 ≤ 10

Android 适配：cv2 / scikit-image 在 Android (CPython 3.13) 上无可用 wheel，
以下图像处理全部改用纯 numpy 实现（颜色转换 / inRange / 3x3 闭运算 /
Zhang-Suen 骨架化 / Douglas-Peucker 折线简化），行为与桌面端保持一致。
"""

import time
import random
import math
import numpy as np

from maa.custom_action import CustomAction
from maa.context import Context
from utils import logger
from maa.agent.agent_server import AgentServer


# ------------------------------------------------------------------
# 纯 numpy 图像处理工具（替代 cv2 / skimage 在桌面端承担的工作）
# ------------------------------------------------------------------

def _bgr_to_rgb(image_bgr):
    """等价 cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)"""
    return image_bgr[..., ::-1]


def _in_range(roi_img, lower, upper):
    """等价 cv2.inRange(roi_img, lower, upper)，返回 uint8 0/255 mask"""
    return np.all((roi_img >= lower) & (roi_img <= upper), axis=-1).astype(np.uint8) * 255


def _binary_dilate3(img):
    """3x3 全 1 结构元二值膨胀（等价 cv2.morphologyEx MORPH_DILATE, 3x3, 边界补 0）"""
    padded = np.pad(img, 1, mode="constant", constant_values=0)
    h, w = img.shape
    out = np.zeros_like(img)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy:dy + h, dx:dx + w]
    return out


def _binary_erode3(img):
    """3x3 全 1 结构元二值腐蚀（等价 cv2.morphologyEx MORPH_ERODE, 3x3, 边界补 255）"""
    padded = np.pad(img, 1, mode="constant", constant_values=255)
    h, w = img.shape
    out = np.full_like(img, 255)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy:dy + h, dx:dx + w]
    return out


def _morph_close3(mask):
    """等价 cv2.morphologyEx(mask, cv2.MORPH_CLOSE, 3x3, iterations=1)"""
    return _binary_erode3(_binary_dilate3(mask))


def _zhang_suen(binary):
    """
    Zhang-Suen 细化，等价 skimage.morphology.skeletonize(..., method='zhang')。
    binary: bool 数组，返回 bool 骨架。
    """
    img = binary.astype(bool)
    changed = True
    while changed:
        changed = False
        p = np.pad(img, 1, mode="constant", constant_values=False)
        p2 = p[0:-2, 1:-1]
        p3 = p[0:-2, 2:]
        p4 = p[1:-1, 2:]
        p5 = p[2:, 2:]
        p6 = p[2:, 1:-1]
        p7 = p[2:, 0:-2]
        p8 = p[1:-1, 0:-2]
        p9 = p[0:-2, 0:-2]

        B = (p2.astype(int) + p3.astype(int) + p4.astype(int) + p5.astype(int)
             + p6.astype(int) + p7.astype(int) + p8.astype(int) + p9.astype(int))
        A = (((~p2) & p3).astype(int) + ((~p3) & p4).astype(int)
             + ((~p4) & p5).astype(int) + ((~p5) & p6).astype(int)
             + ((~p6) & p7).astype(int) + ((~p7) & p8).astype(int)
             + ((~p8) & p9).astype(int) + ((~p9) & p2).astype(int))

        # Step 1
        c1 = (~p2) | (~p4) | (~p6)
        c2 = (~p4) | (~p6) | (~p8)
        m1 = img & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
        if m1.any():
            img[m1] = False
            changed = True

        # Step 2（基于 Step 1 更新后的图重新取邻域）
        p = np.pad(img, 1, mode="constant", constant_values=False)
        p2 = p[0:-2, 1:-1]
        p3 = p[0:-2, 2:]
        p4 = p[1:-1, 2:]
        p5 = p[2:, 2:]
        p6 = p[2:, 1:-1]
        p7 = p[2:, 0:-2]
        p8 = p[1:-1, 0:-2]
        p9 = p[0:-2, 0:-2]

        B = (p2.astype(int) + p3.astype(int) + p4.astype(int) + p5.astype(int)
             + p6.astype(int) + p7.astype(int) + p8.astype(int) + p9.astype(int))
        A = (((~p2) & p3).astype(int) + ((~p3) & p4).astype(int)
             + ((~p4) & p5).astype(int) + ((~p5) & p6).astype(int)
             + ((~p6) & p7).astype(int) + ((~p7) & p8).astype(int)
             + ((~p8) & p9).astype(int) + ((~p9) & p2).astype(int))

        c1 = (~p2) | (~p4) | (~p8)
        c2 = (~p2) | (~p6) | (~p8)
        m2 = img & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
        if m2.any():
            img[m2] = False
            changed = True

    return img


def _approx_poly_dp(points, epsilon):
    """
    Douglas-Peucker 折线简化，等价 cv2.approxPolyDP(points, epsilon, closed=False)。
    points: Nx2 数组（可为 float/int），返回简化后的 [(x, y), ...] 列表。
    """
    pts = [(float(x), float(y)) for x, y in points]
    n = len(pts)
    if n <= 2:
        return pts

    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        x0, y0 = pts[i0]
        x1, y1 = pts[i1]
        dx = x1 - x0
        dy = y1 - y0
        seg_len2 = dx * dx + dy * dy
        max_dist = -1.0
        max_idx = -1
        for i in range(i0 + 1, i1):
            x, y = pts[i]
            if seg_len2 == 0:
                dist = math.hypot(x - x0, y - y0)
            else:
                t = ((x - x0) * dx + (y - y0) * dy) / seg_len2
                t = max(0.0, min(1.0, t))
                px = x0 + t * dx
                py = y0 + t * dy
                dist = math.hypot(x - px, y - py)
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        if max_dist > epsilon and max_idx != -1:
            keep.add(max_idx)
            stack.append((i0, max_idx))
            stack.append((max_idx, i1))

    return [pts[i] for i in sorted(keep)]


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
        image_rgb = _bgr_to_rgb(image_bgr)

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
        mask = _in_range(roi_img, lower, upper)
        pixel_count = np.count_nonzero(mask)

        if pixel_count < 100:
            logger.warning(f"mask非零像素过少: {pixel_count}")
            return []

        # 闭运算填充断裂
        closed = _morph_close3(mask)

        # 骨架化
        skeleton_bool = _zhang_suen(closed > 0)
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
                    simplified = _approx_poly_dp(pts, epsilon)
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
        logger.info("遇见拓印，停止任务。")

        context.tasker.post_stop().wait()

        return CustomAction.RunResult(success=True)
