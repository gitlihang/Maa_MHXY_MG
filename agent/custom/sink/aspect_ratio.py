"""
分辨率检查器

在任务开始时检查模拟器分辨率是否为 16:9（仅横屏），如果不是则停止任务并输出警告。
"""

from maa.agent.agent_server import AgentServer
from maa.tasker import Tasker, TaskerEventSink
from maa.event_sink import NotificationType

from utils.logger import logger

# 目标宽高比：16:9
TARGET_RATIO = 16.0 / 9.0
# 容差范围（±2%）
TOLERANCE = 0.02


def is_aspect_ratio_16x9(width: int, height: int) -> bool:
    """
    检查给定的尺寸是否为横屏（宽 > 高）且宽高比约为 16:9
    竖屏（9:16）或正方形将直接返回 False
    """
    if width <= 0 or height <= 0:
        return False

    # 必须是横屏
    if width <= height:
        return False

    # 计算横屏下的宽高比
    ratio = width / float(height)

    # 检查比例是否在 16:9 的容差范围内
    return abs(ratio - TARGET_RATIO) <= TARGET_RATIO * TOLERANCE


def calculate_aspect_ratio(width: int, height: int) -> float:
    """
    计算宽高比，返回宽度/高度（不进行方向归一化）
    用于日志输出实际比例
    """
    return width / float(height)


def _query_device_size(controller) -> str | None:
    """
    通过 adb shell 查询设备端真实分辨率（wm size），仅用于诊断日志。

    MaaFramework 会把原始截图等比缩放到默认短边 720 后再由 cached_image 返回，
    因此截图尺寸与设备原始分辨率可能不一致，需要额外查询设备端信息辅助定位。
    """
    try:
        controller.post_shell("wm size").wait()
        output = controller.shell_output
    except Exception as e:
        logger.debug(f"查询设备 wm size 失败: {e}")
        return None

    if not output:
        return None

    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    return " | ".join(lines)


@AgentServer.tasker_sink()
class AspectRatioChecker(TaskerEventSink):
    """
    分辨率检查器
    在任务开始时检查设备分辨率是否为 16:9（横屏）
    """

    def __init__(self):
        self._checked = False

    def on_tasker_task(
        self,
        tasker: Tasker,
        noti_type: NotificationType,
        detail: TaskerEventSink.TaskerTaskDetail,
    ):
        # 只在任务开始时检查
        if noti_type != NotificationType.Starting:
            return

        # 忽略停止任务事件
        if detail.entry == "MaaTaskerPostStop":
            logger.debug("收到 PostStop 事件，跳过分辨率检查")
            return

        logger.debug(
            f"任务开始前检查分辨率 - task_id: {detail.task_id}, entry: {detail.entry}"
        )

        # 获取控制器
        controller = tasker.controller
        if controller is None:
            logger.error("无法获取控制器")
            return

        # 获取缓存的图像
        try:
            img = controller.cached_image
            if img is None:
                # 如果没有缓存图像，尝试截图
                img = controller.post_screencap().wait().get()
        except Exception as e:
            logger.error(f"无法获取截图: {e}")
            return

        if img is None:
            logger.error("无法获取截图")
            return

        # 获取图像尺寸
        height, width = img.shape[:2]

        logger.debug(f"截图尺寸: {width} x {height}")

        # 检查宽高比（仅横屏 16:9）
        if not is_aspect_ratio_16x9(width, height):
            actual_ratio = calculate_aspect_ratio(width, height)

            # MaaFramework 默认按短边 720 等比缩放截图（ScreenshotTargetShortSide = 720），
            # cached_image 返回的是缩放后的识别图，所以这里额外查询设备真实分辨率辅助定位。
            device_size = _query_device_size(controller)
            extra = (
                "（提示：该尺寸是 MaaFramework 按默认短边 720 等比缩放后的识别图尺寸，"
                "比例与设备画面一致，并不等于设备物理分辨率）"
            )
            if device_size:
                extra += f"（设备 wm size: {device_size}）"

            if width <= height:
                logger.error(
                    f"🚨 当前设备为竖屏或正方形模式 ({width}x{height})，Maa_MHXY_MG 仅支持横屏 16:9 比例，请旋转模拟器或调整分辨率。{extra}"
                )
            else:
                logger.error(
                    f"🚨 分辨率比例不匹配！任务已停止。"
                    f"当前: {width}x{height} (比例: {actual_ratio:.4f})，"
                    f"Maa_MHXY_MG 仅支持 16:9 比例，请调整为: 2560x1440, 1920x1080, 1600x900, 1280x720(推荐){extra}"
                )

            # 停止任务
            tasker.post_stop()
        else:
            logger.debug(f"分辨率检查通过: {width}x{height} (16:9 横屏)")