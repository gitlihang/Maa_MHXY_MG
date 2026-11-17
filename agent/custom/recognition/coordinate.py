from maa.agent.agent_server import AgentServer
from maa.custom_recognition  import CustomRecognition
from maa.context import Context
from utils import logger
import re

from utils import PositionTracker


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@AgentServer.custom_recognition("coordinate")
class coordinate(CustomRecognition):
    """
    坐标稳定性识别（custom_recognition: "coordinate"）

    attach.mode:
        duration（默认）—— 当前坐标连续不变 sec 秒即通过，不要求次数
        count           —— 当前坐标连续相同 min_count 次即通过
        window          —— 旧逻辑：sec 秒内至少 min_count 次且全部相同

    在 pipeline 节点中调用时，"attach" 的写法如下（字段均可选，不填则用默认值）：
    "attach": {
        "mode": "duration",   # 判定模式：duration（默认）/ count / window，见下方 analyze 内注释
        "sec": 10,            # duration/window 模式：坐标连续不变多少秒算稳定，默认 5.0
        "min_count": 6,       # count/window 模式：坐标连续相同多少次算稳定，默认 2
        "max_gap": 5.0        # 相邻两次坐标记录允许的最大间隔秒数，默认 5.0
    }

    完整节点示例（见 assets/resource/base/pipeline/my_task.json 的 "坐标" 节点）：
    "坐标": {
        "recognition": "Custom",
        "custom_recognition": "coordinate",
        "attach": {
            "mode": "duration",
            "sec": 10,
            "min_count": 6,
            "max_gap": 5.0
        }
    }
    """
    def analyze(
         self,
         context: Context,
         argv: CustomRecognition.AnalyzeArg,
     ) -> CustomRecognition.AnalyzeResult:
        # 识别地图名
        # logger.info(f"[coordinate] 开始识别地图名和坐标")
        image = context.tasker.controller.post_screencap().wait().get()
        reco_result_dtm = context.run_recognition(
            "地图名",
            image,
            pipeline_override={
                "地图名":{"roi" : [109, 23, 121, 33],
                        "expected":[""],
                        "recognition": "OCR"
                       }
                }
        )
        if not reco_result_dtm or not reco_result_dtm.hit:
            # logger.info("[bangpai_decide] 面板无文字，未命中")
            return CustomRecognition.AnalyzeResult(box=None, detail="未识别到地图名")
        # 识别坐标
        reco_result_zb = context.run_recognition(
                    "坐标",
                    image,
                    pipeline_override={
                        "坐标":{"roi" : [129, 58, 99, 26],
                                "expected":[""],
                                "recognition": "OCR"
                               }
                        }
                )
        if not reco_result_zb or not reco_result_zb.hit:
            # logger.info("[bangpai_decide] 面板无文字，未命中")
            return CustomRecognition.AnalyzeResult(box=None, detail="未识别到坐标")
        # 地图名， 例如：长安城
        dtm = reco_result_dtm.best_result.text
        zb  = reco_result_zb.best_result.text
        # logger.info(f"识别结果: 地图名: {dtm}, 坐标: {zb}")
        # 去除两边括号
        zb = re.match(r'^[（(](.+)[）)]$', zb)
        # 用逗号替换点 151.71→151，71
        if zb:
            zb = re.sub(r'\.', ',', zb.group(1))

        # 记录坐标
        PositionTracker.record(dtm, zb, context=context)

        # attach.mode:
        #   duration（默认）—— 当前坐标连续不变 sec 秒即通过，不要求次数
        #   count           —— 当前坐标连续相同 min_count 次即通过
        #   window          —— 旧逻辑：sec 秒内至少 min_count 次且全部相同
        node_data = context.get_node_data(argv.node_name) or {}
        attach = node_data.get("attach") or {}

        mode = str(attach.get("mode", "duration")).strip().lower()
        sec = _as_float(attach.get("sec"), 5.0)
        min_count = attach.get("min_count", 2)
        max_gap = _as_float(attach.get("max_gap"), 5.0)

        if PositionTracker.is_stable(
            mode=mode,
            sec=sec,
            min_count=min_count,
            max_gap=max_gap,
            context=context,
        ):
            logger.debug(
                f"[{argv.node_name}] {mode} 判定坐标稳定: "
                f"sec={sec}, min_count={min_count}, max_gap={max_gap}"
            )
            return CustomRecognition.AnalyzeResult(box=(0, 0, 0, 0), detail="")

        # 返回 None 让 custom_recognition 失败，触发节点失败逻辑
        return None
        



