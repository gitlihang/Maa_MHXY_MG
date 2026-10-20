from maa.agent.agent_server import AgentServer
from maa.custom_recognition  import CustomRecognition
from maa.context import Context
from utils import logger
import re

from utils import PositionTracker

@AgentServer.custom_recognition("coordinate")
class coordinate(CustomRecognition):
    """

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
        # 测试 10 秒内值是否一致
        # r = PositionTracker.compare(10, context=context)
        # if r["has_data"]:
        #     if r["all_same"]:
        #         logger.info(f"10 秒内值一致: {r['values'][0]}")
        #     else:
        #         logger.info(f"10 秒内值变过，共 {r['count']} 条，出现过的值: {r['distinct']}")
        
        sec: dict = context.get_node_data(argv.node_name)["attach"]["sec"]
        min_count: dict = context.get_node_data(argv.node_name)["attach"]["min_count"]

        if PositionTracker.same_within(sec, context=context, min_count=min_count):
            logger.debug(f"在{argv.node_name}任务的{sec}秒内{min_count}次坐标值一致，继续执行next任务")
            return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="")
           
        
        # 返回假结果使custom_recognition失败，触发节点失败逻辑
        # logger.info(f"10秒内坐标值不一致")
        # return CustomRecognition.AnalyzeResult(success=False)
        return None
        



