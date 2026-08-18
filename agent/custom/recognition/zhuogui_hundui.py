from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
import json
import time
import math

from utils import logger, HumanSimulator

@AgentServer.custom_recognition("zhuogui_hundui")
class zhuogui_hundui(CustomRecognition):
    """
    整体任务结束
    多条件结束任务，
    满足以下条件之一，结束任务：
    1. 时间超过定时的时间
    Uset_time_HH = 23
    Uset_time_MM = 50
    2.活力为0或者活力低于用户指定值
    User_points=0
    """

    def analyze(
         self,
         context: Context,
         argv: CustomRecognition.AnalyzeArg,
     ) -> CustomRecognition.AnalyzeResult:
        # logger.info("zhuogui_hundui")
        
        Received_double_points = 0
        Not_Received_double_points = 0
        
        # 解析自定义参数，并判断是否为空
        argv_dict: dict = json.loads(argv.custom_action_param)
        if not argv_dict:
            return  CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="自定义参数为空")
        # 获取自定义参数
        User_points = argv_dict.get("User_points", 0)
        Uset_time_HH = argv_dict.get("Uset_time_HH", 23)
        Uset_time_MM = argv_dict.get("Uset_time_MM", 50)
        # 已领取双倍点数Received_double_points
        image1 = context.tasker.controller.post_screencap().wait().get()
        reco1 = context.run_recognition(
                        "已领取双倍点数",
                        image1,
                        pipeline_override={"已领取双倍点数": {"roi" : [628, 626, 53, 35],
                                                            "expected":[""],
                                                            "recognition": "OCR"
                                                            }
                                            }
                        )
        if not reco1 or not reco1.hit:
            return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="未识别到已领取双倍点数")
        Received_double_points= reco1.best_result.text
        # 未领取双倍点数Not_Received_double_points
        image1 = context.tasker.controller.post_screencap().wait().get()
        reco2 = context.run_recognition(
                        "未领取双倍点数",
                        image1,
                        pipeline_override={"未领取双倍点数": {"roi" : [910, 624, 63, 37],
                                                            "expected":[""],
                                                            "recognition": "OCR"
                                                            }
                                            }
                        )
        if not reco2 or not reco2.hit:
            return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="未识别到未领取双倍点数")
        Not_Received_double_points = reco2.best_result.text

        logger.info(f"已领取双倍点数: {Received_double_points}, 未领取双倍点数: {Not_Received_double_points}")
        # 获取当前时间，并判断是否超过定时的时间Uset_time_HH：Uset_time_MM
        current_time = time.localtime()
        current_hh = current_time.tm_hour
        current_mm = current_time.tm_min
        # 判断点击领取双倍次数
        m = 1000-int(Received_double_points)
        max_click = math.ceil(min(m, int(Not_Received_double_points)) / 100)

        # 判断是否满足结束条件
        if int(Received_double_points) + int(Not_Received_double_points) <= int(User_points): # 活力点数小于用户指定值
            logger.info("满足力点数小于用户指定值，任务结束")
            context.run_task("tuichuduiwu")
            return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="活力点数小于用户指定值，捉鬼任务结束")
        elif current_hh > Uset_time_HH or (current_hh == Uset_time_HH and current_mm > Uset_time_MM): # 时间超过定时的时间
            logger.info("满足时间超过定时的时间，任务结束")
            context.run_task("tuichuduiwu")
            return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="时间大于用户指定时间，捉鬼任务结束")
        else:
            # 根据点击次数，执行领取双倍点数的任务
            for _ in range(max_click):
                context.run_task("混队-抓鬼-双倍点数领取")
                HumanSimulator.sleep_human(1.0, 0.2)
            logger.info("未满足任务结束条件，领取双倍点数，并继续开始捉鬼混队")
            HumanSimulator.sleep_human(1.5, 0.25)
            # 拟人化挂机发呆机制
            HumanSimulator.random_idle(chance=0.1, min_sec=3.0, max_sec=8.0)
            context.run_task("tuichuduiwu")
            HumanSimulator.sleep_human(1.2, 0.2)
            context.run_task("zhuogui_hundui")
            return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="未满足结束条件，继续任务")


@AgentServer.custom_recognition("zhuogui_end_once")
class zhuogui_end_once(CustomRecognition):
    """
    捉鬼单次计算
    """
    def analyze(
             self,
             context: Context,
             argv: CustomRecognition.AnalyzeArg,
         ) -> CustomRecognition.AnalyzeResult:
        
        image = context.tasker.controller.post_screencap().wait().get()
        HumanSimulator.sleep_human(3.0, 0.2)
        image2 = context.tasker.controller.post_screencap().wait().get()
       

        return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="")