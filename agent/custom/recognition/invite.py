from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
from utils import logger, HumanSimulator


@AgentServer.custom_recognition("invite")
class invite(CustomRecognition):
    """
    邀请队员
    "attach": {
            "name": ["姓名1", "姓名2"]
        }
        name: 需要邀请的队员名字列表。
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        # 识别搜索位置并点击
        image = context.tasker.controller.post_screencap().wait().get()
        reco_result = context.run_recognition(
            "搜索",
            image,
            pipeline_override={
                "搜索": {
                    "roi": [336, 224, 64, 38],
                    "expected": ["搜索"],
                    "recognition": "OCR",
                }
            },
        )
        box = [0, 0, 0, 0]
        if reco_result and reco_result.all_results:
            for res in reco_result.all_results:
                box = res.box
        
        if box and len(box) >= 4 and box[2] > 0 and box[3] > 0:
            target_x, target_y = HumanSimulator.get_gaussian_point(box)
            logger.info(f"点击搜索位置: ({target_x}, {target_y})")
            HumanSimulator.sleep_human(0.3, 0.2)
            context.tasker.controller.post_click(target_x, target_y).wait()
            HumanSimulator.sleep_human(0.5, 0.2)

        return CustomRecognition.AnalyzeResult(box=(0, 0, 0, 0), detail="活跃度小于50,任务结束")
