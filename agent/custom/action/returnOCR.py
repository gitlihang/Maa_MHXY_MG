import json
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils import logger, HumanSimulator


@AgentServer.custom_action("returnOCR")
class ReturnOCR(CustomAction):
    """
    自定义返回ocr识别结果，根据action_key执行不同的动作
    "custom_action_param": {
            "action_key": "Click",
            "recognition_name": "识别输出测试",
            "return_text": "输出的描述"
            "click_target": []  # 点击坐标，格式为[x, y, w, h]，仅在action_key为Click时使用
        }
        action_key: 动作名称，用于判断动作类型，如Click、Move等
        recognition_name: task任务名称,用于指定识别任务名称，返回该节点的结果。
        return_text: 输出的描述，用于指定返回的描述
        click_target: 点击坐标，格式为[x, y, w, h]，仅在action_key为Click时使用。如果不提供click_target，则默认点击识别结果的拟人高斯位置。
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        # logger.info("进入returnOCR")
        # 解析自定义参数，并判断是否为空
        argv_dict: dict = json.loads(argv.custom_action_param)
        if not argv_dict:
            return CustomAction.RunResult(success=True)
        # 获取自定义参数
        action_key = argv_dict.get("action_key", "")
        recognition_name = argv_dict.get("recognition_name", "")
        return_text = argv_dict.get("return_text", "")
        click_target = argv_dict.get("click_target", [])

        # 获取ocr识别结果数据
        image = context.tasker.controller.post_screencap().wait().get()
        reco_result = context.run_recognition(
            recognition_name,
            image
        )
        # 打印OCR识别结果
        if reco_result and reco_result.hit:
            best_result = reco_result.best_result
            # 输出到ui界面
            logger.info(f"{return_text}: {best_result.text}")
            # 根据action_key执行不同的动作
            if action_key == "Click":
                # 点击传入参数中的坐标位置（高斯拟人落点）
                if click_target and len(click_target) >= 4:
                    target_x, target_y = HumanSimulator.get_gaussian_point(click_target)
                    logger.debug(f"拟人点击位置: ({target_x}, {target_y})")
                    HumanSimulator.sleep_human(0.2, 0.2)
                    context.tasker.controller.post_click(target_x, target_y).wait()
                    HumanSimulator.sleep_human(0.3, 0.2)
                # 点击最佳识别结果的拟人高斯位置
                elif best_result and best_result.box:
                    target_x, target_y = HumanSimulator.get_gaussian_point(best_result.box)
                    logger.debug(f"拟人点击位置: ({target_x}, {target_y})")
                    HumanSimulator.sleep_human(0.2, 0.2)
                    context.tasker.controller.post_click(target_x, target_y).wait()
                    HumanSimulator.sleep_human(0.3, 0.2)
                else:
                    logger.warning("没有识别到结果，无法执行点击")
            elif action_key == "":
                logger.debug(f"仅返回OCR数据，不执行动作: {action_key}")
        else:
            logger.warning(f"OCR识别失败 - 任务名称: {recognition_name}")
        
        return CustomAction.RunResult(success=True)