from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils import logger, HumanSimulator


@AgentServer.custom_action("sanjie")
class sanjie(CustomAction):
    """
    三界奇缘答题入口动作
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        logger.info("进入sanjie动作")
        tx, ty = HumanSimulator.get_point_offset(34, 137, radius=5.0)
        HumanSimulator.sleep_human(0.2, 0.2)
        context.tasker.controller.post_click(tx, ty).wait()
        HumanSimulator.sleep_human(0.4, 0.2)
        return CustomAction.RunResult(success=True)
