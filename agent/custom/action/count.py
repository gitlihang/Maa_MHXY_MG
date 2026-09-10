from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
import json

# from assets.agent.utils.logger import logger
# from assets.agent.utils.utils import LocalStorage
from utils import logger
from utils import LocalStorage

@AgentServer.custom_action("count")
class count(CustomAction):
    """
    自定义计数器
    "custom_action_param": {
            "count": 0,
            "target_count": 4,
            "nextTask": "panduan_zhujiemian",
            "LoopNode": "背包向上滑动1"
        }
        count: 当前次数
        target_count: 目标次数
        nextTask: 完成目标次数后，执行下一任务节点
        LoopNode: 循环节点名称
    """
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        logger.info("进入count")

        argv_dict: dict = json.loads(argv.custom_action_param)
        if not argv_dict:
            return CustomAction.RunResult(success=True)
    
        # 获取自定义参数
        count = argv_dict.get("count", 0)
        target_count = argv_dict.get("target_count", 0)
        next_task = argv_dict.get("nextTask", "")
        loop_node = argv_dict.get("LoopNode", "")
        #当前次数小于目标次数时，增加计数并执行的任务
        while count < target_count:
            count = argv_dict.get("count", 0)
            # logger.info(f"当前计数: {count}, 目标计数: {target_count}")
            # 计数未达标时：递增计数并执行备用节点
            new_count = count + 1
            argv_dict["count"] = new_count
            # 保存更新后的参数
            context.override_pipeline(
                {argv.node_name: {"custom_action_param": argv_dict}}
            )
            context.run_task(loop_node)
            
        # 达到目标次数时，执行下一任务节点
        logger.info(f"计数已达标，执行下一任务: {next_task}")
        context.run_task(next_task)

        return CustomAction.RunResult(success=True)

@AgentServer.custom_action("countGlobal")
class countGlobal(CustomAction):
    """全局计数器
    "custom_action_param": {
            "target_count": 4,
            "nextTask": "panduan_zhujiemian",
            "LoopNode": "背包向上滑动1"
        }
        target_count: 目标次数，必须
        nextTask: 完成目标次数后，执行下一任务节点 ，必须
        LoopNode: 未达到目标次数，可执行节点，非必须
    """
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        logger.info("进入countGlobal")

        # 获取自定义参数
        argv_dict: dict = json.loads(argv.custom_action_param)
        if not argv_dict:
            return CustomAction.RunResult(success=True)
        # 获取全局本地参数

        # 原子自增（带实例隔离）：首次进入不会再拿 None 去比较而抛 TypeError，
        # 多实例并发也不会互相覆盖。返回的是自增“之后”的计数。
        count = LocalStorage.increment("task", "global_count", 1, context=context)
        if count is None:
            logger.error("全局计数写入失败，本次不计数")
            return CustomAction.RunResult(success=True)

        #获取传参
        target_count = argv_dict.get("target_count", 0)
        next_task = argv_dict.get("nextTask", "")
        LoopNode = argv_dict.get("LoopNode","")
        # count 已自增，故 <= target_count 等价于原来的“自增前 < target_count”
        if count <= target_count:
            logger.info(f"当前计数: {count}, 目标计数: {target_count}")
            #在本次循环需要执行的动作
            context.run_action(LoopNode)
            return CustomAction.RunResult(success=True)

        # 达到目标次数时，执行下一任务节点
        logger.info(f"计数已达标，执行下一任务: {next_task}")
        LocalStorage.set("task", "global_count", 0, context=context)
        context.run_task(next_task)

        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("countZG")
class countZG(CustomAction):
    """
    自定义计数器
    "custom_action_param": {
            "count": 0,
            "target_count": 4,
            "nextTask": "panduan_zhujiemian",
            "LoopNode": "背包向上滑动1"
        }
        count: 当前次数
        target_count: 目标次数
        nextTask: 完成目标次数后，执行下一任务节点
        LoopNode: 循环节点名称
    """
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        logger.info("进入countZG")

        argv_dict: dict = json.loads(argv.custom_action_param)
        if not argv_dict:
            return CustomAction.RunResult(success=True)
        
       # 获取传参节点target_count数据
        data1: dict = context.get_node_data("抓鬼轮数")['action']['param']['custom_action_param']['target_count']
        logger.info(f"抓鬼目标轮数: {data1}")
        target_count = int(data1)
        
        # 获取自定义参数
        count = argv_dict.get("count", 0)
        # target_count = argv_dict.get("target_count", 0)
        next_task = argv_dict.get("nextTask", "")
        loop_node = argv_dict.get("LoopNode", "")
        #当前次数小于目标次数时，增加计数并执行的任务
        while count < target_count:
            count = argv_dict.get("count", 0)
            logger.info(f"当前计数: {count}, 目标计数: {target_count}")
            # 计数未达标时：递增计数并执行备用节点
            new_count = count + 1
            argv_dict["count"] = new_count
            # 保存更新后的参数
            context.override_pipeline(
                {argv.node_name: {"custom_action_param": argv_dict}}
            )
            context.run_task(loop_node)
            return CustomAction.RunResult(success=True)
        # 达到目标次数时，执行下一任务节点
        logger.info(f"计数已达标，执行下一任务: {next_task}")
        context.run_task(next_task)
        
        return CustomAction.RunResult(success=True)
