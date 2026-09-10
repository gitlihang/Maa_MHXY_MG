# -*- coding: utf-8 -*-
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils import logger
from utils import LocalStorage

@AgentServer.custom_action("input_node_success_num")
class input_node_success_num(CustomAction):
    """
    统计节点成功次数（多实例已隔离）。

    计数通过 LocalStorage.increment 原子完成：整个"读取-加一-写回"过程持跨进程锁，
    多开时各实例并发自增也不会互相覆盖。

    实例隔离标识优先取 context.tasker.controller.uuid（每个模拟器各不相同），
    PI_CONTROLLER 拿不到 Adb 设备地址，仅作兜底，详见 utils.get_instance_id。
    """
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        # 原子自增并返回新值；首次运行不再是 None + 1
        num = LocalStorage.increment("node_success", "num", 1, context=context)
        if num is None:
            logger.error("节点成功次数写入失败，本次未计数")
        else:
            logger.info(f"节点成功次数: {num}")

        return CustomAction.RunResult(success=True)

@AgentServer.custom_action("output_node_success_num")
class output_node_success_num(CustomAction):
    """
   输出节点成功次数
    """
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        # 获取次数参数；取不到时按 0 输出，避免打印出 None
        num = LocalStorage.get("node_success", "num", 0, context=context)
        # 输出次数
        logger.info(f"运行次数: {num}")
        # 重置次数
        LocalStorage.set("node_success", "num", 0, context=context)

        return CustomAction.RunResult(success=True)
