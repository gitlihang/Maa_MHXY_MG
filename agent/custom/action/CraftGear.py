from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils import logger
from utils import LocalStorage

@AgentServer.custom_action("craftName")
class craftName(CustomAction):
    """
   打造装备
    """
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        #获取需要打造的装备名称
        image = context.tasker.controller.post_screencap().wait().get()
        reco_result = context.run_recognition(
            "经验链-寻物-装备名称",
            image
        )
        # 判断识别结果
        if not (reco_result and reco_result.hit):
            # 原实现在未命中时 DZ 未定义，会直接抛 NameError 让动作失败
            logger.error("未能识别到装备名称，跳过保存")
            return CustomAction.RunResult(success=True)

        # 保存需要打造的装备名字（带实例隔离）
        LocalStorage.set(
            "craftName", "craftName", reco_result.best_result.text, context=context
        )
        return CustomAction.RunResult(success=True)
    
@AgentServer.custom_action("CraftGear")
class CraftGear(CustomAction):
    """
   打造装备
    """
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        
        # 获取需要打造的装备名称（带实例隔离，取不到时给空串而不是 None）
        DZ = LocalStorage.get("craftName", "craftName", "", context=context)
        
        """
        根据需要打造的装备名称，进行分类划分，50级武器、60级武器、50级防具、60级防具并输出类型
        50级武器：剑：黄金剑、法杖：星云杖、枪：墨杆金钩、魔棒：幽路引魂、弓：玉腰弯弓、爪刺：玄冰刺、扇：劈水扇、飘带：云龙绸带、斧钺：黄金钺、环圈：赤炎环、锤：破甲战锤、鞭：青藤鞭、长刀：破天宝刀、双短剑：鱼骨双剑、降魔杵：金刚杵、弯刀：冷月弯刀、宝珠：蓬莱珠、云锦扇：蝉翼锦、牵星尺：
        """

        return CustomAction.RunResult(success=True)