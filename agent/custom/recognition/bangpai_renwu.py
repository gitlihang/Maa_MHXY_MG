from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
from utils import logger

import json


@AgentServer.custom_recognition("bangpai_renwu_decide")
class BangpaiRenwuDecide(CustomRecognition):
    """帮派任务面板识别 + 黑名单放弃决策（替代原 ``帮派任务单次点击`` 节点的纯 OCR）。

    对任务追踪面板 roi 做一次 OCR，三选一：
    - 命中黑名单关键词（如「金香玉」）→ 先 ``run_task("bangpai_放弃任务")`` 放弃（终点回主界面），
      再 ``run_task("主界面-领取帮派任务")`` 重新领取，然后返回**未命中**（box=None）。JumpBack
      弹栈回到中心节点「已领取帮派任务」，其 next 循环（间隔/单次点击）接着处理新领到的任务；
    - 无黑名单且识别到帮派任务（青龙/白虎/朱雀/玄武）→ 返回该 OCR 框，交给节点
      ``action:"Click"`` 点击，继续 pipeline 内链路；
    - 其余 → 未命中。

    黑名单总开关：``custom_recognition_param.use_blacklist``（bool，默认 True）。
    置为 False 时不再命中黑名单，也就不会放弃/重新领取任务。
    """

    _ABANDON_ENTRY = "bangpai_放弃任务"
    _REACQUIRE_ENTRY = "主界面-领取帮派任务"   # 放弃后重新领取的链路入口（主界面→活动→参加→领取）
    _ACCEPT_KEYWORD = ["青龙", "白虎", "朱雀", "玄武"]
    _DEFAULT_ROI = [1034, 171, 235, 336]
    _DEFAULT_BLACKLIST = ["金香玉", "九转", "蛇胆酒", "长寿面", "珍露酒"]
    _DEFAULT_BLACKLIST_AFTER8 = ["蛇胆酒"]
    _DEFAULT_USE_BLACKLIST = True   # 黑名单功能总开关默认值（未传参时生效）

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        param = self._parse_param(argv.custom_recognition_param)
        roi = self._DEFAULT_ROI
        use_blacklist = self._as_bool(param.get("use_blacklist"), self._DEFAULT_USE_BLACKLIST)

        image = context.tasker.controller.post_screencap().wait().get()
        reco = context.run_recognition(
            "帮派任务面板",
            image,
            pipeline_override={
                "帮派任务面板": {
                    "roi": roi,
                    "expected": [""],
                    "recognition": "OCR",
                }
            },
        )

        if not reco or not reco.hit:
            # logger.info("[bangpai_decide] 面板无文字，未命中")
            return CustomRecognition.AnalyzeResult(box=None, detail="帮派面板无文字")

        full_text = "".join(r.text for r in reco.all_results)
        if not use_blacklist:
            # 黑名单总开关已关闭：跳过放弃/重领逻辑，只按任务名点击
            _black_list = []
        elif "8/10" in full_text or "9/10" in full_text or "10/10" in full_text:
            _black_list = self._DEFAULT_BLACKLIST_AFTER8
        else:
            _black_list = self._DEFAULT_BLACKLIST
        # logger.info(f"[bangpai_decide] 开启黑名单={use_blacklist} 生效黑名单={_black_list} 面板 OCR: {full_text}")

        # ① 黑名单优先：放弃 → 重新领取 → 回中心节点。
        #    本节点返回未命中（box=None），JumpBack 弹栈回到「已领取帮派任务」，
        #    其 next 循环（间隔/单次点击）会接着处理新领到的任务。
        hit_black = next((kw for kw in _black_list if kw and kw in full_text), None)
        if hit_black:
            logger.info(f"[bangpai_decide] 命中黑名单「{hit_black}」，放弃后重新领取")
            context.run_task(self._ABANDON_ENTRY)     # 放弃，终点回主界面
            context.run_task(self._REACQUIRE_ENTRY)   # 主界面→活动→参加→领取帮派任务
            return CustomRecognition.AnalyzeResult(box=None, detail=f"黑名单:{hit_black},已放弃并重新领取")

        # ② 无黑名单且识别到帮派任务（四堂名）：返回其框，交给节点 action:Click
        for res in reco.all_results:
            if any(_key in res.text for _key in self._ACCEPT_KEYWORD):
                # logger.info(f"[bangpai_decide] 识别到帮派任务，返回框 {res.box}")
                return CustomRecognition.AnalyzeResult(box=res.box, detail="点击帮派任务")

        # ③ 都没有：未命中
        # logger.info("[bangpai_decide] 未识别到帮派任务，未命中")
        return CustomRecognition.AnalyzeResult(box=None, detail="未识别到帮派任务")

    @staticmethod
    def _parse_param(raw_param) -> dict:
        """解析 custom_recognition_param，兼容 dict / JSON 字符串两种写法。"""
        if isinstance(raw_param, dict):
            return raw_param
        if isinstance(raw_param, str) and raw_param.strip():
            try:
                data = json.loads(raw_param)
            except json.JSONDecodeError:
                logger.warning(f"[bangpai_decide] custom_recognition_param 解析失败: {raw_param}")
                return {}
            return data if isinstance(data, dict) else {}
        return {}

    @staticmethod
    def _as_bool(value, default: bool = True) -> bool:
        """把 pipeline 传入的开关值宽松地转成 bool，无法识别时回退 default。"""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("true", "1", "yes", "y", "on", "开启", "是"):
                return True
            if text in ("false", "0", "no", "n", "off", "关闭", "否"):
                return False
        return default
