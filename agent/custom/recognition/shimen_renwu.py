from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
import json
import random
import time

from utils import logger, HumanSimulator


@AgentServer.custom_recognition("shimen_renwu_decide")
class ShimenRenwuDecide(CustomRecognition):
    """师门任务单次点击的识别决策。

    在 agent 侧做"点哪个 / 是否点 / 是否触发子链路"的决策，对师门任务追踪面板做一次 OCR，拼出 ``full_text`` 后：

    - **命中打造类任务**：``full_text`` 里出现 ``_50_MAP`` / ``_60_MAP`` 的某个装备名（value）→
      解析出对应的 (等级, 品类, 装备名)，``override_pipeline`` 改写 dazao 链路的三个 OCR 节点
      （选等级 / 选品类 / 装备名判定），``run_task("dazao")`` 启动打造，最后返回未识别（``box=None``）。
    - **非打造类师门任务**：直接在 agent 内 ``post_click`` 点击识别框中心，等 7 秒（替代原节点
      ``post_delay:7000``），再 ``run_task`` 运行一次「抄本兜底」节点清除点击后出现的「使用抄本」按钮，
      最后返回未识别（``box=None``）。**本识别一律不返回命中信号**——点击副作用由 agent 侧完成.
    - **都没有**：返回 ``box=None``（未命中）；连续 ``_MISS_STREAK_LIMIT`` 次未命中则 ``run_task``
      一次 ``panduan_zhujiemian`` 回主界面（兜底重置屏幕状态）再归零计数。任意命中分支（①②）归零计数。

    MAP 形如 ``{"男衣": "夜魔披风", ...}``：key=品类（dazao 左侧列表项），value=装备名（右侧展示）。
    合并两个 MAP 反查时，同名装备 60 级优先（``setdefault`` 保留先插入者）。

    OCR 的 roi 起点 (x,y) 每次 ``analyze`` 随机波动 ±``_ROI_JITTER``（``_jittered_roi``），打破固定
    roi 下 OCR 持续 badcase 导致的决策死循环。
    """

    _RECO_NAME = "师门任务-单次点击-OCR"
    _DEFAULT_ROI = [1036, 113, 240, 180]
    _ROI_JITTER = 5  # OCR 起点 (x,y) 每次随机波动 ±N，打破固定 ROI 的 OCR badcase 死循环
    _DAZAO_ENTRY = "dazao"
    # override 目标：dazao 链路里「选等级 / 选品类 / 装备名判定」三个 OCR 节点
    _DAZAO_LEVEL_NODE = "打造-切换等级-选择目标等级"
    _DAZAO_CATEGORY_NODE = "打造-切换装备-选择目标装备"
    _DAZAO_NAME_NODE = "打造-打造装备-收集材料"
    # 点击师门任务条目后，面板出现「使用抄本」按钮 —— 用兜底节点清除
    _CHAOBEN_FALLBACK_NODE = "师门任务-任务分支-抄本兜底-入口"
    _CHAOBEN_FALLBACK_TIMEOUT = 1000  # 兜底识别上限，防非抄本任务时阻塞状态机
    # 连续 N 次未识别到师门任务 → 飞回长安一次，打破「面板 OCR 持续空/误读」卡死
    _MISS_STREAK_LIMIT = 5
    _BACK_TO_MAIN_ENTRY = "打开大地图_69副本"

    _50_MAP = {
        # 防具
        "鞋": "绿靴", "腰带": "乱牙咬", "项链": "荧光坠子", "发钗": "媚狐头饰",
        "头盔": "羊角盔", "女衣": "金缕羽衣", "男衣": "钢甲",
        # 武器
        "长戈": "三星戈", "牵星尺": "星帆尺", "云锦扇": "蝉翼锦", "宝珠": "蓬莱珠",
        "弯刀": "冷月弯刀", "降魔杵": "金刚杵", "双短剑": "鱼骨双剑", "长刀": "破天宝刀",
        "鞭": "青藤鞭", "锤": "破甲战锤", "环圈": "赤炎环", "斧钺": "黄金钺",
        "飘带": "云龙绸带", "扇": "劈水扇", "爪刺": "玄冰刺", "弓": "玉腰弯弓",
        "魔棒": "幽路引魂", "枪": "墨杆金钩", "法杖": "星云杖", "剑": "黄金剑",
    }
    _60_MAP = {
        # 防具
        "鞋": "追星踏月", "腰带": "双魂引", "项链": "风月宝链", "发钗": "玉女发冠",
        "头盔": "水晶帽", "女衣": "霓裳羽衣", "男衣": "夜魔披风",
        # 武器
        "长戈": "天山辰律", "牵星尺": "北落师门", "云锦扇": "桃之夭夭", "宝珠": "金露函烟",
        "弯刀": "埃兰弯刀", "降魔杵": "摩星杵", "双短剑": "落星双剑", "长刀": "秋水刀",
        "鞭": "雪绒鞭", "锤": "震天锤", "环圈": "蛇形月", "斧钺": "乌金鬼头镰",
        "飘带": "七彩罗刹", "扇": "清秋扇", "爪刺": "青刚刺", "魔棒": "满天星",
        "枪": "玄铁矛", "法杖": "天山雪", "剑": "游龙剑", "弓": "连珠神弓",
    }

    def _jittered_roi(self) -> list:
        """对 ``_DEFAULT_ROI`` 的起点 (x,y) 各加 ±``_ROI_JITTER`` 随机扰动，宽高不变。

        每次 ``analyze`` 用不同的 roi 去截图裁剪做 OCR，扰动喂给 OCR 的像素，避免某个固定 roi
        下 OCR 持续 badcase（如把装备名/「师门」稳定误识别）导致 agent 每轮做同样的错误决策、卡死。
        """
        x, y, w, h = self._DEFAULT_ROI
        dx = random.randint(-self._ROI_JITTER, self._ROI_JITTER)
        dy = random.randint(-self._ROI_JITTER, self._ROI_JITTER)
        return [x + dx, y + dy, w + dx, h + dy]

    def _reset_miss_streak(self) -> None:
        """任意命中分支（打造/师门点击）调用，归零连续未命中计数。"""
        self._miss_streak = 0

    def _on_miss(self, context: Context) -> CustomRecognition.AnalyzeResult:
        """记录一次「未识别到师门任务」。

        连续达到 ``_MISS_STREAK_LIMIT`` 次时，``run_task`` 执行一次 ``panduan_zhujiemian`` 回主界面
        （兜底重置屏幕状态），再归零计数；否则只递增并返回未命中。
        """
        streak = getattr(self, "_miss_streak", 0) + 1
        self._miss_streak = streak
        if streak >= self._MISS_STREAK_LIMIT:
            logger.info(f"[shimen_decide] 连续 {streak} 次未识别到师门任务，回主界面")
            context.run_task(self._BACK_TO_MAIN_ENTRY)
            self._miss_streak = 0
        else:
            logger.info(f"[shimen_decide] 未识别到师门任务（连续 {streak}/{self._MISS_STREAK_LIMIT}）")
        return CustomRecognition.AnalyzeResult(box=None, detail="未识别到师门任务")

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        param: dict = json.loads(argv.custom_recognition_param or "{}")

        roi_used = self._jittered_roi()
        image = context.tasker.controller.post_screencap().wait().get()
        reco = context.run_recognition(
            self._RECO_NAME,
            image,
            pipeline_override={
                self._RECO_NAME: {
                    "roi": roi_used,
                    "expected": [""],
                    "recognition": "OCR",
                    "threshold": 0.7
                }
            },
        )

        if not reco or not reco.hit or not reco.all_results:
            return self._on_miss(context)

        # 按 y 升序（从上到下）拼接，y 相同再按 x 升序（从左到右）——固定阅读顺序
        results = sorted(reco.all_results,
                         key=lambda r: ((r.box[1] if r.box else 0), (r.box[0] if r.box else 0)))
        full_text = "".join(r.text for r in results)
        logger.info(f"[shimen_decide] 面板 OCR (roi={roi_used}): {full_text}")

        # 合并两个等级 MAP：装备名 → (等级, 品类)。
        name_to_target = {}
        for cat, name in self._60_MAP.items():
            name_to_target.setdefault(name, ("60", cat))
        for cat, name in self._50_MAP.items():
            name_to_target.setdefault(name, ("50", cat))

        # ① 命中打造类任务：full_text 含某装备名 → override dazao 三节点 + run_task dazao
        hit_name = next((n for n in name_to_target if n in full_text), None)
        not_own = "拥有0" in full_text or "0/1" in full_text
        if hit_name and not_own:
            level, category = name_to_target[hit_name]
            logger.info(f"[shimen_decide] 命中打造任务: {level}级 {category} -> {hit_name}")
            context.override_pipeline({
                self._DAZAO_LEVEL_NODE: {"expected": [level]},
                self._DAZAO_CATEGORY_NODE: {"expected": [category]},
                self._DAZAO_NAME_NODE: {"expected": [hit_name]},
            })
            context.run_task(self._DAZAO_ENTRY)

            self._reset_miss_streak()
            return CustomRecognition.AnalyzeResult(box=None, detail=f"打造任务:{hit_name} 完成")

        # ② 非打造类师门任务：agent 内直接下发拟人化点击，返回未识别（box=None）
        for res in results:
            if "师门" in res.text:
                target_x, target_y = HumanSimulator.get_gaussian_point(res.box)
                logger.info(f"[shimen_decide] 识别到师门任务，拟人化点击 ({target_x}, {target_y})")
                HumanSimulator.sleep_human(0.3, 0.2)
                context.tasker.controller.post_click(target_x, target_y).wait()
                # 拟人化延迟等待任务面板跳转（原 post_delay:7000，加入正态波动）
                HumanSimulator.sleep_human(7.0, 0.15)
                HumanSimulator.random_idle(chance=0.06, min_sec=2.0, max_sec=5.0)

                # 点击后面板可能出现「使用抄本」按钮 → 运行一次兜底节点清除。
                # run_task 以该节点为 entry 同步执行；节点 next 为空，命中即点一次即结束。
                # 用 override 把 timeout 收窄到 1s，避免非抄本任务时白等 20s 阻塞状态机。
                context.run_task(
                    self._CHAOBEN_FALLBACK_NODE,
                    pipeline_override={
                        self._CHAOBEN_FALLBACK_NODE: {"timeout": self._CHAOBEN_FALLBACK_TIMEOUT},
                    },
                )
                self._reset_miss_streak()
                return CustomRecognition.AnalyzeResult(box=None, detail="已点击师门任务")

        # ③ 都没有：未命中 → 计数，连续 _MISS_STREAK_LIMIT 次则回主界面打破卡死
        return self._on_miss(context)
