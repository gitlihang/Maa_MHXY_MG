# -*- coding: utf-8 -*-
"""
MAA 自定义动作：梦幻小蜜 - 读取金币并推送到钉钉
- 打开包裹后，对金币矩阵区域做 OCR，把识别到的数字"按顺序组合、不删减"取出
- 可选的：将金币矩阵区域截图保存到本地（作为推送备选方案，UI 里可开关）
- 通过钉钉机器人 Webhook 推送"任务完成 + 金币数量"，支持"加签(secret)"（UI 里填了 secret 就加签，空着则不加）

custom_action_param 字段：
    recognition_name : OCR 识别节点名（默认 梦幻小蜜-金币识别）
    roi              : 金币矩阵区域 [左上x, 左上y, 宽, 高]（默认 [177, 633, 174, 45]）
    webhook_url      : 钉钉机器人 Webhook 地址
    secret           : 钉钉机器人加签密钥（没有则留空，跳过加签）
    role             : 角色（可在 UI 自定义输入）
    message_template : 消息模板，支持 {time} {role} {gold} 三个占位符
    msgtype          : 消息类型 text / markdown（默认 text）
    screenshot       : 是否把金币矩阵区域截图保存到本地（默认 false）
    screenshot_dir   : 截图保存目录（相对项目根，默认 ./debug/custom/mengghuan_xiaomi）
"""

import json
import os
import re
import time
import base64
import hmac
import hashlib
import struct
import zlib
import urllib.parse
import urllib.request

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils import logger

# requests 在项目/Android 环境已内置，这里做一次兜底，没有 requests 时退回 urllib
try:
    import requests as _requests
except ImportError:
    _requests = None


@AgentServer.custom_action("mengghuan_xiaomi_push")
class MengghuanXiaomiPush(CustomAction):
    """梦幻小蜜：读取金币并推送钉钉"""

    CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mengghuan_config.json"
    )

    # ------------------------------------------------------------------
    # 读取备用配置文件（UI 注入优先，文件兜底，保证推送一定可用）
    # ------------------------------------------------------------------
    @classmethod
    def _load_config(cls):
        cfg = {}
        try:
            if os.path.exists(cls.CONFIG_PATH):
                with open(cls.CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if not isinstance(cfg, dict):
                    cfg = {}
        except Exception as e:
            logger.warning(f"读取 mengghuan_config.json 失败: {e}")
        return cfg

    # ------------------------------------------------------------------
    # 纯 numpy + zlib 的 PNG 编码（Android 上没有 PIL 时兜底，保证截图可用）
    # ------------------------------------------------------------------
    @staticmethod
    def _write_png(path, rgb):
        h, w = rgb.shape[:2]
        raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))

        def chunk(typ, data):
            return (
                struct.pack(">I", len(data))
                + typ
                + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
            )

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(raw))
        png += chunk(b"IEND", b"")
        with open(path, "wb") as f:
            f.write(png)

    # ------------------------------------------------------------------
    # 保存金币矩阵区域截图（BGR numpy -> RGB -> PNG）
    # ------------------------------------------------------------------
    @staticmethod
    def _save_roi_image(image_bgr, roi, out_dir):
        try:
            x, y, w, h = roi
            if x < 0 or y < 0:
                logger.warning(f"ROI 出现负值，跳过截图: {roi}")
                return ""
            crop = image_bgr[y:y + h, x:x + w]
            rgb = crop[..., ::-1]
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"gold_{time.strftime('%Y%m%d_%H%M%S')}.png")
            try:
                from PIL import Image

                Image.fromarray(rgb, "RGB").save(path)
            except Exception:
                MengghuanXiaomiPush._write_png(path, rgb)
            return path
        except Exception as e:
            logger.warning(f"保存金币矩阵截图失败: {e}")
            return ""

    # ------------------------------------------------------------------
    # 钉钉推送（msgtype: text / markdown），secret 非空则加签
    # ------------------------------------------------------------------
    @staticmethod
    def _push_dingtalk(webhook_url, secret, msg, msgtype="text"):
        if not webhook_url:
            logger.warning("未配置钉钉 Webhook 地址，跳过推送")
            return False

        url = webhook_url
        # 加签：timestamp + "\n" + secret 做 HMAC-SHA256，再 base64 + urlencode
        if secret:
            try:
                # 只打安全摘要（长度/首尾/有无空白），不打全文，避免日志再次泄露凭证。
                # 复制 secret 时混入空格或换行，是「签名不匹配」最常见的原因。
                logger.info(
                    "secret 检查: 长度=%d 前缀=%s 后缀=%s 首尾有空白=%s"
                    % (len(secret), secret[:3], secret[-4:], secret != secret.strip())
                )
                timestamp = str(round(time.time() * 1000))
                string_to_sign = f"{timestamp}\n{secret}"
                hmac_code = hmac.new(
                    secret.encode("utf-8"),
                    string_to_sign.encode("utf-8"),
                    hashlib.sha256,
                ).digest()
                sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}timestamp={timestamp}&sign={sign}"
            except Exception as e:
                logger.warning(f"加签计算失败，按无加签推送: {e}")

        if msgtype == "markdown":
            payload = {
                "msgtype": "markdown",
                "markdown": {"title": "梦幻小蜜", "text": msg},
            }
        else:
            payload = {"msgtype": "text", "text": {"content": msg}}

        try:
            if _requests is not None:
                resp = _requests.post(
                    url,
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
                text = resp.text
            else:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    text = resp.read().decode("utf-8", "ignore")
            logger.info(f"钉钉返回: {text}")
            # 钉钉拒收时 HTTP 仍是 200，错误只藏在响应体的 errcode 里；
            # 不看它就会把「被拒绝」当成「推送成功」，排查时完全看不到线索
            try:
                body = json.loads(text)
            except Exception:
                body = {}
            code = body.get("errcode")
            if code is None or code == 0:
                return True
            logger.warning(
                f"钉钉拒收: errcode={code} errmsg={body.get('errmsg')}"
                f"{MengghuanXiaomiPush._hint(code)}"
            )
            return False
        except Exception as e:
            logger.warning(f"钉钉推送异常: {e}")
            return False

    # ------------------------------------------------------------------
    # 常见错误码的中文提示：钉钉只回英文 errmsg，这里补一句人话
    # ------------------------------------------------------------------
    @staticmethod
    def _hint(code):
        return {
            300001: "（webhook 的 access_token 不对，核对机器人地址）",
            310000: "（安全设置没过：自定义关键词没命中，或加签 secret 与后台不一致）",
            400013: "（机器人已被停用，或已被移出群）",
            410100: "（发得太频繁，每分钟上限 20 条）",
        }.get(code, "")

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(self, context: Context, argv: CustomAction.RunArg):
        logger.info("进入梦幻小蜜推送")

        argv_dict = {}
        if argv.custom_action_param:
            try:
                argv_dict = json.loads(argv.custom_action_param)
            except Exception:
                logger.warning("custom_action_param 解析失败，使用默认配置")

        recognition_name = argv_dict.get("recognition_name", "梦幻小蜜-金币识别")
        roi = argv_dict.get("roi", [177, 633, 174, 45])

        # 取值优先级：UI 注入(custom_action_param) → 备用配置文件 → 默认值
        cfg = self._load_config()
        webhook_url = argv_dict.get("webhook_url") or cfg.get("webhook_url") or ""
        secret = argv_dict.get("secret") or cfg.get("secret") or ""
        role = argv_dict.get("role") or cfg.get("role") or ""
        template = argv_dict.get("message_template") or cfg.get("message_template") or ""
        msgtype = argv_dict.get("msgtype") or cfg.get("msgtype") or "text"
        screenshot = bool(argv_dict.get("screenshot", False))
        screenshot_dir = argv_dict.get(
            "screenshot_dir", "./debug/custom/mengghuan_xiaomi"
        )

        # 1. 截图
        image = context.tasker.controller.post_screencap().wait().get()
        if image is None:
            logger.error("截图失败，无法识别金币")
            return CustomAction.RunResult(success=False)

        # 2. OCR 识别金币矩阵（沿用现成坐标 [177,633,174,45]，也可由 UI/参数覆盖）
        gold_text = ""
        try:
            reco = context.run_recognition(
                recognition_name,
                image,
                pipeline_override={
                    recognition_name: {
                        "roi": roi,
                        "expected": [],
                        "recognition": "OCR",
                    }
                },
            )
            if reco and reco.hit:
                # 按位置顺序组合：先按 y 再按 x 排序，不删减，原样拼接
                results = sorted(
                    reco.all_results, key=lambda r: (r.box[1], r.box[0])
                )
                raw_text = "".join(r.text for r in results if r.text)
                # 只保留数字：去掉 OCR 误识别的千分位小数点/逗号等非数字字符
                gold_text = re.sub(r"\D", "", raw_text)
                logger.info(f"识别金币: {gold_text}")
            else:
                logger.warning("未识别到金币")
                gold_text = "识别失败"
        except Exception as e:
            logger.warning(f"金币 OCR 识别异常: {e}")
            gold_text = "识别失败"

        # 3. 可选：保存金币矩阵截图（作为推送备选方案）
        screenshot_path = ""
        if screenshot:
            screenshot_path = self._save_roi_image(image, roi, screenshot_dir)
            if screenshot_path:
                logger.info(f"金币矩阵截图已保存: {screenshot_path}")

        # 4. 组装消息（替换模板占位符）
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        if not template:
            template = (
                "🎮 任务已完成\n"
                "━━━━━━━━━━━━━━━\n"
                "⏰ 时间: {time}\n"
                "📱 角色: {role}\n"
                "━━━━━━━━━━━━━━━\n"
                "💰 : {gold}\n"
                "━━━━━━━━━━━━━━━"
            )
        msg = (
            template.replace("{time}", now)
            .replace("{role}", str(role))
            .replace("{gold}", str(gold_text))
        )

        # 5. 钉钉推送
        ok = self._push_dingtalk(webhook_url, secret, msg, msgtype)
        if ok:
            logger.info(f"钉钉推送成功，金币: {gold_text}")
        else:
            logger.warning("钉钉推送失败或未配置")

        return CustomAction.RunResult(success=True)
