import ast
from collections.abc import Iterable
import json
import re
from typing import Any

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.define import OCRResult


# 匹配表达式中的占位符，例如 {节点名}
PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")
# 匹配数字，支持整数、小数、正负号
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


class NodeResolutionError(ValueError):
    """节点值解析失败时抛出，携带详细 payload 供上层记录。"""

    def __init__(self, message: str, payload: dict[str, Any]):
        super().__init__(message)
        self.payload = payload


@AgentServer.custom_recognition("ExpressionRecognition")
class ExpressionRecognition(CustomRecognition):
    """自定义识别：根据表达式和多个节点的 OCR 数值结果进行布尔判断。"""

    def analyze(
        self,
        context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult | None:
        """MAA 自定义识别入口。"""
        # 解析 custom_recognition_param，获取表达式
        params = self._parse_params(argv.custom_recognition_param)
        expression = params.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            return CustomRecognition.AnalyzeResult(box=None, detail={"status": "invalid expression"})

        image = argv.image
        # 缓存占位符对应的变量值，变量名形如 _value_0
        values_cache: dict[str, int | float] = {}
        # 记录占位符节点名到变量名的映射
        placeholder_mapping: dict[str, str] = {}
        # 记录每个节点解析的详细结果
        node_results: dict[str, Any] = {}

        def replace_placeholder(match: re.Match[str]) -> str:
            """将表达式中的 {节点名} 替换为内部变量名，并解析节点数值。"""
            node_name = match.group(1).strip()
            if not node_name:
                raise ValueError("empty placeholder")

            variable_name = placeholder_mapping.get(node_name)
            if variable_name is None:
                # 为每个节点分配唯一变量名
                variable_name = f"_value_{len(placeholder_mapping)}"
                placeholder_mapping[node_name] = variable_name
                try:
                    # 解析节点，获取数值和详情
                    value, node_result = self._resolve_node_value(context, image, node_name)
                except NodeResolutionError as exc:
                    node_results[node_name] = exc.payload
                    raise

                values_cache[variable_name] = value
                node_results[node_name] = node_result
            return variable_name

        try:
            # 1. 替换占位符为变量名
            python_expression = PLACEHOLDER_PATTERN.sub(replace_placeholder, expression)
            # 2. 将 &&、||、! 转换为 Python 的 and、or、not
            python_expression = self._normalize_expression(python_expression)
            # 3. 解析为 AST
            parsed = ast.parse(python_expression, mode="eval")
            # 4. 校验 AST，只允许安全语法和已解析的变量名
            self._validate_ast(parsed, set(values_cache.keys()))
            # 5. 在受限环境中求值
            result = eval(
                compile(parsed, "<ExpressionRecognition>", "eval"),
                {"__builtins__": {}},
                values_cache,
            )
        except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as exc:
            # 统一处理表达式错误
            detail = {
                "status": "invalid expression",
                "reason": str(exc),
                "expression": expression,
                "resolved_expression": locals().get("python_expression", expression),
                "resolved_values": values_cache,
                "node_results": node_results,
            }
            detail["summary"] = self._format_summary(detail)
            return CustomRecognition.AnalyzeResult(
                box=None,
                detail=detail,
            )

        # 表达式必须求值为 bool
        if type(result) is not bool:
            detail = {
                "status": "expression did not evaluate to boolean",
                "expression": expression,
                "resolved_expression": python_expression,
                "resolved_values": values_cache,
                "node_results": node_results,
            }
            detail["summary"] = self._format_summary(detail)
            return CustomRecognition.AnalyzeResult(
                box=None,
                detail=detail,
            )

        # 表达式为 False，识别不命中
        if not result:
            detail = {
                "status": "expression evaluated to false",
                "expression": expression,
                "resolved_expression": python_expression,
                "resolved_values": values_cache,
                "node_results": node_results,
            }
            detail["summary"] = self._format_summary(detail)
            return CustomRecognition.AnalyzeResult(
                box=None,
                detail=detail,
            )

        # 表达式为 True，返回命中
        detail = {
            "status": "success",
            "expression": expression,
            "resolved_expression": python_expression,
            "resolved_values": values_cache,
            "node_results": node_results,
        }
        detail["summary"] = self._format_summary(detail)
        return CustomRecognition.AnalyzeResult(
            box=(0, 0, 100, 100),  # 命中时返回一个占位框
            detail=detail,
        )

    def _parse_params(self, raw_params: Any) -> dict[str, Any]:
        """解析 custom_recognition_param，支持 dict 或 JSON 字符串。"""
        if isinstance(raw_params, dict):
            return raw_params
        if isinstance(raw_params, str):
            try:
                parsed = json.loads(raw_params)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    def _resolve_node_value(self, context, image, node_name: str) -> tuple[int | float, dict[str, Any]]:
        """运行指定节点的识别，提取 OCR 文本中的第一个数字并返回。"""
        # 获取节点配置数据
        node_data = context.get_node_data(node_name) or {}
        # 读取配置中的 box_index，用于多结果选择
        box_index = self._get_box_index(node_data)
        # 执行识别
        recognition = context.run_recognition(node_name, image)
        payload = {
            "node_data": self._to_jsonable(node_data),
            "box_index": box_index,
            "recognition": self._summarize_recognition(recognition),
        }
        if not (recognition and recognition.hit):
            raise NodeResolutionError(f"node {node_name} has no OCR result", payload)

        # 从识别结果中提取文本
        text = self._extract_text(recognition, box_index=box_index)
        payload["extracted_text"] = text
        if text is None:
            raise NodeResolutionError(f"node {node_name} has no OCR text", payload)

        # 匹配第一个数字
        match = NUMBER_PATTERN.search(text)
        if match is None:
            raise NodeResolutionError(f"node {node_name} has no numeric OCR text", payload)

        number_text = match.group(0)
        if "." in number_text:
            value = float(number_text)
        else:
            value = int(number_text)

        payload["value"] = value
        return value, payload

    def _get_box_index(self, node_data: Any) -> int | None:
        """从节点配置中读取 recognition.param.box_index。"""
        if not isinstance(node_data, dict):
            return None

        recognition = node_data.get("recognition")
        if not isinstance(recognition, dict):
            return None

        param = recognition.get("param")
        if not isinstance(param, dict):
            return None

        box_index = param.get("box_index")
        if isinstance(box_index, int) and box_index >= 0:
            return box_index
        return None

    def _extract_text(self, result: Any, box_index: int | None = None) -> str | None:
        """从各种识别结果结构中递归提取 OCR 文本。"""
        # 优先按 box_index 提取
        if box_index is not None:
            indexed_text = self._extract_text_by_index(result, box_index)
            if indexed_text is not None:
                return indexed_text

        if isinstance(result, dict):
            direct_text = result.get("text")
            if isinstance(direct_text, str):
                return direct_text

            # 尝试常见的嵌套键
            for key in ("best", "best_result", "detail", "filtered", "filtered_results", "all"):
                if key in result:
                    nested_text = self._extract_text(result[key])
                    if nested_text is not None:
                        return nested_text
            return None

        if isinstance(result, Iterable) and not isinstance(result, (str, bytes, bytearray)):
            # 可迭代对象，逐个尝试
            for item in result:
                item_text = self._extract_text(item)
                if item_text is not None:
                    return item_text
            return None

        if isinstance(result, OCRResult):
            return result.text

        # 对象属性方式
        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text

        detail = getattr(result, "detail", None)
        if detail is not None and detail is not result:
            detail_text = self._extract_text(detail)
            if detail_text is not None:
                return detail_text

        best_result = getattr(result, "best_result", None)
        if best_result is not None and best_result is not result:
            best_text = self._extract_text(best_result)
            if best_text is not None:
                return best_text

        filtered_results = getattr(result, "filtered_results", None)
        if filtered_results is not None and filtered_results is not result:
            for item in filtered_results:
                item_text = self._extract_text(item)
                if item_text is not None:
                    return item_text

        return None

    def _extract_text_by_index(self, result: Any, box_index: int) -> str | None:
        """按索引从子结果列表中提取文本。"""
        children = self._extract_children(result)
        if children is None or not 0 <= box_index < len(children):
            return None

        return self._extract_text(children[box_index])

    def _extract_children(self, result: Any) -> list[Any] | None:
        """尝试获取识别结果的子结果列表。"""
        if isinstance(result, dict):
            for key in ("sub_results", "detail", "all", "filtered", "filtered_results"):
                value = self._coerce_sequence(result.get(key))
                if value is not None:
                    return value

            raw_detail = result.get("raw_detail")
            if raw_detail is not None:
                raw_children = self._extract_children(raw_detail)
                if raw_children is not None:
                    return raw_children
            return None

        sub_results = self._coerce_sequence(getattr(result, "sub_results", None))
        if sub_results is not None:
            return sub_results

        detail = self._coerce_sequence(getattr(result, "detail", None))
        if detail is not None:
            return detail

        raw_detail = getattr(result, "raw_detail", None)
        if raw_detail is not None:
            raw_children = self._extract_children(raw_detail)
            if raw_children is not None:
                return raw_children

        all_results = self._coerce_sequence(getattr(result, "all", None))
        if all_results is not None:
            return all_results

        filtered_results = self._coerce_sequence(getattr(result, "filtered_results", None))
        if filtered_results is not None:
            if len(filtered_results) == 1:
                nested_children = self._extract_children(filtered_results[0])
                if nested_children is not None:
                    return nested_children
            return filtered_results

        best_result = getattr(result, "best_result", None)
        if best_result is not None:
            best_children = self._extract_children(best_result)
            if best_children is not None:
                return best_children

        return None

    def _coerce_sequence(self, value: Any) -> list[Any] | None:
        """将非字符串/字节/字典的可迭代对象转换为列表。"""
        if value is None or isinstance(value, (str, bytes, bytearray, dict)):
            return None

        if isinstance(value, list):
            return value

        try:
            return list(value)
        except TypeError:
            return None

    def _summarize_recognition(self, value: Any) -> Any:
        """将识别结果转为可 JSON 序列化的摘要，并移除 node_data 避免冗余。"""
        summary = self._to_jsonable(value)
        if isinstance(summary, dict):
            summary.pop("node_data", None)
        return summary

    def _format_summary(self, detail: dict[str, Any]) -> str:
        """将 detail 格式化为便于日志阅读的多行摘要。"""
        lines = [
            f"status: {detail.get('status')}",
            f"expression: {detail.get('expression')}",
            f"resolved_expression: {detail.get('resolved_expression')}",
        ]

        reason = detail.get("reason")
        if reason:
            lines.append(f"reason: {reason}")

        resolved_values = detail.get("resolved_values") or {}
        lines.append("resolved_values:")
        if resolved_values:
            for key, value in resolved_values.items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append("  <empty>")

        lines.append("node_results:")
        node_results = detail.get("node_results") or {}
        if node_results:
            for node_name, node_result in node_results.items():
                lines.append(f"  - {node_name}")
                if isinstance(node_result, dict):
                    lines.append(f"    box_index: {node_result.get('box_index')}")
                    lines.append(f"    extracted_text: {node_result.get('extracted_text')}")
                    if "value" in node_result:
                        lines.append(f"    value: {node_result.get('value')}")

                    recognition = node_result.get("recognition")
                    if isinstance(recognition, dict):
                        lines.append(f"    recognition_type: {recognition.get('type')}")
                        lines.append(f"    recognition_name: {recognition.get('name')}")
                        lines.append(f"    recognition_algorithm: {recognition.get('algorithm')}")
                        lines.append(f"    recognition_hit: {recognition.get('hit')}")
                        lines.append(f"    recognition_box: {recognition.get('box')}")
                else:
                    lines.append(f"    {node_result}")
        else:
            lines.append("  <empty>")

        return "\n".join(lines)

    def _to_jsonable(self, value: Any, depth: int = 0) -> Any:
        """递归将对象转换为可 JSON 序列化的结构，限制深度避免无限递归。"""
        if depth >= 6:
            return self._safe_repr(value)

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dict):
            return {
                str(key): self._to_jsonable(item, depth + 1)
                for key, item in value.items()
            }

        if isinstance(value, OCRResult):
            return {
                "type": type(value).__name__,
                "text": value.text,
            }

        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            return [self._to_jsonable(item, depth + 1) for item in value]

        result: dict[str, Any] = {"type": type(value).__name__}
        for attr in (
            "hit",
            "text",
            "box",
            "score",
            "detail",
            "raw_detail",
            "best_result",
            "filtered_results",
            "sub_results",
            "all",
            "algorithm",
            "name",
            "reco_id",
        ):
            if hasattr(value, attr):
                attr_value = getattr(value, attr)
                if attr_value is not None:
                    result[attr] = self._to_jsonable(attr_value, depth + 1)

        if len(result) == 1:
            result["repr"] = self._safe_repr(value)
        return result

    def _safe_repr(self, value: Any, limit: int = 240) -> str:
        """生成安全的 repr 字符串，超长截断。"""
        text = repr(value)
        if len(text) <= limit:
            return text
        return f"{text[:limit]}...<truncated>"

    def _normalize_expression(self, expression: str) -> str:
        """将常见逻辑运算符转换为 Python 语法。"""
        normalized = expression.replace("&&", " and ").replace("||", " or ")
        normalized = re.sub(r"!(?!=)", " not ", normalized)
        return normalized

    def _validate_ast(self, node: ast.AST, allowed_names: set[str]) -> None:
        """校验 AST，只允许安全节点和已解析的变量名，防止代码注入。"""
        for child in ast.walk(node):
            if isinstance(
                child,
                (
                    ast.Expression,
                    ast.Load,
                    ast.BinOp,
                    ast.BoolOp,
                    ast.UnaryOp,
                    ast.Compare,
                    ast.Name,
                    ast.Constant,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.Mod,
                    ast.And,
                    ast.Or,
                    ast.Not,
                    ast.UAdd,
                    ast.USub,
                    ast.Eq,
                    ast.NotEq,
                    ast.Lt,
                    ast.LtE,
                    ast.Gt,
                    ast.GtE,
                ),
            ):
                if isinstance(child, ast.Name) and child.id not in allowed_names:
                    raise ValueError(f"unexpected name {child.id}")
                if isinstance(child, ast.Constant) and not isinstance(
                    child.value, (int, float, bool)
                ):
                    raise ValueError("unsupported constant")
                continue
            raise ValueError(f"unsupported syntax {type(child).__name__}")
