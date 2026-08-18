
from PIL import Image
from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
from typing import Dict, List, Tuple
from utils import logger, SendJinSan, HumanSimulator
import time
import re

from .searchAnswer import SearchQuestions

@AgentServer.custom_recognition("reco2")
class reco2_sjqy(CustomRecognition):
    def analyze(
         self,
         context: Context,
         argv: CustomRecognition.AnalyzeArg,
     ) -> CustomRecognition.AnalyzeResult:
        logger.info("进入reco2_sjqy")
        return CustomRecognition.AnalyzeResult(box=(6,134,57,46),detail="福利")

 

@AgentServer.custom_recognition("sjqy_tiku_V2")
class sjqy_tiku_V2(CustomRecognition):

    def analyze(
         self,
         context: Context,
         argv: CustomRecognition.AnalyzeArg,
     ) -> CustomRecognition.AnalyzeResult:
        logger.info("进入三界奇缘答题agent_V2")

            # 根据位置排序OCR结果
        def sort_ocr_results_by_position(ocr_results):
            # 定义行高阈值，如果两个框的y坐标差距小于这个值，认为它们在同一行
            row_height_threshold = 20
            
            # 按y坐标分组（将接近的y坐标视为同一行）
            rows = {}
            for result in ocr_results:
                y = result.box[1]  # y坐标
                assigned = False
                
                # 检查是否可以分配到现有行
                for row_y in rows.keys():
                    if abs(y - row_y) < row_height_threshold:
                        rows[row_y].append(result)
                        assigned = True
                        break
                
                # 如果不能分配到现有行，创建新行
                if not assigned:
                    rows[y] = [result]
            
            # 对每一行内的元素按x坐标排序
            for row_y in rows:
                rows[row_y].sort(key=lambda r: r.box[0])
            
            # 按y坐标对行进行排序，并将所有结果合并到一个列表中
            sorted_results = []
            for row_y in sorted(rows.keys()):
                sorted_results.extend(rows[row_y])
            
            return sorted_results

        i = 0
        # 未找到答案次数
        NotAnswerCount = 0
        while i < 30:
            i = i+1
            # logger.info(f"第{i}次识别三界奇缘题目")
            #识别三界奇缘题目
            image1 = context.tasker.controller.post_screencap().wait().get()
            reco_detail = context.run_recognition(
                "三界奇缘题目",
                image1,
                pipeline_override={"三界奇缘题目": {"roi" : [447,40,673,94],
                                                    "expected":[""],
                                                    "recognition": "OCR"
                                                    }
                                    }
                )
            #识别题目返回值
            # logger.info("reco_detail为：{reco_detail}")
            
            # logger.info(reco_detail.box)
            # 没有识别到题目
            if not reco_detail or not reco_detail.hit:
                # logger.info("没有识别到题目")
                logger.info(f"未在题库中搜索到答案次数:{NotAnswerCount}，请反馈开发者填充题库。")
                return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="答题结束")
            all_results= reco_detail.all_results
            #按照box进行排序
            sorted_results= sort_ocr_results_by_position(all_results)
            #整合题目
            ext=''
            for item in sorted_results:
                if item.text!='':
                    ext+=item.text
            pattern = r'第\d+题：|[（(]\d+/\d+[）)]'
            def clean_string(s):
                return re.sub(pattern, '', s).strip()
            text = clean_string(ext)
            # 获取答案list[]
            results_value, confidence ,match_type = SearchQuestions(text)
            # 如果可信度为零，点击第一个答案（拟人化偏移与延时）
            if confidence == 0:
                # logger.info(f"题库中未找到答案，问题为:{text}.请反馈开发者填充题库")
                NotAnswerCount = NotAnswerCount + 1
                HumanSimulator.sleep_human(1.2, 0.2)
                tx, ty = HumanSimulator.get_point_offset(500, 344, radius=6.0)
                context.tasker.controller.post_click(tx, ty).wait()
                HumanSimulator.sleep_human(2.5, 0.25)
                continue
        
            # 识别答案位置
            # new_context = context.clone()
            # image2 = context.tasker.controller.post_screencap().wait().get()
            new_reco_detail = context.run_recognition(
                            "三界奇缘答案位置",
                            image1,
                            pipeline_override={"三界奇缘答案位置": {"roi" : [439,218,678,212],
                                                                "expected":results_value,
                                                                "recognition": "OCR"
                                                                }
                                                }
                            )
            # logger.info("new_reco_detail为：{new_reco_detail}")
            # logger.info(new_reco_detail.box)
            # 点击答案（拟人化高斯落点与延时）
            if new_reco_detail and new_reco_detail.hit:
                box = new_reco_detail.box  # 假设box为(x, y, w, h)
                target_x, target_y = HumanSimulator.get_gaussian_point(box)
                HumanSimulator.sleep_human(1.2, 0.2)
                click_job = context.tasker.controller.post_click(target_x, target_y)
                click_job.wait()  # 等待点击操作完成
                HumanSimulator.sleep_human(1.8, 0.25)
                HumanSimulator.random_idle(chance=0.08, min_sec=2.0, max_sec=5.0)
            else:  # 没找到答案，拟人化点击第一个
                HumanSimulator.sleep_human(1.2, 0.2)
                tx, ty = HumanSimulator.get_point_offset(500, 344, radius=6.0)
                context.tasker.controller.post_click(tx, ty).wait()
                HumanSimulator.sleep_human(1.0, 0.2)

        logger.info(f"未在题库中搜索到答案次数:{NotAnswerCount}，请反馈开发者填充题库。")
        return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="答题结束")

@AgentServer.custom_recognition("sjqy_tiku_V3")
class sjqy_tiku_V3(CustomRecognition):

    def analyze(
         self,
         context: Context,
         argv: CustomRecognition.AnalyzeArg,
     ) -> CustomRecognition.AnalyzeResult:
        # logger.info("进入三界奇缘答题agent_V3")

            # 根据位置排序OCR结果
        def sort_ocr_results_by_position(ocr_results):
            # 定义行高阈值，如果两个框的y坐标差距小于这个值，认为它们在同一行
            row_height_threshold = 20
            
            # 按y坐标分组（将接近的y坐标视为同一行）
            rows = {}
            for result in ocr_results:
                y = result.box[1]  # y坐标
                assigned = False
                
                # 检查是否可以分配到现有行
                for row_y in rows.keys():
                    if abs(y - row_y) < row_height_threshold:
                        rows[row_y].append(result)
                        assigned = True
                        break
                
                # 如果不能分配到现有行，创建新行
                if not assigned:
                    rows[y] = [result]
            
            # 对每一行内的元素按x坐标排序
            for row_y in rows:
                rows[row_y].sort(key=lambda r: r.box[0])
            
            # 按y坐标对行进行排序，并将所有结果合并到一个列表中
            sorted_results = []
            for row_y in sorted(rows.keys()):
                sorted_results.extend(rows[row_y])
            
            return sorted_results

        # 未找到答案次数
        NotAnswerCount = 0
        # 当第一次进入答题时，需要暂停一秒后在开始截图。
        max_hit = context.get_hit_count("活动-三界奇缘-开始答题_agent")
        if max_hit == 0:
            HumanSimulator.sleep_human(1.5, 0.2)
        
        #识别三界奇缘题目
        image1 = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition(
            "三界奇缘题目",
            image1,
            pipeline_override={"三界奇缘题目": {"roi" : [447,40,673,94],
                                                "expected":[".*"],
                                                "recognition": "OCR",
                                                "replace":[
                                                    ["味","昧"],
                                                    ["邮","邺"],
                                                    ["尺","尸"],
                                                    ["频","濒"],
                                                    ["铜","锢"]
                                                ]
                                                }
                                }
            )
        #识别题目返回值
        # logger.info("reco_detail为：{reco_detail}")
        
        # logger.info(reco_detail.box)
        # 没有识别到题目
        if not reco_detail or not reco_detail.hit:
            logger.info("没有识别到题目")
            # logger.info(f"未在题库中搜索到答案次数:{NotAnswerCount}，请反馈开发者填充题库。")
            return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="答题结束")
        filtered_results= reco_detail.filtered_results
        #按照box进行排序
        sorted_results= sort_ocr_results_by_position(filtered_results)
        #整合题目
        ext=''
        for item in sorted_results:
            if item.text!='':
                ext+=item.text
        pattern = r'第\d+题：|[（(]\d+/\d+[）)]'
        def clean_string(s):
            return re.sub(pattern, '', s).strip()
        text = clean_string(ext)
        # 获取答案list[]
        results_value, confidence ,match_type = SearchQuestions(text)
        # 获取时间并格式化
        formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 如果可信度为零，点击第一个答案
        if confidence < 80:
            # logger.info(f"题库中未找到答案，问题为:{text}.请反馈开发者填充题库")
            # 预留接口，未在题库中找得到问题，可以把截图发送到服务器。
            # context.tasker.controller.post_screencap().wait().save("{text}.png")
            # 发送到金山文档1.识别到的问题，2.匹配度，3.题库答案，4.时间
            data_to_send = [text,confidence,results_value,formatted]
            # logger.info(f"发送到金山文档的数据为:{data_to_send}")   
            if SendJinSan.send(data_to_send):
                logger.info(f"[color:red]未与题库匹配。[/color]识别题目：{text}，匹配度：{confidence}，已经登记在线文档")
            else:
                logger.error(f"登记在线文档失败:{text}")
            HumanSimulator.sleep_human(1.2, 0.2)
            tx, ty = HumanSimulator.get_point_offset(500, 344, radius=6.0)
            context.tasker.controller.post_click(tx, ty).wait()
            HumanSimulator.sleep_human(2.5, 0.25)
            return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="题库缺少本问题")
        elif confidence >=80 and confidence < 100:
            
            # 发送到金山文档
            data_to_send = [text,confidence,results_value,formatted]
            # logger.info(f"发送到金山文档的数据为:{data_to_send}")   
            if SendJinSan.send(data_to_send):
                logger.info(f"本问题与题库非100%匹配:{text},已经登记在线文档")
            else:
                logger.error(f"登记在线文档失败:{text}")
        # 识别答案位置
        # new_context = context.clone()
        # image2 = context.tasker.controller.post_screencap().wait().get()
        new_reco_detail = context.run_recognition(
                        "三界奇缘答案位置",
                        image1,
                        pipeline_override={"三界奇缘答案位置": {"roi" : [439,218,678,212],
                                                            "expected":results_value,
                                                            "recognition": "OCR"
                                                            }
                                            }
                        )
        # logger.info("new_reco_detail为：{new_reco_detail}")
        # logger.info(new_reco_detail.box)
        # 点击答案（拟人化高斯落点与延时）
        if new_reco_detail and new_reco_detail.hit:
            box = new_reco_detail.box  # 假设box为(x, y, w, h)
            target_x, target_y = HumanSimulator.get_gaussian_point(box)
            HumanSimulator.sleep_human(1.2, 0.2)
            click_job = context.tasker.controller.post_click(target_x, target_y)
            click_job.wait()  # 等待点击操作完成
            logger.info(f"[color:blue]正确点击答案。[/color]识别题目：{text}。题库答案：{results_value}。匹配度：{confidence}。落点：({target_x}, {target_y})")
            HumanSimulator.sleep_human(1.8, 0.25)
            HumanSimulator.random_idle(chance=0.08, min_sec=2.0, max_sec=5.0)
        else:  # 没找到答案，点击第一个
            HumanSimulator.sleep_human(1.2, 0.2)
            tx, ty = HumanSimulator.get_point_offset(500, 344, radius=6.0)
            context.tasker.controller.post_click(tx, ty).wait()
            logger.info(f"[color:red]未在界面找到答案，点击第一个答案。[/color]识别题目：{text}。题库答案：{results_value}。匹配度：{confidence}。落点：({tx}, {ty})")
            HumanSimulator.sleep_human(1.0, 0.2)

        # logger.info(f"未在题库中搜索到答案次数:{NotAnswerCount}，请反馈开发者填充题库。")
        return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="答题结束")
