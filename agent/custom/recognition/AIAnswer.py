from maa.agent.agent_server import AgentServer
from maa.custom_recognition  import CustomRecognition
from maa.context import Context
from utils import logger
import requests
import json
import time


@AgentServer.custom_recognition("AIAnswer")
class AIAnswer(CustomRecognition):
        def analyze(
         self,
         context: Context,
         argv: CustomRecognition.AnalyzeArg,
     ) -> CustomRecognition.AnalyzeResult:
            # logger.info("进入AIAnswer")
            # 对问题进行排序
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

            # 获取界面图片
            image1 = context.tasker.controller.post_screencap().wait().get()
            reco_detail = context.run_recognition(
                            "科举乡试题目",
                            image1,
                            
                            pipeline_override={"科举乡试题目": {"roi" : [511,186,602,107],
                                                                "expected":[""],
                                                                "recognition": "OCR"
                                                                }
                                                }
                            )
            # 没有识别到科举乡试题目
            if not reco_detail or not reco_detail.hit:
                # logger.info("没有识别到科举乡试题目")
                # logger.info(f"未在题库中搜索到答案次数:{NotAnswerCount}，请反馈开发者填充题库。")
                return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="答题结束")
            all_results= reco_detail.all_results
            #按照box进行排序
            sorted_results= sort_ocr_results_by_position(all_results)
            #整合科举乡试题目
            question=''
            for item in sorted_results:
                if item.text!='':
                    question+=item.text
            # logger.info(f"question:{question}")
            
            # 获取答案
            A= ""
            B= ""
            C= ""
            D= ""
            #科举乡试答案a
            reco_detail_A=context.run_recognition(
                            "科举乡试答案a",
                            image1,
                            pipeline_override={"科举乡试答案a": {"roi" : [509,306,269,91],
                                                                "expected":[""],
                                                                "recognition": "OCR"
                                                                }
                                                }
                            )
            # logger.info(f"reco_detail_A:{reco_detail_A}")
            for res in reco_detail_A.all_results:
                A =res.text
                # logger.info(f"A:{A}")
            
            #科举乡试答案b
            reco_detail_B=context.run_recognition(
                            "科举乡试答案b",
                            image1,
                            pipeline_override={"科举乡试答案b": {"roi" : [825,304,270,95],
                                                                "expected":[""],
                                                                "recognition": "OCR"
                                                                }
                                                }
                            )
            for res in reco_detail_B.all_results:
                B =res.text
                # logger.info(f"B:{B}")
            
            #科举乡试答案c
            reco_detail_C=context.run_recognition(
                            "科举乡试答案c",
                            image1,
                            pipeline_override={"科举乡试答案c": {"roi" : [506,408,268,88],
                                                                "expected":[""],
                                                                "recognition": "OCR"
                                                                }
                                                }
                            )
            if reco_detail_C and reco_detail_C.hit:
                for res in reco_detail_C.all_results:
                    C =res.text
                    # logger.info(f"C:{C}")
            
            # 科举乡试答案d
            reco_detail_D=context.run_recognition(
                            "科举乡试答案d",
                            image1,
                            pipeline_override={"科举乡试答案d": {"roi" : [831,404,265,96],
                                                                "expected":[""],
                                                                "recognition": "OCR"
                                                                }
                                                }
                            )
            if reco_detail_D and reco_detail_D.hit:
                for res in reco_detail_D.all_results:
                    D =res.text
                    # logger.info(f"D:{D}")
            
            answer = {"A":A,
                      "B":B,
                      "C":C,
                      "D":D}
            # logger.info(f"问题为：{question}")
            # logger.info(f"答案为：{answer}")
            # 获取传参节点apikey数据
            # UIpiKey: dict = context.get_node_data("AIapikey")['recognition']['param']['custom_recognition_param']['apikey']
            UIpiKey: dict = context.get_node_data("活动-科举乡试-开始答题API")["attach"]["apikey"]
            UIurl:dict = context.get_node_data("活动-科举乡试-开始答题API")["attach"]["url"]
            UImodel:dict = context.get_node_data("活动-科举乡试-开始答题API")["attach"]["model"]
            # logger.info(f"用户输入的aikey: {data1}")
            # 向ai发送问题及答案并获得正确答案
            def get_ai_answer(question, answers):
                # 过滤掉值为空的答案
                valid_answers = {k: v for k, v in answers.items() if v}
                
                # 构建 prompt
                prompt = f"问题：{question}\n"
                prompt += "请从以下选项中选择一个最正确的答案，并只返回选项的字母（例如：A, B, C, D）。\n"
                for key, value in valid_answers.items():
                    prompt += f"{key}: {value}\n"

                # openai api协议调用
                url = UIurl
                apiKey = UIpiKey

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {apiKey}"
                }

                data = {
                    "model": UImodel,   
                    "messages": [
                        {"role": "system", "content": "你是一个答题助手"},
                        {"role": "user", "content": prompt}
                    ],
                    "thinking": {"type": "disabled"},
                    "temperature": 0.7,
                    "max_tokens": 10,
                    "stream": False
                }

                try:
                    response = requests.post(url, headers=headers, data=json.dumps(data))
                    response.raise_for_status()  # 如果请求失败，则引发HTTPError

                    # 解析AI的回复
                    ai_response = response.json()
                    ai_answer = ai_response['choices'][0]['message']['content'].strip().upper()

                    # 验证AI的回复
                    if ai_answer in valid_answers:
                        return ai_answer
                    else:
                        # 如果回复不在ABCD中，留下注释
                        return f"AI回复无效: {ai_answer}"

                except requests.exceptions.RequestException as e:
                    return f"请求错误: {e}"
                except (KeyError, IndexError) as e:
                    return f"解析回复时出错: {e}"
            listAnswer= get_ai_answer(question,answer)
            # logger.info(f"listAnswer为：{listAnswer}")
            # 点击box中心位置
            def clickBox(box):
                new_context = context.clone()
                center_x = box[0] + box[2] // 2
                center_y = box[1] + box[3] // 2 
                time.sleep(2)
                click_job = new_context.tasker.controller.post_click(center_x, center_y)
                click_job.wait()  # 等待点击操作完成
                # 向用户ui界面输出日志info
                logger.info(f"AI返回答案：{listAnswer}。识别题目：{question}，识别答案列表：{answer}")
                time.sleep(2)
            if listAnswer =="A" or listAnswer == "a":
                abox=[509,306,269,91]
                clickBox(abox)
            elif listAnswer =="B" or listAnswer == "b":
                bbox=[825,304,270,95]
                clickBox(bbox)
            elif listAnswer =="C" or listAnswer == "c":
                cbox= [506,408,268,88]
                clickBox(cbox)
            elif listAnswer =="D" or listAnswer == "d":
                dbox= [831,404,265,96]
                clickBox(dbox)
            else:
                logger.info(f"ai返回值有问题：{listAnswer}，默认选择第a答案")
                abox=[509,306,269,91]
                clickBox(abox)
            return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="ai答题完成")

