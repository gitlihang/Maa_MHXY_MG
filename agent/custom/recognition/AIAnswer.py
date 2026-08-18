from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
from utils import logger, HumanSimulator
import requests
import json
import time
from zai import ZhipuAiClient

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
            # 点击选项位置（拟人化高斯落点与延时）
            def clickBox(box):
                new_context = context.clone()
                target_x, target_y = HumanSimulator.get_gaussian_point(box)
                HumanSimulator.sleep_human(1.2, 0.2)
                click_job = new_context.tasker.controller.post_click(target_x, target_y)
                click_job.wait()  # 等待点击操作完成
                # 向用户ui界面输出日志info
                logger.info(f"AI返回答案：{listAnswer}。识别题目：{question}，识别答案列表：{answer}。落点坐标：({target_x}, {target_y})")
                HumanSimulator.sleep_human(1.5, 0.25)
                HumanSimulator.random_idle(chance=0.08, min_sec=2.0, max_sec=5.0)
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

@AgentServer.custom_recognition("zhipu")     
class zhipu(CustomRecognition):
    def analyze(
         self,
         context: Context,
         argv: CustomRecognition.AnalyzeArg,
     ) -> CustomRecognition.AnalyzeResult:
        logger.info("进入zhipu")

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
        uipikey: dict = context.get_node_data("活动-科举乡试-开始答题agent-智谱")["attach"]["apikey"]
        # logger.info(f"uipikey:{uipikey}")
        # 调用智谱ai
        def get_answer_from_zhipu(question, options):
            """
            发送问题给智谱AI并获取答案
            """
            
            # 初始化客户端，请替换为您自己的 API Key
            client = ZhipuAiClient(api_key=uipikey)

            # 构造提示词
            prompt = f"{question}\n"
            prompt += "请严格从以下选项中选择一个最合适的答案（只回复字母A、B、C或D）：\n"
            for key, value in options.items():
                prompt += f"{key}. {value}\n"

            try:
                response = client.chat.completions.create(
                    model="GLM-4-Flash-250414",  # 或其他模型model="glm-4"
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"API调用出错: {e}")
                return None

        # 对结果进行分析并格式化
        def solve_riddle(question, answers):
            # 1. 过滤为空的答案
            valid_answers = {k: v for k, v in answers.items() if v and v.strip()}
            
            if not valid_answers:
                # print("错误：没有有效的选项。")
                return f"错误：没有有效的选项。"

            # print(f"问题：{question}")
            # print(f"有效选项：{valid_answers}")

            # 2. 发送请求给AI
            ai_response = get_answer_from_zhipu(question, valid_answers)

            if ai_response is None:
                # print("AI未能返回结果。")
                return f"AI未能返回结果。"

            # print(f"AI原始回复：{ai_response}")

            # 3. 验证AI的回复是否在ABCD中
            # 提取回复中的字母（防止AI回复 "答案是A" 这种情况）
            # 这里假设valid_answers的key就是A,B,C,D...
            valid_keys = valid_answers.keys()
            
            # 简单清洗：去除标点和空格，提取首字母
            cleaned_response = ai_response.strip().upper()
            
            # 如果回复是 "A" 或 "选A" 或 "A."，我们尝试提取核心字母
            final_choice = None
            for char in cleaned_response:
                if char in valid_keys:
                    final_choice = char
                    break
                    
            # 4. 验证逻辑
            if final_choice:
                # print(f"最终答案：{final_choice}")
                return final_choice
            else:
                return f"# 注释：AI的回复 '{ai_response}' 不在有效选项 {list(valid_keys)} 中，无法确定答案。"
        listAnswer= solve_riddle(question,answer)
        # 点击选项位置（拟人化高斯落点与延时）
        def clickBox(box):
            new_context = context.clone()
            target_x, target_y = HumanSimulator.get_gaussian_point(box)
            HumanSimulator.sleep_human(1.2, 0.2)
            click_job = new_context.tasker.controller.post_click(target_x, target_y)
            click_job.wait()  # 等待点击操作完成
            # 向用户ui界面输出日志info
            logger.info(f"AI返回答案：{listAnswer}。识别题目：{question}，识别答案列表：{answer}。落点坐标：({target_x}, {target_y})")
            HumanSimulator.sleep_human(1.5, 0.25)
            HumanSimulator.random_idle(chance=0.08, min_sec=2.0, max_sec=5.0)
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