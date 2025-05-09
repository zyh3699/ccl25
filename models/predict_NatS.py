import json
# import ollama # 移除 ollama
import google.generativeai as genai # 导入 Gemini SDK
from tqdm import tqdm
import os
import re

# --- 配置 ---
INPUT_FILE = '../Sample_Set/NatS_20250407.json' #自然语料库，可以改成人工的
OUTPUT_FILE = '../output/predictions_NatS.json'
# OLLAMA_MODEL = 'qwen2:1.5b' # 移除 Ollama 模型配置

# Gemini API 配置
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL_NAME = 'gemini-2.5-pro-exp-03-25' # 您可以根据需要更改为其他 Gemini 模型，例如 'gemini-1.5-flash-latest'

if not GEMINI_API_KEY:
    print("错误：GEMINI_API_KEY 环境变量未设置。请设置该变量后重试。")
    exit()

# 获取脚本所在的目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE_PATH = os.path.join(SCRIPT_DIR, INPUT_FILE)
OUTPUT_FILE_PATH = os.path.join(SCRIPT_DIR, OUTPUT_FILE)


# --- 导数据 ---
try:
    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"错误：输入文件 '{INPUT_FILE_PATH}' 未找到。")
    exit()
except json.JSONDecodeError:
    print(f"错误：无法解码 '{INPUT_FILE_PATH}' 中的 JSON。")
    exit()
except Exception as e:
    print(f"加载文件时发生未知错误: {e}")
    exit()


# --- 初始化 Gemini API ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        GEMINI_MODEL_NAME,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2 #较低的温度有助于分类任务获得更一致的输出
        )
        # 您可以在此处添加 safety_settings 如果需要调整安全级别
        # safety_settings=[
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        # ]
    )
    print(f"Gemini API 初始化成功，将使用模型: {GEMINI_MODEL_NAME}")
except Exception as e:
    print(f"错误：无法初始化 Gemini API 或模型: {e}")
    exit()

# --- 提示词模板 ---
PROMPT_TEMPLATE = """

好的，基于我们之前的讨论和所有样例的学习，如果我现在要重写一个更完善、更细致的提示词模板，我会尝试包含以下方面：

---

## **任务：精细化叙实性判断 (输出 T, F, U)**

### **最高指令：**
你将获得三项信息：**背景句 (P)**、**结论句 (H)**、**谓词 (V)**。
你的核心任务是：**严格依据背景句(P)提供的信息，判断结论句(H)的真实性。谓词(V)是帮助你定位P中关键信息和说话者态度的线索。**
请进行详细的逻辑推理，并最终输出 T（真）、F（假）或 U（无法判断）。

---

### **一、核心推理框架**

1.  **理解谓词 (V) 的性质及其对背景句中目标陈述 (S) 真实性的暗示：**
    *   在背景句(P)中，谓词(V)通常会引出或关联一个目标陈述或事件，我们称之为 **S**。
    *   你的首要任务是判断 **S 在背景句(P)的语境下是真是假，还是不确定**。

2.  **谓词分类与S的真实性判断：**

    *   **A. 强事实性谓词 (Strongly Factive Predicates):**
        *   **示例:** 看见、发现、知道、意识到、清楚、听出、证实、目睹、揭露、获悉、披露、记得（某事发生）、（正确）猜到、领悟到、注意到、承认（事实）。
        *   **规则:** 这类谓词强烈预设其关联的 **S** 在背景句(P)的语境下为 **真 (TRUE)**。
        *   *即使P中对这类谓词本身进行了否定（如“他没有意识到S”），S（如“他错了”）通常仍然被预设为 TRUE。特别注意这种结构。*

    *   **B. 强反事实性谓词 (Strongly Counter-factive Predicates):**
        *   **示例:** 谎称、假装、幻想、污蔑、诬陷、妄称。
        *   **规则:** 这类谓词强烈暗示其关联的 **S** 在背景句(P)的语境下为 **假 (FALSE)**。

    *   **C. 主观/非事实性/弱指示性谓词 (Subjective/Non-factive/Weakly Indicative Predicates):**
        *   **示例:**
            *   **表达观点/主张/信念:** 认为、声称、表示、说、主张、相信、声明、重申、感觉（到）、以为、估计、猜、猜测、怀疑、指责、批评、抱怨、埋怨、哀叹、感叹、吹嘘。
            *   **表达情感/态度（S的真实性常依赖说话者主观感受或未经验证的判断）:** 后悔（某事发生为真，但后悔的情感是主观的）、羡慕（被羡慕的对象状态通常为真，但羡慕本身是主观的）。
        *   **规则:**
            *   **默认情况:** 这类谓词引导的 **S** 的真实性**本身是不确定的 (UNCERTAIN)**，它仅仅是某人的言论、观点、猜测、情感或未经验证的断言。
            *   **重要例外与细化：**
                *   **"以为A，但/却/后来发现/实际上是B"结构：** 此结构中，A通常被视为说话者曾经的错误认知（**S<sub>A</sub> 为 FALSE**），而B是修正后的正确认知（**S<sub>B</sub> 为 TRUE**）。
                *   **"后悔"：** "后悔做了S" 或 "后悔没做S"。 "做了S" 或 "没做S" 本身是 **TRUE** 的。结论句如果是关于S是否发生，则为T/F。如果结论句是关于后悔这种情感本身，则可能为U。
                *   **"羡慕"：** "羡慕X拥有Y"。 "X拥有Y" 通常在语境中是 **TRUE** 的。
                *   **"哀叹/抱怨/批评/指责S"：** S的真实性**高度依赖上下文**。
                    *   如果P中仅提到某人哀叹/抱怨/批评S，而无其他佐证，则S的真实性为 **UNCERTAIN**。
                    *   如果P中有其他信息强烈暗示S是基于事实的（如`Nat_0001`），则S可能为 **TRUE**。
                    *   如果P中有信息反驳S，则S可能为 **FALSE** 或 **UNCERTAIN**。
                *   **"感觉/觉察出/觉出/觉得/觉着S"：**
                    *   如果S描述的是**客观可验证的事实或状态**，并且P中没有转折或否定，那么S通常被认为是说话者在P语境下的真实感知，S可视为 **TRUE**。
                    *   如果S描述的是**纯粹的主观感受或不确定的推断**（如“他感觉会下雨”），则S的真实性是 **UNCERTAIN**。
                *   **"怀疑S"：** S的真实性为 **UNCERTAIN**。除非背景句明确证实或证伪了怀疑的内容。
                *   **"承认S"：** 如果是“承认（某个事实或错误）”，则S为 **TRUE**。
                *   **"听说"：** 引导的S真实性为 **UNCERTAIN**，除非P中有其他信息证实。
                *   **"声言/声明"：** 引导的S真实性为 **UNCERTAIN**，除非P中有其他信息证实。

3.  **比较S的真实性与结论句(H)：**

    *   **如果 S 被判定为 TRUE：**
        *   若 H **肯定或等同于 S** (或其逻辑推论) -> 输出 **T**。
        *   若 H **否定或与 S 矛盾** -> 输出 **F**。

    *   **如果 S 被判定为 FALSE：**
        *   若 H **肯定或等同于 S** (即声称这个虚假的事情是真的) -> 输出 **F**。
        *   若 H **否定或与 S 矛盾** (即声称这个虚假的事情是假的，从而陈述了事实) -> 输出 **T**。

    *   **如果 S 被判定为 UNCERTAIN：**
        *   若 H 是关于 S 本身的真实性判断 -> 输出 **U**。
        *   若 H 是关于说话者是否持有该观点/情感（且P中明确表达了），则可能为T，但这通常不是本任务的考察点。**主要关注S的真实性。**

---

### **二、重要注意事项与启发式规则**

1.  **严格依赖背景句(P)：** 你的判断**唯一依据**是背景句提供的信息。不要引入外部知识或个人推断。如果背景句没有明确说明或强烈暗示，则无法判断。

2.  **关注否定词与转折词：**
    *   P或H中的“不”、“没有”、“未”等否定词会反转判断。
    *   P中的“但是”、“然而”、“却”、“实际上”、“后来才发现”等转折词往往引出与前述内容相反或修正的事实。

3.  **“U”的倾向性：** 对于主观/非事实性谓词引导的S，除非背景句有**明确的、独立的佐证信息**来证实或证伪S，否则**倾向于输出U**。仅仅是某人的“认为”、“感觉”、“声称”、“抱怨”等，不足以将其内容S判定为T或F。

4.  **聚焦核心事件：** 结论句(H)可能只涉及目标陈述(S)的一部分。确保你判断的是H所指的具体内容。

5.  **区分“事件本身”与“对事件的描述/态度”：**
    *   例如：“他后悔<ins>打了人</ins>。” -> “打了人”是事实(T)。“他是否真的后悔”可能是U（除非背景句有更多信息）。
    *   本任务通常是判断“打了人”这个事件的真实性。

6.  **考虑句式和语气：**
    *   疑问句或反问句引导的内容，其真实性需要结合上下文判断。
    *   “难道...不...吗？”通常暗示肯定。
    *   “无非是”、“只不过是”等可能削弱或限定其后内容的强度。

7.  **处理复杂句：** 如果S本身包含多个子句或条件，确保H与P中S的对应部分完全匹配或逻辑一致/矛盾。

8.  **从样例中学习细微差别：** 不同的主观/非事实性谓词在实际语境中，其引导S为U的强度可能不同。通过分析大量样例，可以更好地把握这种细微差别。例如，虽然“哀叹”通常引导U，但在特定强上下文中（如`Nat_0001`），其引导的内容可能更接近事实。**但如果没有强上下文佐证，保守给出U。**

---

### **三、输出格式**

*   **仅输出 T, F, 或 U 中的一个字母。**

---

**现在，请基于以上更精细的规则，重新评估所有输入并给出判断。**
**一步步进行推理，不要输出 R 或其他任何文字。**
""" # 提示词模板结束

# --- 预测 ---
results = []
original_answers = {}
print(f"使用模型 '{GEMINI_MODEL_NAME}' 处理 {len(data)} 条记录...")
for item in tqdm(data, desc="Predicting"):
    d_id = item.get('d_id')
    text = item.get('text') # 背景句 P
    hypothesis = item.get('hypothesis') # 结论句 H
    predicate_val = item.get('predicate', "N/A") # 谓词 V, 如果没有则为 N/A

    ground_truth = item.get('answer')

    if d_id is None or text is None or hypothesis is None:
        print(f"警告：跳过缺少 'd_id', 'text', 或 'hypothesis' 的记录: {item}")
        results.append({'d_id': d_id, 'answer': 'Error_Missing_Data'})
        continue

    if ground_truth is not None:
        original_answers[d_id] = ground_truth

    # 构建完整的提示，将实际数据附加到指令模板后
    full_prompt = f"""{PROMPT_TEMPLATE}

---
**请根据以上规则，对以下内容进行判断：**

背景句 (P): {text}
结论句 (H): {hypothesis}
谓词 (V): {predicate_val}
---
**你的判断 (仅输出 T, F, 或 U):**
"""

    predicted_answer = 'Error' # 默认值，以防 API 调用或解析失败
    try:
        gemini_response = model.generate_content(full_prompt)
        
        answer_raw = ""
        
        # 检查是否有即时阻塞 (prompt_feedback)
        if gemini_response.prompt_feedback and gemini_response.prompt_feedback.block_reason:
            block_reason_name = gemini_response.prompt_feedback.block_reason.name if hasattr(gemini_response.prompt_feedback.block_reason, 'name') else str(gemini_response.prompt_feedback.block_reason)
            block_reason_msg = gemini_response.prompt_feedback.block_reason_message or block_reason_name
            print(f"警告：Gemini API 请求因 prompt_feedback 被阻止 (d_id: {d_id})。原因: {block_reason_msg}")
            # predicted_answer 保持为 'Error'
        
        # 检查候选答案
        elif gemini_response.candidates:
            candidate = gemini_response.candidates[0]
            is_safe_content = True

            # 检查内容是否因安全原因被阻止
            if candidate.finish_reason and hasattr(candidate.finish_reason, 'name') and candidate.finish_reason.name == "SAFETY":
                is_safe_content = False
                print(f"警告：Gemini API 内容因 SAFETY 被阻止 (d_id: {d_id}).")
                if candidate.safety_ratings:
                    for rating in candidate.safety_ratings:
                        print(f"  - Category: {rating.category.name}, Probability: {rating.probability.name}")
                # predicted_answer 保持为 'Error'

            # 如果内容安全且存在，则提取文本
            if is_safe_content and candidate.content and candidate.content.parts:
                answer_raw = "".join(part.text for part in candidate.content.parts).strip()
            
            if answer_raw: # 如果成功获取到一些原始文本
                if answer_raw in ['T', 'F', 'U', 'R']: # 直接匹配
                    predicted_answer = answer_raw
                else:
                    # 尝试用正则从开头或结尾提取 T/F/U/R
                    match = re.search(r"^[TFUR]\b|\b[TFUR]$", answer_raw, re.IGNORECASE)
                    if match:
                        predicted_answer = match.group(0).upper()
                    else:
                        print(f"警告：无法从模型响应 '{answer_raw}' 中明确解析出 T/F/U/R (d_id: {d_id}). 标记为 Invalid。")
                        predicted_answer = 'Invalid' # 无法解析，标记为无效
            elif is_safe_content and predicted_answer == 'Error': # 没有提取到文本，但内容是安全的
                 print(f"警告：Gemini API 返回了空内容或无法提取文本 (d_id: {d_id}). Candidate finish_reason: {candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else candidate.finish_reason}")
                 # predicted_answer 保持为 'Error'

        else: # Gemini API 没有返回候选答案
            print(f"警告：Gemini API 未返回候选答案 (d_id: {d_id})。完整响应: {gemini_response}")
            # predicted_answer 保持为 'Error'

    except Exception as e:
        print(f"错误：调用 Gemini API 时发生异常 (d_id: {d_id}): {e}")
        # predicted_answer 已经默认为 'Error'

    results.append({'d_id': d_id, 'answer': predicted_answer})

# --- 计算准确率 --- (这部分代码保持不变)
correct_count = 0
t_f_u_prediction_count = 0 
r_prediction_count = 0     
invalid_prediction_count = 0 
error_count = 0            
missing_ground_truth = 0   
errors_context = []
if original_answers: 
    print("\n正在计算准确率（基于样例集）...")
    for result in results:
        d_id = result['d_id']
        predicted_answer = result['answer']
        ground_truth = original_answers.get(d_id)

        if ground_truth is None:
            missing_ground_truth += 1
            continue 

        if predicted_answer == 'Error' or predicted_answer == 'Error_Missing_Data': # 包含了数据缺失的错误
            error_count += 1
            continue
        elif predicted_answer == 'Invalid':
            invalid_prediction_count += 1
            continue
        elif predicted_answer == 'R':
            r_prediction_count +=1
        elif predicted_answer in ['T', 'F', 'U']:
             t_f_u_prediction_count += 1
             if predicted_answer == ground_truth:
                 correct_count += 1
             else:
                errors_context.append({
                    "d_id": d_id,
                    "text": next((item['text'] for item in data if item.get('d_id') == d_id), "N/A"),
                    "hypothesis": next((item['hypothesis'] for item in data if item.get('d_id') == d_id), "N/A"),
                    "predicate": next((item.get('predicate') for item in data if item.get('d_id') == d_id), "N/A"),
                    "predicted_answer": predicted_answer,
                    "ground_truth": ground_truth
                })

    if errors_context:
        print("\n--- 预测错误的上下文 ---")
        for error_item in errors_context: # Renamed 'error' to 'error_item' to avoid conflict
            print(f"d_id: {error_item['d_id']}")
            print(f"背景句: {error_item['text']}")
            print(f"结论句: {error_item['hypothesis']}")
            print(f"谓词: {error_item['predicate']}")
            print(f"模型预测: {error_item['predicted_answer']}")
            print(f"真实答案: {error_item['ground_truth']}")
            print("-" * 50)
           
    accuracy = (correct_count / t_f_u_prediction_count) * 100 if t_f_u_prediction_count > 0 else 0

    print(f"\n--- 准确率统计 (基于样例集) ---")
    print(f"总记录数: {len(data)}")
    print(f"处理记录数: {len(results)}")
    print(f"缺少原始答案的记录数: {missing_ground_truth}")
    print(f"API 调用或数据错误数: {error_count}")
    print(f"模型回答无效数 (Invalid): {invalid_prediction_count}")
    print(f"模型预测为拒绝回答数 (R): {r_prediction_count}")
    print(f"模型做出 T/F/U 预测总数: {t_f_u_prediction_count}")
    print(f"其中正确预测数 (T/F/U): {correct_count}")
    print(f"准确率 (Correct T/F/U / Total T/F/U Predictions): {accuracy:.2f}%")
else:
    print("\n未找到原始答案（可能处理的是测试集），跳过准确率计算。")
    print(f"--- 预测统计 ---")
    error_count = sum(1 for r in results if r['answer'] == 'Error' or r['answer'] == 'Error_Missing_Data')
    invalid_prediction_count = sum(1 for r in results if r['answer'] == 'Invalid')
    r_prediction_count = sum(1 for r in results if r['answer'] == 'R')
    t_f_u_prediction_count = sum(1 for r in results if r['answer'] in ['T', 'F', 'U'])
    print(f"总记录数: {len(data)}")
    print(f"处理记录数: {len(results)}")
    print(f"API 调用或数据错误数: {error_count}")
    print(f"模型回答无效数 (Invalid): {invalid_prediction_count}")
    print(f"模型预测为拒绝回答数 (R): {r_prediction_count}")
    print(f"模型做出 T/F/U 预测总数: {t_f_u_prediction_count}")

# --- 保存结果 --- (这部分代码保持不变)
try:
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\n预测结果已保存到 '{OUTPUT_FILE_PATH}'")
except IOError as e:
    print(f"错误：无法写入输出文件 '{OUTPUT_FILE_PATH}': {e}")
except Exception as e:
    print(f"保存文件时发生未知错误: {e}")

print("处理完成。")