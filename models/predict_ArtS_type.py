import json
import ollama
from tqdm import tqdm
import os
import re 

# --- 配置 ---
INPUT_FILE = '../Sample_Set/Art_20250430_prompt.json' #自然语料库，可以改成人工的
OUTPUT_FILE = '../output/predictions_ArtS.json'
OLLAMA_MODEL = 'qwen3'

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


# --- 初始化 ---
try:
    client = ollama.Client()
    # 检查模型是否存在 (ollama python库本身不直接提供检查特定模型的功能，但调用 list 可以间接确认服务)
    client.list()
    print(f"Ollama 客户端初始化成功，将使用模型: {OLLAMA_MODEL}")
except Exception as e:
    print(f"错误：无法连接到 Ollama 或初始化客户端: {e}")
    print(f"请确保 Ollama 服务正在运行并且模型 '{OLLAMA_MODEL}' 可用。")
    exit()

# --- 提示词模板 ---
# ...existing code...

# --- 提示词模板 ---
PROMPT_TEMPLATE_COUNTER_FACTUAL = """
# 任务：叙实性判断 (反叙实)

## 指令
1.  **核心原则**：对于反叙实谓词 "{predicate}"（如“假装”、“想象”、“污蔑”、“以为(错误地认为)”），其后的内容在背景句的语境下被明确指示为 **假的** 或与事实相反。
2.  **推理步骤（关键！）**：
    a.  仔细阅读背景句，识别反叙实谓词 "{predicate}"。
    b.  明确提取出谓词后面所描述的 **“假的内容”**。
    c.  **进行逻辑取反**：推断出与“假的内容”完全相反的 **“隐含的真实情况”**。这是判断的基础！
    d.  将结论句 "{hypothesis}" 与你推断出的 **“隐含的真实情况”** 进行比较。
    e.  如果结论句与 **“隐含的真实情况”** 一致，则判断为 T。
    f.  如果结论句与 **“隐含的真实情况”** 矛盾，则判断为 F。
3.  **请一步一步进行推理**，特别是明确写出“假的内容”和“隐含的真实情况”，并根据上述定义输出最终判断 (T 或 F)。

## 输入
背景句：{text}
结论句：{hypothesis}
谓词：{predicate} (提示：这是一个反叙实谓词，如“妄称”、“假装”)

请一步一步进行推理，**不要输出 R 或其他任何文字。**, 你的回答 **只能是 T或者F **。
 ##  最终判断 (请仅输出 T或F):
"""
PROMPT_TEMPLATE_FACTUAL = """
# 任务：叙实性判断 (正叙实)

## 指令
1.  **核心原则**：对于正叙实谓词 "{predicate}"，其后的内容在背景句的语境下通常被视为 **事实**。
2.  **重点关注否定形式**：
    *   当谓词前有否定词（如“没有”、“不”）时，例如“小张 **没有** 意识到小李哭了”，这通常意味着主语“小张”**未察觉**或**未执行**谓词动作，但谓词后的内容“小李哭了”**仍然是事实**。你需要基于这个 **隐含的事实** 来判断结论句。

3.  **推理步骤**：
    a.  仔细分析背景句，识别谓词 "{predicate}" 以及它前面是否有否定词。
    b.  根据上述原则，确定背景句所 **预设或隐含的事实**。特别注意否定形式下的事实预设。
    c.  如果不是特殊情况，将提取出的事实与结论句 "{hypothesis}" 进行比较。
    d.  如果结论句与事实一致，则判断为 T。
    e.  如果结论句与事实矛盾，则判断为 F。
4.  **请一步一步进行推理**，并根据上述定义输出最终判断 (T 或 F)。

## 输入
背景句：{text}
结论句：{hypothesis}
谓词：{predicate} (提示：这是一个正叙实谓词)

请一步一步进行推理， **不要输出 R 或其他任何文字** ,你的回答 **只能是 T或者F **。

 ##  最终判断 (请仅输出 T或F ):
"""
PROMPT_TEMPLATE_NON_FACTUAL = """
# 任务：叙实性判断 (非叙实)

## 指令
1.  背景句中，谓词 "{predicate}" 表明其后的内容的真实性 **无法确定**。这可能是一个观点、信念、声明或猜测。
2.  由于背景句未提供足够信息判断谓词后内容的真假，因此通常无法判断结论句 "{hypothesis}" 的真假。
3.  **一步步进行推理**，解释为什么无法判断，并输出最终判断 (通常为 U)。

## 输入
背景句：{text}
结论句：{hypothesis}
谓词：{predicate} (提示：这是一个非叙实谓词，如“估计”、“相信”、“说”)

请一步一步进行推理，**不要输出 R 或其他任何文字。**, 你的回答 **只能是U **。
 ##  最终判断 (请仅输出U):
"""


# --- 预测 ---
results = []
original_answers = {} 
print(f"使用模型 '{OLLAMA_MODEL}' 处理 {len(data)} 条记录...")

for item in tqdm(data, desc="Predicting"):
    d_id = item.get('d_id')
    text = item.get('text')
    hypothesis = item.get('hypothesis')
    predicate = item.get('predicate')
    factivity_type = item.get('type')

    # 从样例集中获取原始答案用于后续计算准确率
    ground_truth = item.get('answer')

    if d_id is None or text is None or hypothesis is None:
        print(f"警告：跳过缺少 'd_id', 'text', 或 'hypothesis' 的记录: {item}")
        continue

    # 存储原始答案
    if ground_truth is not None:
         original_answers[d_id] = ground_truth

    # 可以选择性地在这里包含 predicate 和 type (如果需要)
    # predicate = item.get('predicate')
    # factivity_type = item.get('type')
    # prompt = PROMPT_TEMPLATE.format(text=text, hypothesis=hypothesis, predicate=predicate, type=factivity_type) # 如果模板包含这些字段
    if factivity_type == "正叙实":
        prompt = PROMPT_TEMPLATE_FACTUAL.format(text=text, hypothesis=hypothesis, predicate=predicate)
    elif factivity_type == "反叙实":
        prompt = PROMPT_TEMPLATE_COUNTER_FACTUAL.format(text=text, hypothesis=hypothesis, predicate=predicate)
    elif factivity_type == "非叙实":
        prompt = PROMPT_TEMPLATE_NON_FACTUAL.format(text=text, hypothesis=hypothesis, predicate=predicate)
    else:
        print(f"警告：未知的叙实类型 '{factivity_type}' (d_id: {d_id}). 跳过。")
        continue # 或者使用一个通用模板
   

    predicted_answer = 'Error' # 默认值以防 API 调用失败
    try:
        response = client.chat(model=OLLAMA_MODEL, messages=[
            {'role': 'user', 'content': prompt}
        ])
        answer_raw = response['message']['content'].strip()

        # --- 更鲁棒的答案解析逻辑 ---
        # 1. 尝试直接匹配单个大写字母 T/F/U/R
        if answer_raw in ['T', 'F', 'U', 'R']:
            predicted_answer = answer_raw
        else:
            # 2. 尝试在回答的开头或结尾查找 T/F/U/R (忽略大小写)
            #    使用正则表达式查找，优先匹配句首或句尾的单个字母
            match = re.search(r"^[TFUR]\b|\b[TFUR]$", answer_raw, re.IGNORECASE)
            if match:
                 predicted_answer = match.group(0).upper()
            else:
                # 3. 如果上面都找不到，可能模型没有按要求回答
                print(f"警告：无法从模型响应中明确解析出 T/F/U/R (d_id: {d_id}): '{answer_raw}'. 标记为 Invalid。")
                predicted_answer = 'Invalid' # 标记为无效回答

    except Exception as e:
        print(f"错误：调用 Ollama API 时出错 (d_id: {d_id}): {e}")
        predicted_answer = 'Error' # 标记 API 调用错误

    results.append({'d_id': d_id, 'answer': predicted_answer})

# --- 计算准确率 ---
correct_count = 0
t_f_u_prediction_count = 0 # 分母：模型做出了T/F/U预测的数量
r_prediction_count = 0     # 模型预测为 R 的数量
invalid_prediction_count = 0 # 模型回答无效的数量
error_count = 0            # API调用错误数量
missing_ground_truth = 0   # 缺少原始答案无法比对的数量
errors_context = []
if original_answers: # 只有当 original_answers 非空时（即处理的是样例集）才计算准确率
    print("\n正在计算准确率（基于样例集）...")
    for result in results:
        d_id = result['d_id']
        predicted_answer = result['answer']
        ground_truth = original_answers.get(d_id)

        if ground_truth is None:
            missing_ground_truth += 1
            continue # 无法比较

        if predicted_answer == 'Error':
            error_count += 1
            continue
        elif predicted_answer == 'Invalid':
            invalid_prediction_count += 1
            continue
        elif predicted_answer == 'R':
            r_prediction_count +=1
            # R 是否算对，取决于评测标准，这里我们先只统计 R 的数量
            # 如果需要将 R 算入准确率（即模型正确地拒绝回答）
            # if ground_truth == 'R':
            #     correct_count += 1
            #     t_f_u_r_prediction_count += 1 # 假设 R 也算有效预测
            # continue
        elif predicted_answer in ['T', 'F', 'U']:
             t_f_u_prediction_count += 1
             if predicted_answer == ground_truth:
                 correct_count += 1
             else:
                # 记录预测错误的上下文
                errors_context.append({
                    "d_id": d_id,
                    "text": next((item['text'] for item in data if item['d_id'] == d_id), "N/A"),
                    "hypothesis": next((item['hypothesis'] for item in data if item['d_id'] == d_id), "N/A"),
                    "predicted_answer": predicted_answer,
                    "ground_truth": ground_truth
                })

    if errors_context:
        print("\n--- 预测错误的上下文 ---")
        for error in errors_context:
            print(f"d_id: {error['d_id']}")
            print(f"背景句: {error['text']}")
            print(f"结论句: {error['hypothesis']}")
            print(f"模型预测: {error['predicted_answer']}")
            print(f"真实答案: {error['ground_truth']}")
            print("-" * 50)
           

    # 计算准确率（分母为 T/F/U 的预测总数）
    accuracy = (correct_count / t_f_u_prediction_count) * 100 if t_f_u_prediction_count > 0 else 0

    print(f"\n--- 准确率统计 (基于样例集) ---")
    print(f"总记录数: {len(data)}")
    print(f"处理记录数: {len(results)}")
    print(f"缺少原始答案的记录数: {missing_ground_truth}")
    print(f"API 调用错误数: {error_count}")
    print(f"模型回答无效数 (Invalid): {invalid_prediction_count}")
    print(f"模型预测为拒绝回答数 (R): {r_prediction_count}")
    print(f"模型做出 T/F/U 预测总数: {t_f_u_prediction_count}")
    print(f"其中正确预测数 (T/F/U): {correct_count}")
    print(f"准确率 (Correct T/F/U / Total T/F/U Predictions): {accuracy:.2f}%")
else:
    print("\n未找到原始答案（可能处理的是测试集），跳过准确率计算。")
    print(f"--- 预测统计 ---")
    error_count = sum(1 for r in results if r['answer'] == 'Error')
    invalid_prediction_count = sum(1 for r in results if r['answer'] == 'Invalid')
    r_prediction_count = sum(1 for r in results if r['answer'] == 'R')
    t_f_u_prediction_count = sum(1 for r in results if r['answer'] in ['T', 'F', 'U'])
    print(f"总记录数: {len(data)}")
    print(f"处理记录数: {len(results)}")
    print(f"API 调用错误数: {error_count}")
    print(f"模型回答无效数 (Invalid): {invalid_prediction_count}")
    print(f"模型预测为拒绝回答数 (R): {r_prediction_count}")
    print(f"模型做出 T/F/U 预测总数: {t_f_u_prediction_count}")


# --- 保存结果 ---
try:
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\n预测结果已保存到 '{OUTPUT_FILE_PATH}'")
except IOError as e:
    print(f"错误：无法写入输出文件 '{OUTPUT_FILE_PATH}': {e}")
except Exception as e:
    print(f"保存文件时发生未知错误: {e}")


print("处理完成。")