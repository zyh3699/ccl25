import json
import ollama
from tqdm import tqdm
import os
import re 

# --- 配置 ---
INPUT_FILE = '../Sample_Set/ArtS_20250325.json' #自然语料库，可以改成人工的
OUTPUT_FILE = '../output/predictions_vote.json'
OLLAMA_MODEL = 'qwen2:7b'

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
PROMPT_TEMPLATE = """

# 任务：叙实性判断 (必须输出 T 或 F, 除非绝对无法判断)

## 指令
仔细阅读“背景句”和“结论句”。  
你的任务是严格依据“背景句”判断“结论句”的真实性，并且必须从 'T', 'F', 'U' 三个选项中给出一个。"T": "真",
            "F": "假",
            "U": "不能确定",
  **请一步步进行推理并得出结论**

## 示例

输入：
背景句: 小张没说不会妥协。
结论句: 小张不会妥协。
推理过程：
理解背景：识别出小张没有明确表示自己永远不会妥协的事实。
分析结论：认识到结论提出了一个强烈的肯定陈述，即小张永远不妥协。
比较信息：对比了这两个声明之间的差异和联系。
评估充分性：注意到背景信息不足以支持这样的绝对结论。
确定正确性：最终判断为“无法确定”（U），因为现有证据不足以验证或否定这个结论的准确性。
因此：输出最终判断为 U。
输出: U

    ## 输入
    背景句：{text}
    结论句：{hypothesis}

    请一步一步进行推理，**不要输出 R 或其他任何文字。** 你的回答 **只能是 T, F, 或 U 中的一个字母**。
    ##  最终判断 (请仅输出 T, F, 或 U):
    """

# --- 预测 ---
results = []
original_answers = {} 
print(f"使用模型 '{OLLAMA_MODEL}' 处理 {len(data)} 条记录...")

# --- 配置 ---
NUM_SAMPLES = 5  # 每次生成的样本数量

# # --- 修改预测逻辑 ---
# for item in tqdm(data, desc="Predicting"):
#     d_id = item.get('d_id')
#     text = item.get('text')
#     hypothesis = item.get('hypothesis')

#     # 从样例集中获取原始答案用于后续计算准确率
#     ground_truth = item.get('answer')

#     if d_id is None or text is None or hypothesis is None:
#         print(f"警告：跳过缺少 'd_id', 'text', 或 'hypothesis' 的记录: {item}")
#         continue

#     # 存储原始答案
#     if ground_truth is not None:
#         original_answers[d_id] = ground_truth

#     prompt = PROMPT_TEMPLATE.format(text=text, hypothesis=hypothesis)

#     # 多数投票逻辑
#     votes = {'T': 0, 'F': 0, 'U': 0}
#     predicted_answer = 'Error'  # 默认值以防 API 调用失败

#     try:
#         for _ in range(NUM_SAMPLES):
#             response = client.chat(model=OLLAMA_MODEL, messages=[
#                 {'role': 'user', 'content': prompt}
#             ])
#             answer_raw = response['message']['content'].strip()

#             # --- 答案解析逻辑 ---
#             # 1. 尝试直接匹配单个大写字母 T/F/U/R
#             if answer_raw in ['T', 'F', 'U', 'R']:
#                 answer = answer_raw
#             else:
#                 # 2. 尝试在回答的开头或结尾查找 T/F/U/R (忽略大小写)
#                 match = re.search(r"^[TFUR]\b|\b[TFUR]$", answer_raw, re.IGNORECASE)
#                 if match:
#                     answer = match.group(0).upper()
#                 else:
#                     # 3. 如果无法解析出答案，标记为 Invalid
#                     print(f"警告：无法从模型响应中明确解析出 T/F/U/R (d_id: {d_id}): '{answer_raw}'. 标记为 Invalid。")
#                     answer = 'Invalid'

#             # 统计有效投票
#             if answer in votes:
#                 votes[answer] += 1

#         # 根据投票结果确定最终答案
#         predicted_answer = max(votes, key=votes.get)  # 选择票数最多的选项
#         if votes[predicted_answer] == 0:
#             predicted_answer = 'Invalid'  # 如果没有有效投票，标记为无效

#     except Exception as e:
#         print(f"错误：调用 Ollama API 时出错 (d_id: {d_id}): {e}")
#         predicted_answer = 'Error'  # 标记 API 调用错误

#     results.append({'d_id': d_id, 'answer': predicted_answer})
answers_per_data = {item['d_id']: [] for item in data}  # 存储每条数据的五次答案

for round_idx in range(NUM_SAMPLES):  # 循环五轮
    print(f"\n--- 第 {round_idx + 1} 轮预测 ---")
    for item in tqdm(data, desc=f"Round {round_idx + 1}"):
        d_id = item.get('d_id')
        text = item.get('text')
        hypothesis = item.get('hypothesis')
        # 从样例集中获取原始答案用于后续计算准确率
        ground_truth = item.get('answer')

        if d_id is None or text is None or hypothesis is None:
            print(f"警告：跳过缺少 'd_id', 'text', 或 'hypothesis' 的记录: {item}")
            continue

        # 存储原始答案
        if ground_truth is not None:
            original_answers[d_id] = ground_truth

        if d_id is None or text is None or hypothesis is None:
            print(f"警告：跳过缺少 'd_id', 'text', 或 'hypothesis' 的记录: {item}")
            continue

        prompt = PROMPT_TEMPLATE.format(text=text, hypothesis=hypothesis)

        try:
            response = client.chat(model=OLLAMA_MODEL, messages=[
                {'role': 'user', 'content': prompt}
            ])
            answer_raw = response['message']['content'].strip()

            # --- 答案解析逻辑 ---
            if answer_raw in ['T', 'F', 'U', 'R']:
                answer = answer_raw
            else:
                match = re.search(r"^[TFUR]\b|\b[TFUR]$", answer_raw, re.IGNORECASE)
                if match:
                    answer = match.group(0).upper()
                else:
                    print(f"警告：无法从模型响应中明确解析出 T/F/U/R (d_id: {d_id}): '{answer_raw}'. 标记为 Invalid。")
                    answer = 'Invalid'

            # 存储答案
            answers_per_data[d_id].append(answer)

        except Exception as e:
            print(f"错误：调用 Ollama API 时出错 (d_id: {d_id}): {e}")
            answers_per_data[d_id].append('Error')  # 标记 API 调用错误

# --- 对每条数据进行投票 ---
for item in data:
    d_id = item['d_id']
    votes = {'T': 0, 'F': 0, 'U': 0, 'R': 0, 'Invalid': 0, 'Error': 0}
    for answer in answers_per_data[d_id]:
        if answer in votes:
            votes[answer] += 1

    # 根据投票结果确定最终答案
    predicted_answer = max(votes, key=votes.get)  # 选择票数最多的选项
    if votes[predicted_answer] == 0:
        predicted_answer = 'Invalid'  # 如果没有有效投票，标记为无效

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