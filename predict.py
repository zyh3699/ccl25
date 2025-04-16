import json
import ollama
from tqdm import tqdm
import os
import re 

# --- 配置 ---
INPUT_FILE = 'Sample_Set/NatS_20250407.json' #自然语料库，可以改成人工的
OUTPUT_FILE = 'predictions.json'
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
**你的唯一任务是判断“结论句”的真实性，并且必须从 'T', 'F', 'U' 三个选项中给出一个。**
**严格依据“背景句”进行判断。**

**示例执行过程**
1. 背景句："小张假装小李已经离开了，但实际上小李还在房间里。"
2. 结论句："小李已经离开了。"
3.  思维链推理过程：1. 背景句中的谓词是“假装”，表示小张的行为并不反映事实，而是虚假的表现。
2. 背景句进一步明确说明“小李还在房间里”，这是一个直接的事实陈述。
3. 结论句假设“小李已经离开了”，这与背景句中明确的事实“小李还在房间里”相矛盾。
4. 因此，根据背景句的内容，结论句为假。

## 一、核心推理步骤
1. **关键谓词定位**  
   - 提取背景句中表达认知/行为的核心谓词（动词/形容词）

2. **谓词语义分类**  
   ```mermaid
   graph TD
   A[谓词类型] --> B[真性谓词]
   A --> C[假性谓词] 
   A --> D[中性谓词]
   B -->|典型词| E[知道, 承认, 证实...]
   C -->|典型词| F[诬陷, 假装, 谎称...]
   D -->|典型词| G[认为, 声称, 怀疑...]
   if 存在直接事实陈述（如"实际上..."）:
    return 事实对应的T/F
elif 谓词类型 == 真性谓词:
    return T（除非有明确反驳）
elif 谓词类型 == 假性谓词:
    return F（除非有明确证实）
else:
    if 背景句有支持/反驳线索:
        return 对应T/F
    else:
        return 语境倾向判断

**判断原则：**
1.  仿照核心推理步骤，进行思维链推理，判断结论句的真实性。
2.  分析背景句中的关键谓词（动词/形容词）。
3.  **尽可能做出 T 或 F 的判断：** 即使线索不完全充分，也要根据谓词的典型含义或背景句中最可能的暗示做出 T 或 F 的判断。
4.  **极少使用 U: ** **只有当背景句信息完全缺失或存在直接矛盾，导致在逻辑上绝对无法做出任何倾向性判断时，才允许使用 'U'。** 否则，必须选择 T 或 F。
5.  **不要输出 R 或其他任何文字。** 你的回答 **只能是 T, F, 或 U 中的一个字母**。

## 输入
背景句：{text}
结论句：{hypothesis}

## 最终判断 (请仅输出 T, F, 或 U):
"""

# --- 预测 ---
results = []
original_answers = {} 
print(f"使用模型 '{OLLAMA_MODEL}' 处理 {len(data)} 条记录...")

for item in tqdm(data, desc="Predicting"):
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

    # 可以选择性地在这里包含 predicate 和 type (如果需要)
    # predicate = item.get('predicate')
    # factivity_type = item.get('type')
    # prompt = PROMPT_TEMPLATE.format(text=text, hypothesis=hypothesis, predicate=predicate, type=factivity_type) # 如果模板包含这些字段

    prompt = PROMPT_TEMPLATE.format(text=text, hypothesis=hypothesis)

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