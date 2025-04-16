import json
import ollama
from tqdm import tqdm
import os
import re 

# --- 配置 ---
INPUT_FILE = '../Sample_Set/NatS_20250407.json' #自然语料库，可以改成人工的
OUTPUT_FILE = '../output/predictions_NatS.json'
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

**请严格按以下步骤分析：**
## 一、核心推理原则

### 1. 谓词类型判定（强化识别例外）
- **假性谓词**：**暗示所述内容为虚构或不真实的指控**  
  典型词：**诬陷、污蔑、谎称、捏造、反咬、嫁祸、妄称**  
   **处理规则**：结论**否定指控内容** → **T**  
  **错误警示**：如背景使用“污蔑、诬陷”等假性词，**但结论确认了指控内容（如“确实是娼妓出身”）→ 必为 F**

- **真性谓词**：**对某内容表示确认、证实、认定为真**  
  典型词：**确认、证实、承认、清楚、检测显示、听出、看到**  
  处理规则：结论**肯定该内容** → **T**；否定 → **F**  

- **中性谓词**：**表达推测、个人观点、误认**  
  典型词：**以为、认为、怀疑、以为是、有人说、声称**  
  处理规则：**结论肯定该内容 → 默认 F**，除非背景中有独立佐证。  
  结论否定该内容 → 默认 F，除非有明确证据支持  
  **误区警示**：不要将“以为”当作真性谓词处理！

### 2. 否定穿透与双重否定规则
- “确认/清楚/听出 + 否定内容” → 实为**肯定否定事实**
  - 例：背景“听出他是韩国孩子” → 他是韩国孩子（= T）
- **结论双重否定处理：**
  - 奇数个否定 → 表示否定  
  - 偶数个否定 → 表示肯定  
  - 必须**转为肯定/否定语义后再进行匹配**。

---

## 二、事实提取与重述模板

| 类型 | 背景句关键词 | 实际事实（应提取为） |
|------|---------------|----------------------|
| 误判 | “听出是韩国孩子” | 是韩国孩子（真性谓词确认） |
| 误判 | “我讥讽她，她说我尖酸刻薄” | 我讥讽她（真性谓词确认） |
| 误判 | “污蔑她们是娼妓” | 她们不是娼妓（假性谓词） |
| 正确 | “诬陷我撞了他” | 我没撞他（假性谓词） |
| 正确 | “以为她丈夫以她为中心” | 是否以她为中心 = 未确认（中性）→ 任何结论都为 F |

---

## 三、关键判断模板（错误纠正类）

- 背景：“听出他是韩国孩子”  
  错误结论：“他不是韩国孩子” → **F**（听出 = 真性谓词）

- 背景：“诬陷我撞了他”  
  正确结论：“我没撞他” → **T**（诬陷 = 假性谓词）

- 背景：“她以为写书的都是圣人君子”  
  任意结论：“写书的都是 / 不是 圣人君子” → **U**（中性谓词 + 无佐证）

- 背景：“以为她丈夫以她为中心”  
  任意结论：“丈夫确实以 / 不以她为中心” → **F**（观点表达，无证据）

---
## 常见陷阱识别与规则（关键）

###  1. 误认误解类表达，一律标为 F

若背景中出现：**以为、误以为、误把……当成……、误解、还以为、看错了、把……错当成……** 等词汇：

- 说明背景表达的是**错误认知，不是真相本身**。
- 若结论 **肯定** 这种错误认知 → 判 `F`
- 若结论 **否定** 它，但背景没明说正确事实 → 判 `F` 或 `U`

**示例：**

| 背景 | 结论 | 标签 |
|-------|--------|-------|
| 他以为敌人投降了 | 敌人确实投降了 | `F` |
| 他们误把警察当成劫匪 | 警察确实是劫匪 | `F` |
| 他还以为那是她的男朋友 | 那个确实是她的男朋友 | `F` |

---

###  2. 中性表达 ≠ 事实支撑，一律标 F 或 U

出现词汇如：**羡慕、以为、觉得、笔者认为、我希望、令人感到、或许、怀疑** 等主观情绪/判断/建议：

- 它们不构成事实证据。
- 若结论句使用 “确实是/确实不是” 等确定性判断 → 属于过度推断 → 判 `F`

**示例：**

| 背景 | 结论 | 标签 |
|-------|--------|-------|
| 她说她羡慕别人的丈夫体贴 | 她丈夫确实很体贴 | `F` |
| 笔者认为这是最优选择 | 确实是最优选择 | `F` |

---

###  3. 未见 ≠ 不存在，一律标 U

背景中出现有限经验判断（**“未遇见”、“没听说”、“尚未发现”、“目前还没有”** 等）：

- 不代表客观事实。
- 若结论为 “确实不存在/没有发生” → 过度推断 → 判 `U`

**示例：**

| 背景 | 结论 | 标签 |
|-------|--------|-------|
| 我从没遇见有人能做到这点 | 确实没人能做到 | `U` |
| 我没听说过这种情况 | 这种情况确实没发生 | `U` |

---

###  4. 反转型表达需取反理解

出现词汇如：**诬陷、污蔑、谣传、误解、假装** 等反向表达：

- 表明原本判断是错的，要从反向理解。
- 若结论错把反转认知当作真相 → 判 `F`

**示例：**

| 背景 | 结论 | 标签 |
|-------|--------|-------|
| 他被诬陷为间谍 | 他确实是间谍 | `F` |
| 她被误解为冷漠 | 她确实冷漠 | `F` |

---

## 四、补充即时检验

在输出判断前，请完成以下 checklist：
1. 背景中是否明确陈述了支持/否定结论的事实？
2. 是否包含主观判断（如“以为”、“羡慕”、“我觉得”）？
3. 是否出现误解/误认/误判类表达？
4. 是否存在模糊性表述（如“可能”、“未见过”、“没听说”）？
5. 是否存在反向判断（如“假装、误解、污蔑”）？是否取反处理？

---

    **判断原则：**
    1.  严格按照文本推理的步骤进行分析。
    2.  分析背景句中的关键谓词（动词/形容词）。
    3.  必须进行常见陷阱识别与规则应用。
    4.  **尽可能做出 T 或 F 的判断：** 即使线索不完全充分，也要根据谓词的典型含义或背景句中最可能的暗示做出 T 或 F 的判断。
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