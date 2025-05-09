import json
import random
import ollama
from tqdm import tqdm
import os
import re 

# --- 配置 ---
INPUT_FILE = '../Sample_Set/Nat_20250430_prompt_1.json' #自然语料库，可以改成人工的
OUTPUT_FILE = '../output/predictions_NatS_predicate_2.0.json'
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
    
# if len(data) > 100:
#     print(f"从 {len(data)} 条记录中随机抽取 100 条进行评测...")
#     data = random.sample(data, 100)
#     print(f"已抽取 100 条数据。")
# elif len(data) > 0:
#     print(f"数据记录总数 ({len(data)}) 不足或等于 100 条，将处理所有数据...")
# else:
#     print("错误：数据集中没有记录可供处理。")
#     exit()

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
# 任务：叙实性判断（输出 T, F 或 U）

## 指令
你将获得三项信息：背景句、结论句、谓词。
你的任务是：严格依据“背景句”判断“结论句”的真实性，并利用“谓词”辅助聚焦判断焦点。
**请一步步进行推理并得出结论。**

---
### 输出要求：
- 只能从以下三项中选择一个最终输出：
  - T（结论句为真：根据背景句可以明确断定结论句是事实）
  - F（结论句为假：根据背景句可以明确断定结论句与事实相反）
  - U（无法判断真伪：背景句未提供足够信息来明确判断结论句的真伪）

---
### 推理核心步骤：

**第一步：理解谓词的性质及其对背景句中相关事件真实性的暗示。**

*   **A. 事实性谓词 (Factive Predicates):** 如 看见、发现、知道、意识到、清楚、听出、羡慕、后悔、证实、遇见（某事发生）。
    *   这类谓词预设其关联的事件/陈述（称之为 **S**）在背景句的语境下为 **真 (TRUE)**。
    *   *例如：“他意识到<ins>自己错了</ins>。” -> S = “自己错了” 是 TRUE。*
    *   *即使背景句对事实性谓词本身进行否定（如“他没有意识到<ins>自己错了</ins>”），S（“自己错了”）通常仍然被预设为 TRUE。*

*   **B. 反事实性谓词 (Counter-factive Predicates):** 如 谎称、假装、幻想、污蔑、诬陷。
    *   这类谓词表明其关联的事件/陈述（称之为 **S**）在背景句的语境下为 **假 (FALSE)**。
    *   *例如：“他谎称<ins>自己去了北京</ins>。” -> S = “自己去了北京” 是 FALSE。*

*   **C. 非事实性谓词 (Non-factive Predicates):** 如 声称、表示、认为、说、主张、建议、指责、以为、猜、估计、相信、怀疑、希望。
    *   这类谓词表明其关联的事件/陈述（称之为 **S**）的真实性**本身是不确定的 (UNCERTAIN)**，仅仅是某人的言论、观点、猜测或信念。
    *   **特别注意“以为”这类词：**
        *   如果背景句表述为“某人以为S”，且句子中**没有**后续信息修正或否定S，则S的真实性为UNCERTAIN。
        *   如果背景句表述为“某人以为S，但/却/后来发现/实际上是Y”，这通常意味着S是**FALSE**，而Y是**TRUE**。

**第二步：将结论句与第一步中判定的事件S的真实性进行比较。**

*   **1. 如果S被判定为 TRUE (来自事实性谓词，或“以为...但...”结构中的修正事实Y)：**
    *   若结论句**肯定或等同于S** -> 输出 **T**。
        *   *例：背景句：“他意识到<ins>自己错了</ins>”(S="自己错了"为TRUE)。结论句：“他确实错了。” -> T。*
    *   若结论句**否定或与S矛盾** -> 输出 **F**。
        *   *例：背景句：“他意识到<ins>自己错了</ins>”(S="自己错了"为TRUE)。结论句：“他确实没错。” -> F。*
        *   *例: 背景句: "...听出，<ins>我方多次占领阵地</ins>..." (S="我方多次占领阵地"为TRUE)。结论句: "我方确实没有占领到阵地。" (否定S) -> F。*
        *   *例: 背景句: "陈芳听出<ins>我在讥讽她</ins>..." (S="我在讥讽她"为TRUE)。结论句: "我确实没有在讥讽她。" (否定S) -> F。*

*   **2. 如果S被判定为 FALSE (来自反事实性谓词，或“以为S，但...”结构中的S)：**
    *   若结论句**肯定或等同于S**（即声称这个虚假的事情是真的） -> 输出 **F**。
        *   *例：背景句：“他谎称<ins>自己去了北京</ins>”(S="自己去了北京"为FALSE)。结论句：“他确实去了北京。” -> F。*
        *   *例: 背景句: "...污蔑<ins>法国政府与伊拉克萨达姆政权之间有同谋关系</ins>..." (S="法国...有同谋关系"为FALSE)。结论句: "法国政府与伊拉克萨达姆政权之间确实有同谋关系。" (肯定S) -> F。*
    *   若结论句**否定或与S矛盾**（即声称这个虚假的事情是假的，从而陈述了事实） -> 输出 **T**。
        *   *例：背景句：“他谎称<ins>自己去了北京</ins>”(S="自己去了北京"为FALSE)。结论句：“他其实没去北京。” -> T。*
        *   *例: 背景句: "...污蔑<ins>李秀英是“假”证人</ins>。" (S="李秀英是假证人"为FALSE)。结论句: "李秀英确实不是假证人。" (否定S，即李秀英是真的证人) -> T。*

*   **3. 如果S被判定为 UNCERTAIN (来自非事实性谓词，且背景句无进一步佐证或修正)：**
    *   若结论句是关于S本身的真实性判断 -> 输出 **U**。
        *   *例：背景句：“他声称<ins>那是个好主意</ins>。”(S="那是个好主意"为UNCERTAIN)。结论句：“那确实是个好主意。” -> U。*
        *   *例: 背景句: "笔者以为，<ins>这不失为正党风、扬正气的好办法</ins>..." (S="这不失为好办法"为UNCERTAIN)。结论句: "这确实是正党风、扬正气的好办法。" -> U。*
        *   *例: 背景句: "...并未遇见到<ins>任何机器具有这些特性</ins>。" (说话者“未遇见”是事实，但“任何机器（都）不具有这些特性”是对此的推广，其真实性从背景句无法完全确定，应为U)。结论句: "确实没有任何机器具有这些特性。" -> U。*

---
### 注意事项：
- **严格依据背景句内容：** 不要依赖你的世界知识或外部信息。判断的核心是背景句是否*明确提供或强烈暗示了*结论句的真实性。
- **关注否定词和转折词：** 特别注意背景句或结论句中的“不”、“没有”、“但是”、“然而”、“却”等词，它们直接影响对事件真实性的判断和比较。
- **“声称”、“认为”类词汇：** 这些词汇表达的是观点、主张或指控。除非背景句有其他信息佐证其内容的真伪，否则其内容的真实性通常是 **U**。
- **“污蔑”、“谎称”类词汇：** 这些词汇强烈暗示其后的内容是虚假的。
- **“以为A，但是/后来发现B”：** 这种结构通常意味着A是说话人曾经的错误认知（应视为假），而B是修正后的正确认知（应视为真）。
- **请一步步推理，最后仅输出 T, F 或 U。**

## 输入：

背景句：{text}
结论句：{hypothesis}
谓词：{predicate}

---
请一步一步进行推理，**不要输出 R 或其他任何文字。** 你的回答 **只能是 T, F, 或 U 中的一个字母**。
##  最终判断 (请仅输出 T, F, 或 U):

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

    prompt = PROMPT_TEMPLATE.format(text=text, hypothesis=hypothesis, predicate=predicate)

    predicted_answer = 'Error' # 默认值以防 API 调用失败
    try:
        response = client.chat(model=OLLAMA_MODEL, messages=[
            {'role': 'user', 'content': prompt}
        ])
        answer_raw = response['message']['content'].strip()

        # 过滤掉 <think>...</think> 之间的内容
        answer_raw = re.sub(r'<think>.*?</think>', '', answer_raw, flags=re.DOTALL).strip()

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