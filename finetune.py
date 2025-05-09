import json

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

def transform_data(input_filepath, output_filepath):
    """
    Transforms the original dataset into a fine-tuning format.

    Args:
        input_filepath (str): Path to the original JSON file.
        output_filepath (str): Path to save the transformed JSON file.
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f_in:
            original_data = json.load(f_in)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_filepath}")
        return
    except json.JSONDecodeError:
        print(f"错误：无法解析输入文件 {input_filepath}")
        return

    transformed_data = []
    # instruction = "仅根据以下文本内容，判断假设内容是否为真。请回答“真”、“假”或“不能确定”。" # 旧的 instruction

    for item in original_data:
        item_text = item.get('text', '')
        item_hypothesis = item.get('hypothesis', '')
        item_predicate = item.get('predicate', '')
        item_type = item.get('type', '').strip()
        # 假设 'answer' 字段在输入 JSON 中已经是 "T", "F", 或 "U"
        output_text = item.get('answer', '')
        
        current_instruction_template = None
        if item_type == "正叙实":
            current_instruction_template = PROMPT_TEMPLATE_FACTUAL
        elif item_type == "反叙实":
            current_instruction_template = PROMPT_TEMPLATE_COUNTER_FACTUAL
        elif item_type == "非叙实":
            current_instruction_template = PROMPT_TEMPLATE_NON_FACTUAL
        
        final_instruction = ""
        if current_instruction_template:
            try:
                final_instruction = current_instruction_template.format(
                    predicate=item_predicate,
                    text=item_text,
                    hypothesis=item_hypothesis
                )
            except KeyError as e:
                print(f"警告：格式化指令时发生错误（缺少键 {e}）。项目 ID: {item.get('d_id', 'N/A')}")
                final_instruction = f"错误：无法为此项生成指令。类型：{item_type}, 谓词：{item_predicate}"
        else:
            print(f"警告：数据项 {item.get('d_id', 'N/A')} 的类型 '{item_type}' 没有对应的指令模板。将使用通用指令。")
            # 可以选择跳过此项，或使用一个非常通用的指令
            final_instruction = (
                f"任务：根据以下信息判断结论句的真假。\n"
                f"类型：{item_type} (无特定模板)\n"
                f"谓词：{item_predicate}\n"
                f"背景句：{item_text}\n"
                f"结论句：{item_hypothesis}\n"
                f"请根据背景句判断结论句，输出 T, F, 或 U."
            )

        transformed_data.append({
            "instruction": final_instruction,
            "input": "",  # 输入现在是指令的一部分
            "output": output_text
        })

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            json.dump(transformed_data, f_out, ensure_ascii=False, indent=4)
        print(f"成功将转换后的数据保存到 {output_filepath}")
    except IOError:
        print(f"错误：无法写入输出文件 {output_filepath}")

# 定义输入和输出文件路径
input_file = r"d:\大二下\代码\知识工程\Sample_Set\previous version\20250325\ArtS_20250325.json"
output_file = r"d:\大二下\代码\知识工程\ArtS_finetune_data_with_detailed_instructions.json" # 修改了输出文件名以作区分

# 执行转换
transform_data(input_file, output_file)