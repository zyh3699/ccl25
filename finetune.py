import json

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
    instruction = "仅根据以下文本内容，判断假设内容是否为真。请回答“真”、“假”或“不能确定”。"

    for item in original_data:
        input_text = f"文本：{item.get('text', '')}\n假设：{item.get('hypothesis', '')}"
        output_text = item.get('answer', '')

        # Map original answers to required format if necessary (already T/F/U)
        # output_map = {"T": "真", "F": "假", "U": "不能确定"}
        # final_output = output_map.get(output_text, "不能确定") # Keep original T/F/U as requested

        transformed_data.append({
            "instruction": instruction,
            "input": input_text,
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
output_file = r"d:\大二下\代码\知识工程\ArtS_finetune_data.json"

# 执行转换
transform_data(input_file, output_file)
