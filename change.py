import json

def replace_invalid_answers(file1_path, file2_path, output_path):
    # 读取第一个 JSON 文件
    with open(file1_path, 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)
    
    # 读取第二个 JSON 文件
    with open(file2_path, 'r', encoding='utf-8') as file2:
        data2 = json.load(file2)
    
    # 将第二个文件的数据转换为字典，方便查找
    data2_dict = {item['d_id']: item['answer'] for item in data2 if 'd_id' in item and 'answer' in item}
    
    # 替换 answer 为 "invalid" 的字段
    for item in data1:
        if item.get('answer') == "Invalid" and 'd_id' in item:
            item_id = item['d_id']
            if item_id in data2_dict:
                item['answer'] = data2_dict[item_id]
    
    # 将结果写入输出文件
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(data1, output_file, ensure_ascii=False, indent=4)

# 示例调用
file1 = "predictions_NatS.json"  # 第一个 JSON 文件路径
file2 = "predictions_NatS (1)(1).json"  # 第二个 JSON 文件路径
output = "output1.json"  # 输出文件路径

replace_invalid_answers(file1, file2, output)