import json

# 加载 predictions.json 和 NatS_20250407.json
with open("e:\\Study\\大二下\\知识工程\\ccl25_fie\\models\\predictions.json", "r", encoding="utf-8") as pred_file:
    predictions = json.load(pred_file)

with open("e:\\Study\\大二下\\知识工程\\ccl25_fie\\Sample_Set\\NatS_20250407.json", "r", encoding="utf-8") as nats_file:
    nats = json.load(nats_file)

# 初始化计数器
total_samples = len(nats)
correct_count = 0

# 遍历 predictions 和 nats，进行比对
for pred, nat in zip(predictions, nats):
    if pred["answer"] == nat["answer"]:
        correct_count += 1

# 计算准确率
accuracy = (correct_count / total_samples) * 100

# 输出结果
print(f"总样本数: {total_samples}")
print(f"正确预测数: {correct_count}")
print(f"准确率: {accuracy:.2f}%")