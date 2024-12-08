import json
from datetime import datetime

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

# 合并JSON数据
def merge_jsonl(data1, data2_dict):
    merged_data_list = []
    for item1 in data1:
        question_id = item1["question_id"]
        item1['answer'] = ", ".join(item1['answer'])
        if str(question_id) in data2_dict:
            # 找到对应的data2数据
            item2 = data2_dict[str(question_id)]
            # 构建问题和答案
            question = item2["question"]
            predict = item1["answer"]  
            answer = item2["answer"]
            merged_data = {
                "video_name": item2["video"],
                "question": question,
                "question_id": question_id,
                "answer": answer,
                "predict": predict,  # 将answer改为predict
                "answer_type": item2["answer_type"],
                "dataset": item2["dataset"],
                "frame_length": item2["frame_length"]
            }
            merged_data_list.append(merged_data)
    
    return merged_data_list

# 将列表中的JSON对象写入JSONL文件
def write_jsonl_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

# 主程序
if __name__ == "__main__":
    # 读取JSON文件
    data1 = read_json_file('/result/20240812_1649/beam5/test_epoch4.json')
    data2 = read_json_file('/msrvtt/annotation/qa_test.json')

    # 将data2转换为字典，以便于根据question_id快速访问
    data2_dict = {str(k): v for k, v in data2.items()}
    result = merge_jsonl(data1, data2_dict)
    timestamp = get_timestamp()
    output_file_path = f'/output_{timestamp}.jsonl'

    # 合并数据
    write_jsonl_file(result, output_file_path)

    print("Merged data has been written to output.jsonl")
