# import json
#
# # 打开并读取文件内容
# with open('./save_data/descriptions.jsonl', 'r') as file:
#     lines = file.readlines()
#
# # 检查每一行是否为有效JSON
# for i, line in enumerate(lines):
#     line = line.strip()
#     print(line)
#     if not line:
#         print(f"第 {i + 1} 行是空的")
#         continue
#     try:
#         data = json.loads(line)
#         print(f"第 {i + 1} 行内容: {data}")
#     except json.JSONDecodeError as e:
#         print(f"第 {i + 1} 行有问题: {e}")
import os
import json
import sys
import base64
import re
from fuzzywuzzy import fuzz
def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def write_new_desc(desc_path,root_dir): #书写新扰动问题描述
    with open(desc_path,'r') as f:
        for line in f:
            context = json.loads(line)
            folder_name = context[0]
            folder_path = os.path.join(root_dir, folder_name)

            if not os.path.exists(folder_path):
                print(f"文件夹 {folder_name} 不存在，跳过。")
                continue

            file_path = os.path.join(folder_path, 'description_gpt4.txt')
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(context[2]['choices'][0]['message']['content'])
            print(f"内容已写入 {file_path}")
def get_verilog(data_path,file_path_json):
    with open(data_path, 'r') as f:
        for line in f:
            code = {}
            lines = json.loads(line)
            task_id = lines[0]
            code['task_id'] = task_id
            verilog_line = lines[2]['choices'][0]['message']['content']
            pattern = r"```verilog(.*?)```"
            matches = re.findall(pattern, verilog_line, re.DOTALL)
            if matches:
                verilog_code = matches[0]
                code['verilog_code'] = verilog_code
                #print(verilog_code)
            else:
                print("No match found.")
            json_string = json.dumps(code, ensure_ascii=False)
            with open(file_path_json, "a+", encoding='utf-8') as f:
                f.write(json_string + "\n")


import json


import json

def get_verilog_golden(data_path, file_path_json):
    # Open the target file once for appending text.
    with open(file_path_json, "a+", encoding='utf-8') as file_json:
        # Open the source JSON file and read contents line by line.
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Parse each line as a JSON object.
                verilog_entry = json.loads(line)  # Assuming the file contains JSON Lines.

                # Prepare the dictionary to be written as JSON.
                code = {
                    'task_id': verilog_entry['task_id'],
                    'verilog_code': verilog_entry['verilog']
                }

                # Convert dictionary to JSON string.
                json_string = json.dumps(code, ensure_ascii=False)

                # Write the JSON string to the file.
                file_json.write(json_string + "\n")


def debug_pass(llm_result_path, golden_result_path):
    # 逐行读取 JSONL 文件，并将每行的 JSON 对象加载到字典中
    with open(llm_result_path, encoding='utf-8') as f1, open(golden_result_path, "r", encoding='utf-8') as f2:
        llm_result = {json.loads(line)['task_id']: json.loads(line) for line in f1}
        golden_result = {json.loads(line)['task_id']: json.loads(line) for line in f2}

    total = 0
    correct = 0
    threshold = 50

    for task_id, task in llm_result.items():
        if task_id in golden_result:
            content = golden_result[task_id].get('verilog', '')
            get_llm_date = task.get('verilog_code', '')
            # Apply fuzzy matching to check if content is similar to any substring in verilog_code
            similarity = fuzz.partial_ratio(content, get_llm_date)
            if similarity >= threshold:
                correct += 1
        total += 1

    acc = correct / total
    return acc
if __name__ == '__main__':
    desc_path = 'save_data/descriptions_gpt4.jsonl'
    root_dir = './data'
    file_path_json = './save_data/descriptions_golden.jsonl'
    mode = 'text'
    count = 0
    #write_new_desc(desc_path,root_dir)
    #get_verilog(desc_path,file_path_json)
    # get_verilog_golden(desc_path,file_path_json)
    # with open(desc_path,'r') as f:
    #     for line in f:
    #         context = json.loads(line)
    #         folder_name = context['task_id']
    #         if mode == 'text' and 'M' in folder_name:
    #             continue
    #         if mode == 'mm' and 'M' not in folder_name:
    #             continue
    #         print(folder_name)
    #         count +=1
    #     print(count)
    llm_result_path = 'copy/eval_text_gpt4_debug.jsonl'
    golden_result_path = 'copy/descriptions_test_golden_top_debug_code.jsonl'

    acc = debug_pass(llm_result_path, golden_result_path)
    print(acc)
