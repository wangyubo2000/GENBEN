import os
import json

# 文件路径
golden_path = './save_data/descriptions_test_golden_top_debug_code.jsonl'
new_golden_path = './save_data/descriptions_test_new_gpt4_golden_top_debug_code.jsonl'
llm_ds_path = './save_data/description_top_debug_code_done_gpt4.jsonl'

# 读取golden_path的内容
with open(golden_path, 'r', encoding='utf-8') as f_golden:
    golden_lines = [json.loads(line) for line in f_golden.readlines()]

# 读取llm_ds_path的内容
with open(llm_ds_path, 'r', encoding='utf-8') as f_llm:
    llm_lines = [json.loads(line) for line in f_llm.readlines()]

# 创建一个以task_id为键的字典，方便查找
llm_dict = {entry[0]: entry[2]['choices'][0]['message']['content'] for entry in llm_lines}

# 遍历golden_lines，查找并替换相应的task_commend
for golden_entry in golden_lines:
    task_id = golden_entry.get('task_id')
    if task_id in llm_dict:
        # 替换task_commend为llm_dict中对应的content
        golden_entry['task_commend'] = llm_dict[task_id]

# 将修改后的内容写入new_golden_path
with open(new_golden_path, 'w', encoding='utf-8') as f_new_golden:
    for entry in golden_lines:
        f_new_golden.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"内容已成功替换并保存到 {new_golden_path}")


