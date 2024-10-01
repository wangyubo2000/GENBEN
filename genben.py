import os
import re
from xml.dom.minidom import parse
import base64
import openai
import argparse
import aiohttp  # for making API calls concurrently
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import time  # for sleeping after rate limit is hit
import subprocess
from dataclasses import dataclass, field  # for storing API inputs, outputs, and metadata
from utils.openai_client import model
from fuzzywuzzy import fuzz
os.environ["OPENAI_API_url"] = "https://a.fe8.cn/v1/chat/completions"
os.environ["OPENAI_API_KEY"] = "sk-jxGxEstKGNXlwD4gaFIIveFQOAWm7hRxUZth8k191o5m8DE4"


# TODO


# policy = asyncio.WindowsSelectorEventLoopPolicy()
# asyncio.set_event_loop_policy(policy)

@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""
    completed_tasks = 1  # 类变量，用于跟踪已完成任务的数量
    task_id: str
    request_json: dict
    attempts_left: int
    result: list = field(default_factory=list)

    async def call_api(
            self,
            session: aiohttp.ClientSession,
            retry_queue: asyncio.Queue,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        timeout = aiohttp.ClientTimeout(total=600)  # 每个请求的超时为200s
        error = None
        try:
            async with session.post(url=run_config["request_url"], headers=request_header,
                                    data=json.dumps(self.request_json),
                                    timeout=timeout) as response:
                try:
                    response = await response.json(encoding='utf-8')
                except:
                    logging.warning(f'response is not a json: {response}')
                if 'error' in response:
                    error = response['error']
                    logging.warning(
                        f'Request {self.task_id} failed, The response have an error: {error}')
        except openai.APIConnectionError as e:
            logging.warning(
                f'Request {self.task_id} failed, The server could not be reached')
            error = e
        except openai.RateLimitError as e:
            logging.warning(
                f'Request {self.task_id} failed, A 429 status code was received;')
            await asyncio.sleep(60)
            error = e
        except openai.APIStatusError as e:
            logging.warning(
                f'Request {self.task_id} failed, Another non-200-range status code was received, status_code: '
                f'{e.status_code}, error response: {e.response}')
            error = e
        except asyncio.TimeoutError as e:
            logging.warning(f"Request {self.task_id} failed, with Exception TimeoutError")
            error = e
        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            if e:
                logging.warning(f"Request {self.task_id} failed, with Exception {e}")
            else:
                logging.warning(f"Request {self.task_id} failed, with unknown error！")
                e = "unknown error"
            error = e
        # 如果发生错误均进行重试， 如果没有错误，将结果写入done文件
        if error:
            if self.attempts_left > 0:
                retry_queue.put_nowait(self)
                logging.warning(f"Request {self.task_id} input to retry queue!")
                self.attempts_left -= 1
            else:
                try:
                    self.messages = self.request_json['messages']
                    self.model = self.request_json['model']
                    logging.error(f"Request {self.task_id} failed after all attempts. {error}")
                    append_to_jsonl([self.task_id, self.messages, "", self.model],
                                    run_config["undone_filepath"])
                    del tasks[self.task_id]
                except Exception as e:
                    logging.error(f"Request {self.task_id} write to undone file have an error:{e}")

        else:
            try:
                self.messages = self.request_json['messages']
                self.model = self.request_json['model']
                data = [self.task_id, self.messages, response, self.model]
                append_to_jsonl(data, run_config["done_filepath"])
                logging.info(f"Request {self.task_id} saved to done file!")
                del tasks[self.task_id]
                APIRequest.completed_tasks += 1  # 任务完成后累加计数
            except Exception as e:
                logging.error(f"Request {self.task_id} write to done file have an error:{e}")


# functions
def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a", encoding='utf-8') as f:
        f.write(json_string + "\n")


def del_ann(code):
    code = code.strip()
    # 将所有的多行注释去掉
    code_ = re.sub('/\*+[\s\S]*\*/', '', code)
    lines = code_.split('\n')
    new_lines = []
    for line in lines:
        new_line = line.lstrip()
        if new_line.startswith('//'):  # 将注释行去掉
            continue
        new_line = re.sub('//[\s\S]*', '', line.rstrip())  # 将代码后面的注释去掉
        new_lines.append(new_line)
    new_code = '\n'.join(new_lines)
    new_code = new_code.strip()
    return new_code


def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


async def main(mode):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    requests_num = 0
    rpm = run_config["max_requests_per_minute"]
    next_request = None  # variable to hold the next request to call
    file_not_finished = True  # after file is empty, we'll skip reading it
    # initialize trackers
    retry_queue = asyncio.Queue()
    conn = aiohttp.TCPConnector(ssl=False, limit=0)  # 防止ssl报错
    # 停止程序标记
    end_num = 0
    total_tasks = 1
    # initialize file reading
    with open(run_config["requests_filepath"], encoding='utf-8') as file:
        # requests will provide requests one at a time
        requests = file.__iter__()
        logging.info(f"File opened. Entering main loop")
        start = time.time()
        task_start = False
        task = None
        task_id = 's'
        async with aiohttp.ClientSession(
                connector=conn) as session:  # Initialize ClientSession here
            try:
                while True:
                    task_start = False
                    # get next request (if one is not already waiting for capacity)
                    if next_request is None:
                        if not retry_queue.empty():  # 如果重试队列不为空时，优先从重试队列中获取请求
                            next_request = retry_queue.get_nowait()
                            logging.info(
                                f"Retrying request {next_request.task_id}-----------------------"
                            )
                        elif file_not_finished:
                            try:
                                line = next(requests)
                                # TODO 在这里根据不同的数据格式要进行修改
                                next_data = json.loads(line.strip())
                                task_id = next_data['task_id']
                                if mode == 'text' and 'M' in task_id:  # 如果mode的模式为text,遇到M则跳过
                                    continue
                                elif mode == 'mm' and 'M' not in task_id:  # 如果mode的模式为mm,遇到没有M则跳过
                                    continue
                                if "images" in next_data.keys():
                                    image_paths = next_data['images']
                                    # encoded_images = []
                                    for image_path in image_paths:
                                        img_url = os.path.join('./data', image_path)
                                        encoded_images = []
                                        encoded_image = encode_image_to_base64(os.path.join('./data', image_path))
                                        encoded_images.append(encoded_image)
                                else:
                                    encoded_image = None
                                    img_url = None
                                if task_id in done_list:
                                    continue
                                description = next_data.get('task_recommend', '')
                                request_json = dict()
                                request_json['temperature'] = run_config['temperature']
                                request_json["model"] = run_config['model']
                                prompt = (
                                    "Choose the correct answer to the following questions based on the information provided. Please output the final answer only, without providing any explanations."
                                    f'{description.strip()}\n'
                                    )
                                request_json['messages'] = [
                                    {"role": "system",
                                     "content": "You are now a designer with extensive knowledge of hardware design."},
                                    {
                                        "type": "image_url",
                                        "image_url": img_url,
                                    },
                                    {"role": "user", "content": prompt.strip()}]
                                # 创建api请求对象
                                next_request = APIRequest(
                                    task_id=task_id,
                                    request_json=request_json,
                                    attempts_left=run_config["max_attempts"],
                                )
                                # logging.info(f"Reading request {next_request.task_id}: {next_request}")
                            except StopIteration:
                                # if file runs out, set flag to stop reading it
                                logging.info("Read file exhausted")
                                file_not_finished = False
                        else:
                            await asyncio.sleep(10)
                            logging.info(
                                "run while true with retry queue is empty and file finished! this need sleep 10s")
                            end_num += 1
                            if APIRequest.completed_tasks == total_tasks:
                                break
                            if end_num > 180:
                                logging.info("wait time 30 mins, end!")
                                break
                    # update available capacity
                    if next_request:
                        current = time.time()
                        if current - start > 60:
                            requests_num = 0
                            start = current
                        if requests_num < rpm:
                            await asyncio.sleep(interval_time)  # 确保一分钟不超限流rpm
                            # await asyncio.sleep(0.3) #确保一分钟不超限流rpm
                            # call API
                            task = asyncio.create_task(
                                next_request.call_api(
                                    session=session,
                                    retry_queue=retry_queue
                                )
                            )
                            task_start = True
                            requests_num += 1
                            tasks[task_id] = task
                            next_request = None  # reset next_request to empty
                            total_tasks += 1
                        else:
                            # todo 默认的设置
                            # todo 由于请求序列比较长，响应速度比较慢，所以之前的等待时间不再适应，需要自定义等待时间，如下所示，每个批次的请求发送出去之后，等待2分钟
                            logging.warning(
                                '{}s内已请求{}, 需等待:{}s'.format(round((current - start), 2),
                                                                   len(tasks),
                                                                   60))
                            await asyncio.sleep(120)

            except asyncio.CancelledError:
                logging.warning("Waiting for ongoing tasks to complete...")
                if task_start:
                    if task_id not in tasks:
                        tasks[task_id] = task
                running_tasks = list(tasks.values())
                # 还有在运行的任务时，需要等任务全部结束
                if running_tasks:
                    await asyncio.gather(*running_tasks, return_exceptions=True)
                    logging.info("all task complete, and then waite 3s!")
                    await asyncio.sleep(3)
                else:
                    logging.info("At this point, the task list is empty")
                    await asyncio.sleep(3)
            except Exception as e:
                logging.error(f'main loop have an error:{e}')
        # after finishing, log final status
        logging.info("Parallel processing complete.-----------------------")

############  拆分结果的函数  ####################
def get_verilog(data_path, file_path_json,file_path_debug_json):
    with open(data_path, 'r') as f:
        for line in f:
            code = {}
            lines = json.loads(line)
            task_id = lines[0]
            code['task_id'] = task_id
            verilog_line = lines[2]['choices'][0]['message']['content']
            if 'D' not in code['task_id']:
                pattern = r"```verilog(.*?)```"
                matches = re.findall(pattern, verilog_line, re.DOTALL)
                if matches:
                    verilog_code = matches[0]
                    code['verilog_code'] = verilog_code
                    # print(verilog_code)
                else:
                    code['verilog_code'] = """
                                        module top (out, in, clk); 
                                            output out;
                                            input in, clk;
                                            reg s1, s2, s3;

                                            assign out = s1==0 & s2==0 & s3==0;

                                            initial begin
                                                s1 = 0;
                                                s2 = 0;
                                                s3 = 0;
                                            end

                                            always @(posedge clk) begin
                                                s1 <= in;
                                                s2 <= s1;
                                                s3 <= s2;
                                            end
                                        endmodule
                                        """
                    print("No match found.")
                json_string = json.dumps(code, ensure_ascii=False)
                with open(file_path_json, "a+", encoding='utf-8') as f:
                    f.write(json_string + "\n")
            else:
                code['verilog_code'] = verilog_line
                json_string = json.dumps(code, ensure_ascii=False)
                with open(file_path_debug_json, "a+", encoding='utf-8') as f:
                    f.write(json_string + "\n")
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
def merge_done_files(final_done_path, parts_paths):
    # 打开目标文件（合并后的文件）
    with open(final_done_path, 'a', encoding='utf-8') as final_done_file:
        for part_path in parts_paths:
            # 检查文件是否存在
            if os.path.exists(part_path):
                # 读取每个 parts_path 文件并写入最终的 done 文件
                with open(part_path, 'r', encoding='utf-8') as part_file:
                    final_done_file.write(part_file.read())
                # 合并后删除中间文件
                os.remove(part_path)
                print(f"Deleted file: {part_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the corresponding code according to the large model.")
    parser.add_argument("--mode", type=str, default='all',
                        help="control mode")  #### 设计模态
    parser.add_argument("--model", type=str, default='gpt4o', help="model descrption")
    args = parser.parse_args()

    requests_filepath = './save_data/descriptions_test_golden_choice.jsonl'  #### 描述

    sva_dir = './save_data'  # 修改-> 提供生成路径
    os.makedirs(sva_dir, exist_ok=True)
    for i in range(5): ###### llm 发出几次指令
        done_path = os.path.join(sva_dir, f'answer_done_choice_{args.model}_{args.mode}_parts_{i}.jsonl')  #### 保存成功结果的文件
        undone_path = os.path.join(sva_dir, f'answer_undone_choice_{args.model}_{args.mode}_parts_{i}.jsonl')  #### 保存结果不成功的文件
        log_filepath = os.path.join(sva_dir, 'description.log')

        run_config = {
            "requests_filepath": requests_filepath,
            "done_filepath": done_path,
            "undone_filepath": undone_path,
            "log_filepath": log_filepath,
            "request_url": os.environ["OPENAI_API_url"],
            "api_key": os.environ["OPENAI_API_KEY"],
            "max_requests_per_minute": 10,
            "max_attempts": 2,
            "interval_time": 0.2,
            "logging_level": logging.INFO,
            "model": "gpt-4o",  #### 切换模型
            #"model": "claude-3-opus-20240229",
            "temperature": 0.9  ### 0.2-0.5需要准确性 0.7-1.0 创造性
        }

        # 已经完成的任务
        done_list = []
        if os.path.exists(run_config['done_filepath']):
            with open(run_config['done_filepath'], 'r', encoding='utf-8') as done_f:
                for line in done_f.readlines():
                    line = json.loads(line.strip())
                    # tid = int(line[0])
                    tid = line[0]
                    if tid not in done_list:
                        done_list.append(tid)

        if os.path.exists(run_config['undone_filepath']):
            with open(run_config['undone_filepath'], 'r', encoding='utf-8') as undone_f:
                for line in undone_f.readlines():
                    line = json.loads(line.strip())
                    # tid = int(line[0])
                    tid = line[0]
                    if tid not in done_list:
                        done_list.append(tid)
        tasks = dict()
        api_key = run_config["api_key"]
        request_header = {"Content-Type": "application/json;charset=UTF-8",
                          "Authorization": f"Bearer {api_key}"}
        # constants
        interval_time = run_config['interval_time']  # limits max throughput to 150 requests per second
        # 存储错误信息的日志文件
        # logger = get_logger(run_config["log_filepath"])
        # 在控制台显示信息
        log_level = run_config["logging_level"]
        # logging.basicConfig(level=log_level, filename=run_config['log_filepath'])
        logging.basicConfig(level=log_level)
        logging.info(f"Initialization complete.")

        # run script
        try:
            asyncio.run(main(args.mode))
            APIRequest.completed_tasks = 1        #### 重置回答数 ########
        except Exception as e:
            print('main task have error:', e)
    parts_paths = []
    for i in range(5):  # 有 5 个 parts 文件
        parts_path = os.path.join(sva_dir, f'answer_done_choice_{args.model}_{args.mode}_parts_{i}.jsonl')
        parts_paths.append(parts_path)

    # 最终的 done 文件路径
    done_filepath = os.path.join(sva_dir, f'answer_done_choice_{args.model}_{args.mode}.jsonl')  ###### the answer of LLM
    merge_done_files(done_filepath, parts_paths)
    save_file = os.path.join('./save_data', f'eval_{args.mode}_{args.model}_code.jsonl')  ##### the answer of code generate
    save_file_debug = os.path.join('./save_data',f'eval_{args.mode}_{args.model}_debug.jsonl')  ####### the answer of debug
    ######## 直接拆分结果 ####################################
    # get_verilog(done_filepath, save_file,save_file_debug)
    # print('Debugging and code generation tasks are complete.')
    # ########## 验证debug的结果 pass@5 #########################
    # llm_result_path = save_file_debug
    # golden_result_path = requests_filepath
    # acc = debug_pass(llm_result_path, golden_result_path)
    # print(f'code debug pass@5 {acc}')
    # ######## 测试功能结果  ######################
    # def run_command(save_file):
    #     problem_file_path = f'save_data/descriptions_test_golden_top_code_{args.mode}.jsonl'
    #     # Command setup
    #     command = [
    #         "evaluate_functional_correctness",
    #         save_file,
    #         f"--problem_file={problem_file_path}"
    #     ]
    #
    #     try:
    #         # Execute the command
    #         result = subprocess.run(command, capture_output=True, text=True)
    #
    #         # Print the output from stdout
    #         print("Output:\n", result.stdout)
    #
    #         # Check for errors and print them from stderr
    #         if result.stderr:
    #             print("Errors:\n", result.stderr)
    #
    #         # Check the exit status
    #         if result.returncode != 0:
    #             print("Command failed with return code:", result.returncode)
    #         else:
    #             print("Command executed successfully!")
    #
    #     except Exception as e:
    #         print("An error occurred while running the command:", e)
    #
    # run_command(save_file)
    #
