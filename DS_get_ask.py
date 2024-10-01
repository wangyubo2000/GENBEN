################# 利用大模型为问题描述提供扰动 ##########################
import os
import re

import openai

import aiohttp  # for making API calls concurrently
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import time  # for sleeping after rate limit is hit
from dataclasses import dataclass, field  # for storing API inputs, outputs, and metadata

os.environ["OPENAI_API_url"] = "https://a.fe8.cn/v1/chat/completions"
os.environ["OPENAI_API_KEY"] = "sk-jxGxEstKGNXlwD4gaFIIveFQOAWm7hRxUZth8k191o5m8DE4"

#TODO
requests_filepath = './save_data/descriptions_test_golden_top_debug_code.jsonl'


sva_dir = './save_data' # 修改-> 提供生成路径
os.makedirs(sva_dir, exist_ok=True)
done_filepath = os.path.join(sva_dir, 'description_top_debug_code_done_gpt4.jsonl')
undone_filepath = os.path.join(sva_dir, 'description_top_debug_code_undone_gpt4.jsonl')
log_filepath = os.path.join(sva_dir, 'description.log')

run_config = {
    "requests_filepath": requests_filepath,
    "done_filepath": done_filepath,
    "undone_filepath": undone_filepath,
    "log_filepath": log_filepath,
    "request_url": os.environ["OPENAI_API_url"],
    "api_key": os.environ["OPENAI_API_KEY"],
    "max_requests_per_minute": 10,
    "max_attempts": 2,
    "interval_time": 0.2,
    "logging_level": logging.INFO,
    "model": "gpt-4-turbo",
    #"model": "claude-3-opus-20240229",
    "temperature": 0.2
}

#policy = asyncio.WindowsSelectorEventLoopPolicy()
#asyncio.set_event_loop_policy(policy)


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

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


async def main():
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
        # `requests` will provide requests one at a time
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
                                # spec_id = int(task_id.split('_')[0])
                                # 判断语法检查是否正确，不正确时跳过
                                # if spec_id not in passed_ids:
                                #     continue
                                if task_id in done_list:
                                    continue
                                description = next_data.get('task_recommend','')
                                request_json = dict()
                                request_json['temperature'] = run_config['temperature']
                                request_json["model"] = run_config['model']

                                prompt = ( "Please rephrase the following paragraph while retaining the original meaning of the sentence:\n"
                                f'{description.strip()}\n'
                                )

                                request_json['messages'] = [
                                    {"role": "system",
                                     "content": "You are now a designer with extensive knowledge of hardware design."},
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


if __name__ == "__main__":
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
        asyncio.run(main())
    except Exception as e:
        print('main task have error:', e)
