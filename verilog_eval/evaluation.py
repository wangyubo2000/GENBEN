from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict, Tuple, Optional
import itertools

import numpy as np
import tqdm

from verilog_eval.data import read_problems, stream_jsonl, write_jsonl
from verilog_eval.execution import check_correctness, clean_up_simulation,yosys_check_correctness


def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],  # 每个问题的样本数量
        num_correct: Union[List[int], np.ndarray],  # 每个问题中正确的样本数量
        k: int  # 正确的样本数量
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))  # 计算公式

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])  # 返回结果


def contain_passing_completion(
        problem: Dict,
        completions: List[str],  # 待解决的方案
        n_workers: int = 4,  # 一次至执行几个
        timeout: float = 30.0,  # 执行时长
        unit_test_length: Optional[int] = None,
        clean_up: bool = True,
) -> Tuple[bool, str]:
    with ProcessPoolExecutor(max_workers=n_workers) as executor:  # 多进程

        futures = []

        for idx, completion in enumerate(completions):
            args = (problem, completion, timeout, idx, unit_test_length)  #
            future = executor.submit(check_correctness, *args)
            futures.append(future)

        for future in as_completed(futures):
            result = future.result()
            if result["passed"]:
                return True, completions[result["completion_id"]]  # 保存结果

    if clean_up:  # 清理资源
        clean_up_simulation()

    return False, ""


def evaluate_functional_correctness(
        sample_file: str,
        problem_file: str,
        k: List[int] = [1, 10, 100],
        n_workers: int = 4,
        timeout: float = 30.0,
        unit_test: bool = False,
        clean_up: bool = True,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    problems = read_problems(problem_file)  #
    ########################## iverilog #############################
    # Check the generated samples against test suites.
    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["verilog_code"]
            #print(completion)
            if unit_test:
                args = (problems[task_id], completion, timeout, completion_id[task_id], 100)
            else:
                args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    if clean_up:
        clean_up_simulation()

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]  # false 和 correct
        total.append(len(passed))
        correct.append(sum(passed))
    sys_total, sys_correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["result"] for r in result]
        # print(passed)
        if any("failed: syntax error." in message for message in passed):
            passed = [False]
        else:
            passed = [True]
        sys_total.append(len(passed))
        sys_correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)
    sys_total = np.array(sys_total)
    sys_correct = np.array(sys_correct)

    ks = k
    pass_at_k = {f"Functionality_pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}
    pass_sys_k = {f"Syntax_pass@{k}": estimate_pass_at_k(sys_total, sys_correct, k).mean()
                  for k in ks if (total >= k).all()}

    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))
    ##################yosys 仿真结果###############
    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        futures_yosys = []
        completion_id = Counter()
        n_samples = 0
        results_yosys = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["verilog_code"]
            if unit_test:
                args = (problems[task_id], completion, timeout, completion_id[task_id], 100)
            else:
                args = (problems[task_id], completion, timeout, completion_id[task_id])
            future_yosys = executor.submit(yosys_check_correctness, *args)
            futures_yosys.append(future_yosys)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future_yosys in tqdm.tqdm(as_completed(futures_yosys), total=len(futures_yosys)):
            result_yosys = future_yosys.result()
            results_yosys[result_yosys["task_id"]].append((result_yosys["completion_id"], result_yosys))
    #print(results_yosys)

    #print(results_yosys)
    if clean_up:
        clean_up_simulation()

        # Calculate pass@k.
    total_yosys, correct_yosys = [], []
    #print('draw the result of yosys:',results_yosys)
    for result in results_yosys.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]  # false 和 correct
        total_yosys.append(len(passed))
        correct_yosys.append(sum(passed))
    total_yosys = np.array(total_yosys)
    correct_yosys = np.array(correct_yosys)
    ks = k
    pass_yosys_k = {f"Synthesize_pass@{k}": estimate_pass_at_k(total_yosys, correct_yosys, k).mean()
                 for k in ks if (total >= k).all()}
    out_yosys_file = sample_file + "_yosys_results.jsonl"
    print(f"Writing results to {out_yosys_file}...")
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results_yosys[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample
    write_jsonl(out_yosys_file, tqdm.tqdm(combine_results(), total=n_samples))
    #return pass_at_k, pass_sys_k
    return pass_at_k, pass_sys_k,pass_yosys_k
