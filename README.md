# GenBen: A Benchmark for LLM-Aided Design
 GenBen (Generative Hardware Benchmark)is a new benchmark for evaluating large language models (LLMs) in hardware design.We have provided some of the problems in data.zip. We have provided some problems in the data.zip file, along with the corresponding executable code.
## Install
We closely follow guidance from VerilogEval(https://github.com/joshual-xlnx/verilog-eval) and notebook.ipynb 
## Usage  
### In the first step, you can use DS_get_ask.py to obtain the problem description after adding disturbances; and use get_new_ds.py to obtain the JSONL file of the new disturbed problem description.
```
python DS_get_ask.py
python get_new_ds.py
```
### In the second step, please be sure to modify the requests_filepath in genben.py to the newly generated JSON file from the previous step. After completing this, run the following code:
```python genben.py --mode all --model gpt4```
