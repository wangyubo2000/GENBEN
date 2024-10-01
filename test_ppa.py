import sys
import importlib
import os
import re
import json

sys.path.insert(0, '/root/autodl-tmp/GENBEN_test/openlane_ipynb')

# 清除模块缓存
if 'openlane' in sys.modules:
    del sys.modules['openlane']

# 重新导入 openlane 模块
try:
    import openlane

    print("OpenLane 模块成功加载")
    print(openlane.__file__)
except ImportError as e:
    print(f"ImportError: {e}")
from openlane.config import Config
from openlane.steps import Step
from openlane.state import State
##########get power#######################

def extract_total_power(log_file_path):
    try:
        with open(log_file_path, 'r') as log_file:
            lines = log_file.readlines()

        # Variables to hold total power value
        total_power = None

        # Pattern to match the Total row
        total_pattern = re.compile(r"^Total\s+([\d\.\+eE\-]+)\s+([\d\.\+eE\-]+)\s+([\d\.\+eE\-]+)\s+([\d\.\+eE\-]+)")

        # Iterate through the lines to find the Total row
        for line in lines:
            match = total_pattern.match(line)
            if match:
                total_power = match.group(4)  # The 4th group corresponds to the Total Power (Watts)
                break

        if total_power:
            print(f"Total Power (Watts): {total_power}")
        else:
            print("Total power value not found in the log file.")

    except FileNotFoundError:
        print(f"File {log_file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return total_power
################ TNS####################
def extract_tns_values(log_file_path):
    try:
        with open(log_file_path, 'r') as log_file:
            lines = log_file.readlines()

        # 修改后的正则表达式，匹配表格中包含 'nom_tt_025C_1v80' 的行
        # 并提取 Hold TNS 和 Setup TNS 的值
        row_pattern = re.compile((r"nom_tt_025C_1v80\s*\│\s*([-\d\.]+)\s*\│\s*([-\d\.]+)\s*\│\s*([-\d\.]+)\s*\│\s*([-\d\.]+)\s*\│\s*([-\d\.]+)\s*\│\s*([-\d\.]+)\s*\│\s*(N/A|[-\d\.]+)\s*\│\s*(N/A|[-\d\.]+)\s*\│\s*(N/A|[-\d\.]+)\s*\│\s*(N/A|[-\d\.]+)"))

        hold_tns = None
        setup_tns = None
        
        # 查找并提取 'nom_tt_025C_1v80' 对应行的 Hold TNS 和 Setup TNS 值
        for line in lines:
            match = row_pattern.search(line)
            if match:
                hold_tns = match.group(3)  # 第 4 组对应 Hold TNS
                setup_tns = match.group(8)  # 第 9 组对应 Setup TNS
                print(f"匹配行：{line.strip()}")
                print(f"Hold TNS: {hold_tns}, Setup TNS: {setup_tns}")
                break

        if not hold_tns or not setup_tns:
            print("Hold TNS 或 Setup TNS 未找到。")

    except FileNotFoundError:
        print(f"文件 {log_file_path} 未找到")
    except Exception as e:
        print(f"发生错误: {e}")
    return hold_tns,setup_tns
############### Area ############################
def get_area(log_file_path):
    last_line = None

    # 尝试读取文件并保存最后一行
    try:
        with open(log_file_path, 'r') as log_file:
            for line in log_file:
                if "Chip area for module" in line:
                    last_line = line  # 每次找到新的匹配行都覆盖 last_line
        if last_line:
            # 使用正则表达式提取行中的数值部分
            match = re.search(r"Chip area for module.*:\s*([\d\.]+)", last_line)
            if match:
                area_value = match.group(1)
                print(area_value)  # 仅打印数值
            else:
                print("No numeric value found in the matching line.")
        else:
            print("No matching line found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return area_value


def make_run(verilog):
    Config.interactive(
        "top",
        PDK="sky130A",
        CLOCK_PORT="clk",
        CLOCK_NET="clk",
        CLOCK_PERIOD=10
    )
    Synthesis = Step.factory.get("Yosys.Synthesis")
    # Redirect stdout and stderr to /dev/null to suppress terminal output
    with open(os.devnull, 'w') as devnull:
        # Save the original stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            # Redirect stdout and stderr
            sys.stdout = devnull
            sys.stderr = devnull

            # Start the synthesis process
            synthesis = Synthesis(
                VERILOG_FILES=[verilog],
                state_in=State(),
            )
            synthesis.start()

            # Start the STA Pre-PnR process
            STAPrePNR = Step.factory.get("OpenROAD.STAPrePNR")
            sta_pre_pnr = STAPrePNR(state_in=synthesis.state_out)
            sta_pre_pnr.start()

        finally:
            # Restore the original stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
