import ast
import argparse
from typing import List


class LossRecorder:
    def __init__(self):
        self.loss_list: List[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        return self.loss_total / len(self.loss_list)


def print_gpu_utilization():
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlInit,
    )

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024**2} MB.")


def parse_dict(input_str):
    """Convert string input into a dictionary."""
    try:
        # Use ast.literal_eval to safely evaluate the string as a Python literal (dict)
        return ast.literal_eval(input_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {input_str}")
