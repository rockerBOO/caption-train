import ast
import argparse
from typing import Any


class LossRecorder:
    """Records and tracks training loss values with efficient moving average calculation.

    This class maintains a list of loss values and efficiently calculates the moving
    average. It handles both initial recording (epoch 0) and replacement of loss
    values for subsequent epochs at the same step.

    Attributes:
        loss_list: List of loss values, indexed by step
        loss_total: Running sum of all loss values for efficient average calculation
    """

    def __init__(self):
        self.loss_list: list[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        """Add or update a loss value for a specific epoch and step.

        Args:
            epoch: Training epoch number (0-indexed)
            step: Training step within the epoch
            loss: Loss value to record

        Note:
            For epoch 0, new loss values are appended to the list.
            For subsequent epochs, existing loss values at the same step are replaced.
        """
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        """Calculate the moving average of all recorded loss values.

        Returns:
            float: Average loss across all recorded steps
        """
        return self.loss_total / len(self.loss_list)


def parse_dict(input_str: str) -> Any:
    """Convert string input into a dictionary."""
    try:
        # Use ast.literal_eval to safely evaluate the string as a Python literal (dict)
        return ast.literal_eval(input_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {input_str}")


def get_group_args(args: argparse.Namespace, group: argparse._ArgumentGroup):
    return {action.dest: getattr(args, action.dest) for action in group._group_actions}
