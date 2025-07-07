import pytest
import argparse
from caption_train.util import LossRecorder, parse_dict, get_group_args


def test_loss_recorder_initial_add():
    recorder = LossRecorder()
    recorder.add(epoch=0, step=0, loss=1.0)
    assert recorder.loss_list == [1.0]
    assert recorder.loss_total == 1.0
    assert recorder.moving_average == 1.0


def test_loss_recorder_multi_add():
    recorder = LossRecorder()
    recorder.add(epoch=0, step=0, loss=1.0)
    recorder.add(epoch=0, step=1, loss=2.0)

    assert recorder.loss_list == [1.0, 2.0]
    assert recorder.loss_total == 3.0
    assert recorder.moving_average == 1.5


def test_loss_recorder_replace():
    recorder = LossRecorder()
    recorder.add(epoch=0, step=0, loss=1.0)
    recorder.add(epoch=1, step=0, loss=2.0)

    assert recorder.loss_list == [2.0]
    assert recorder.loss_total == 2.0
    assert recorder.moving_average == 2.0


def test_parse_dict_valid():
    assert parse_dict("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}
    assert parse_dict("{}") == {}


def test_parse_dict_invalid():
    with pytest.raises(Exception):  # This catches broader exceptions including SyntaxError
        parse_dict("invalid dict")


def test_get_group_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("test_group")
    group.add_argument("--foo", default=1)
    group.add_argument("--bar", default="test")

    args = parser.parse_args(["--foo", "42", "--bar", "hello"])
    group_args = get_group_args(args, group)

    assert group_args == {"foo": "42", "bar": "hello"}  # Arguments are parsed as strings
