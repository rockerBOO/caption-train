# Caption trainer

Train IQA for captioning using 🤗 compatible models.

<!--toc:start-->

- [Caption Trainer](#caption-trainer)
  - [Support](#support)
  - [Install](#install)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [Setup dataset from text/image pairs](#setup-dataset-from-textimage-pairs)
  - [Development](#development)
    - [Test](#test)

<!--toc:end-->

## Support

- [BLIP][blip]
- [Florence 2][florence]

## Install

Using [uv][uv] for dependencies.

### Dependencies

For PyTorch Cuda:

`uv sync --extra cu124`

For PyTorch CPU:

`uv sync --extra cpu`

# Usage

- [BLIP][blip]
- [Florence 2][florence]

## Development

[uv][uv] for dependencies.
[Ruff][ruff] for linting and formatting.
[Pytest][pytest] for testing.

### Test

```sh
$ uv run pytest
platform linux -- Python 3.11.10, pytest-8.3.4, pluggy-1.5.0 -- /home/rockerboo/code/caption-train/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/rockerboo/code/caption-train
configfile: pyproject.toml
plugins: anyio-4.8.0
collected 27 items
caption_train/tests/test_llm.py::test_remove_think_tags PASSED
...
```

[blip]: docs/blip
[florence]: docs/florence
[uv]: https://docs.astral.sh/uv/
[ruff]: https://docs.astral.sh/ruff/
[pytest]: https://docs.pytest.org/en/stable/
