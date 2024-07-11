# A work in progress. Not documented.

A viewer of captions and the images in the browser.

## Usage

Script run.py


```python
from server import run, SubHTTPServer, ServerHandler
run("~/art/images", SubHTTPServer, ServerHandler)
```

Run the server

```bash
$ python run.py
```

Then in your browser

```
http://localhost:8000
```
