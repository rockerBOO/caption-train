from http.server import HTTPServer, BaseHTTPRequestHandler, SimpleHTTPRequestHandler
import json
from pathlib import Path
import os
from urllib.parse import unquote


class SubHTTPServer(HTTPServer):
    """The main server, you pass in base_path which is the path you want to serve requests from"""

    def __init__(
        self,
        base_local_path,
        base_url_path,
        server_address,
        RequestHandlerClass=BaseHTTPRequestHandler,
    ):
        self.base_local_path = base_local_path
        self.base_url_path = base_url_path
        HTTPServer.__init__(self, server_address, RequestHandlerClass)


def run(localpath, server_class=HTTPServer, handler_class=BaseHTTPRequestHandler):
    server_address = ("", 8000)
    httpd = server_class(
        localpath,
        "/img",
        server_address,
        handler_class,
    )
    print(f"Listen on {server_address[0]}:{server_address[1]}")
    httpd.serve_forever()


class ServerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # if directory is None:
        #     raise ValueError("Directory not passed")

        super().__init__(*args, **kwargs)

    def translate_path(self, path):
        if path.startswith(self.server.base_url_path):
            path = path[len(self.server.base_url_path) :]
            if path == "":
                path = "/"
            print(self.server.base_local_path + unquote(path))
            return self.server.base_local_path + unquote(path)
        else:
            return SimpleHTTPRequestHandler.translate_path(self, path)

    def do_GET(self):
        if self.path == "/list.json":
            self.response_with_json()
        else:
            SimpleHTTPRequestHandler.do_GET(self)

    def response_with_json(self):
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()

        testing = []

        for file in Path(self.server.base_local_path).iterdir():
            if file.suffix not in [".png", ".jpeg", ".jpg", ".webp", ".bmp"]:
                continue

            for suffix in [".txt", ".caption"]:
                caption_file = file.with_name(f"{file.stem}{suffix}")
                if caption_file.exists():
                    with open(caption_file) as f:
                        testing.append(
                            {
                                "image": str(file.name),
                                "caption": " ".join(f.readlines()),
                            }
                        )
                        continue

        # what we write in this function it gets visible on our
        # web-server
        self.wfile.write(json.dumps(testing).encode())
