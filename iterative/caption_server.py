from server import run, SubHTTPServer, ServerHandler

dir = "~/art/images"
run(dir, SubHTTPServer, ServerHandler)
