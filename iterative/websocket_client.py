#!/usr/bin/env python

from websockets.sync.client import connect
import json


def hello():
    with connect("ws://localhost:8765") as websocket:
        req = dict(
            req_from="client",
            type="caption_new",
            payload=dict(new_caption="hello new caption"),
        )
        websocket.send(json.dumps(req))
        message = websocket.recv()
        print(f"Received: {message}")


hello()
