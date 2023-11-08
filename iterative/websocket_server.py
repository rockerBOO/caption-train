#!/usr/bin/env python

import asyncio
from websockets.server import serve


# Server interconnects the client and the web application server

# Client (views/updates the caption to the image)
# Web socket server handles communications between the web application server and the client
# Server handles serving images, json about the application
# Traiing s

# RequestFrom = Enum("from", ["client", "training"])
# RequestType = Enum("type", ["caption.quality"])
#
# class NewCaption(Schema):
#     new_caption = fields.Str()
#
# class RequestSchema(Schema):
#     req_from = fields.Enum(RequestFrom)
#     type = fields.Enum(RequestType)
#     payload = fields.Dict(validate=validate.OneOf([NewCaption]))


request = {
    "from": "client",
    "type": "caption.quality",
    "payload": {"new_caption": "a man sitting in a chair"},
}


CLIENTS = set()


async def relay(queue, websocket):
    while True:
        # Implement custom logic based on queue.qsize() and
        # websocket.transport.get_write_buffer_size() here.
        message = await queue.get()
        await websocket.send(message)


async def process(message, websocket):
    print(message)
    await websocket.send("hi")


async def handler(websocket):
    queue = asyncio.Queue()
    relay_task = asyncio.create_task(relay(queue, websocket))
    CLIENTS.add(queue)
    try:
        async for message in websocket:
            await process(message, websocket)
        await websocket.wait_closed()
    finally:
        CLIENTS.remove(queue)
        relay_task.cancel()


# def broadcast(message):
#     for queue in CLIENTS:
#         queue.put_nowait(message)


async def echo(websocket):
    async for message in websocket:
        await websocket.send(message)


async def main():
    async with serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever


asyncio.run(main())
