import asyncio
from websockets import serve


def send_and_wait_for_response(websocket, message):
    req = dict(
        req_from="trainer",
        type="caption_new",
        payload=dict(new_caption="hello new caption"),
    )
    websocket.send(req)
    recv_message = websocket.recv()
    print(f"Received: {recv_message}")
    return recv_message


async def websocket_process(message, websocket):
    print(message)
    await websocket.send("hi")


# Websocket clients connected
CLIENTS = set()


def broadcast(message):
    for queue in CLIENTS:
        queue.put_nowait(message)


async def broadcast_and_wait(message) -> list[str]:
    for queue in CLIENTS:
        queue.put_nowait(message)

    responses = []
    for queue in CLIENTS:
        responses.append(await queue.get())

    return responses


async def relay(queue, websocket):
    while True:
        # Implement custom logic based on queue.qsize() and
        # websocket.transport.get_write_buffer_size() here.
        message = await queue.get()
        await websocket.send(message)


async def websocket_handler(websocket):
    queue = asyncio.Queue()
    # relay_task = asyncio.create_task(relay(queue, websocket))
    CLIENTS.add(queue)
    try:
        await websocket.wait_closed()
    finally:
        CLIENTS.remove(queue)
        # relay_task.cancel()


async def start_websocket_server():
    async with serve(websocket_handler, "localhost", 8765):
        await asyncio.Future()  # run forever
