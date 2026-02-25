import asyncio
import json


async def handle_hello_world(scope, receive, send):
    assert scope["type"] == "http"
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"text/plain"),
            ],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": b"Hello, world!",
        }
    )


async def handle_latency(scope, receive, send):
    assert scope["type"] == "http"
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"text/plain"),
            ],
        }
    )
    await asyncio.sleep(1.0)
    await send(
        {
            "type": "http.response.body",
            "body": b"Hello, world!",
        }
    )


async def handle_post(scope, receive, send):
    assert scope["type"] == "http"
    more_body = True
    while more_body:
        message = await receive()
        # Ignore the body, but consume it to make it a fair benchmark
        more_body = message.get("more_body", False)
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"text/plain"),
            ],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": b"Hello World!",
        }
    )


dummy_json = json.dumps({f"key{i}": ["value"] * 20 for i in range(20)}).encode("utf-8")


async def handle_json_response(scope, receive, send):
    assert scope["type"] == "http"
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/json"),
            ],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": dummy_json,
        }
    )


dummy_chunk = b"a" * 1024


async def handle_chunked(scope, receive, send):
    assert scope["type"] == "http"
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"text/plain"),
            ],
        }
    )
    for _ in range(63):
        await send(
            {
                "type": "http.response.body",
                "body": dummy_chunk,
                "more_body": True,
            }
        )
    await send(
        {
            "type": "http.response.body",
            "body": dummy_chunk,
        }
    )


async def handle_not_found(scope, receive, send):
    assert scope["type"] == "http"
    await send(
        {
            "type": "http.response.start",
            "status": 404,
            "headers": [
                (b"content-type", b"text/plain"),
            ],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": b"Not found",
        }
    )


HANDLERS = {
    "/hello": handle_hello_world,
    "/json": handle_json_response,
    "/chunked": handle_chunked,
    "/latency": handle_latency,
    "/upload": handle_post,
}


async def app(scope, receive, send):
    assert scope["type"] == "http"
    handler = HANDLERS.get(scope["path"], handle_not_found)
    return await handler(scope, receive, send)
