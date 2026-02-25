import asyncio
import datetime
import resource
import ssl
import subprocess
import sys
import time
import traceback
from contextlib import AsyncExitStack, contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from math import sqrt
from random import expovariate
from time import monotonic, perf_counter
from typing import Iterator
from urllib.request import urlopen

import aiohttp
import httpx
import httpx_aiohttp
import niquests
import orjson
import pycurl
import pyreqwest.client
import trio
import uvloop
from pyreqwest.compatibility.httpx import HttpxTransport as HttpxPyreqwestTransport
from pyreqwest.exceptions import StatusError


def poisson_process(duration: float, initial_rate: float, final_rate: float):
    # We simulate a non-homogeneous process by simulating a a homegeneous
    # process on the unit interval, with the right number of expected arrivals,
    # and then transforming the resulting arrival times
    expected_arrivals = (initial_rate + final_rate) / 2 * duration
    a = initial_rate / expected_arrivals
    b = final_rate / expected_arrivals

    m = (b - a) / duration  # transform to a slope, to keep the math simpler
    c = a

    if m == 0:

        def transform(u):
            return u * duration
    else:

        def transform(u):
            return (sqrt(c**2 + 2 * m * u) - c) / m

    for u in uniform_poisson_process(expected_arrivals):
        yield transform(u)


def uniform_poisson_process(lambd):
    t = 0
    while True:
        t += expovariate(lambd)
        if t >= 1:
            break
        yield t


server_ca_cert_location = "ca.crt"
with open(server_ca_cert_location, "rb") as f:
    server_ca_cert_pem = f.read()
ssl_context = ssl.create_default_context(
    ssl.Purpose.SERVER_AUTH, cafile=server_ca_cert_location
)


class Stats:
    def __init__(self):
        self.start_time = monotonic()
        self.successes = 0
        self.failures = 0
        self.total_latency = 0.0
        self.sum_of_squares = 0.0
        self.max_latency = 0.0
        self.delays = 0

    def record_result(self, latency: float):
        self.successes += 1
        self._add_latency(latency)

    def record_failure(self, latency: float):
        self.failures += 1
        self._add_latency(latency)

    def _add_latency(self, latency: float):
        self.total_latency += latency
        self.sum_of_squares += latency * latency
        if latency > self.max_latency:
            self.max_latency = latency

    def record_delay(self):
        self.delays += 1

    def mean_latency(self):
        if self.successes == 0:
            return 0.0
        return self.total_latency / self.successes

    def latency_stddev(self):
        if self.successes == 0:
            return 0.0
        mean = self.mean_latency()
        try:
            return sqrt(self.sum_of_squares / self.successes - mean * mean)
        except ValueError:
            return 0.0

    def successes_per_second(self):
        elapsed = monotonic() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.successes / elapsed

    def __iadd__(self, other: "Stats"):
        self.successes += other.successes
        self.failures += other.failures
        self.total_latency += other.total_latency
        self.sum_of_squares += other.sum_of_squares
        self.max_latency = max(self.max_latency, other.max_latency)
        self.delays += other.delays
        return self

    def print_stats(self):
        print(
            f"Successes: {self.successes}, Failures: {self.failures}, "
            f"Mean latency: {self.mean_latency():.3f}s, "
            f"Latency stddev: {self.latency_stddev():.3f}s, "
            f"Max latency: {self.max_latency:.3f}s, "
            f"Successes/s: {self.successes_per_second():.2f}, "
            f"Delays: {self.delays}",
            file=sys.stderr,
        )


MAX_CONNECTION_POOL_SIZE = 2000


class BaseBenchmark:
    def __init__(
        self,
        start_time_generator: Iterator[float],
        output_file: str,
        url: str,
        expected_duration: float,
        body: dict | None = None,
        timeout: float = 10.0,
        debug: bool = False,
    ):
        self._start_time_generator = start_time_generator
        self._output_file = open(output_file, "w")
        self._url = url
        self._expected_duration = expected_duration
        self._body = body
        self._timeout = timeout
        self._overall_stats = Stats()
        self._current_stats = Stats()
        self._debug = debug

    async def make_request(self):
        raise NotImplementedError("Subclasses must implement make_request")

    def run_test(self):
        raise NotImplementedError("Subclasses must implement run_test")

    async def run_test_in_loop(self):
        async with self:
            test_start_time = monotonic()
            for start_time in self._start_time_generator:
                delay = test_start_time + start_time - monotonic()
                if delay > 0:
                    await self.sleep(delay)
                elif delay < -self._timeout:
                    print(
                        f"Falling behind schedule by {-delay:.2f}s, ending test",
                        file=sys.stderr,
                    )
                    break
                else:
                    self._record_delay(start_time, -delay)
                    await self.sleep(0)
                self.spawn_request(start_time)

    def spawn_request(self, timestamp: float):
        raise NotImplementedError("Subclasses must implement spawn_request")

    async def sleep(self, delay: float):
        raise NotImplementedError("Subclasses must implement sleep")

    async def _do_one_request(self, timestamp: float):
        start_perf_counter = perf_counter()
        try:
            await self._do_request_with_timeout()
        except Exception as e:
            self._record_failure(
                timestamp, e, latency=perf_counter() - start_perf_counter
            )
        else:
            end_perf_counter = perf_counter()
            latency = end_perf_counter - start_perf_counter
            self._record_result(timestamp, latency)

    async def _do_request_with_timeout(self):
        raise NotImplementedError("Subclasses must implement _do_request_with_timeout")

    async def __aenter__(self):
        self.exit_stack = AsyncExitStack()
        await self.exit_stack.__aenter__()
        await self.setUp()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._flush_stats()
        print("Final stats:", file=sys.stderr)
        self._overall_stats.print_stats()
        await self.tearDown()
        await self.exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        self._output_file.close()

    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    def _record_result(self, start_time_monotonic: float, latency: float):
        self._output_file.write(f"success,{start_time_monotonic},{latency}\n")
        self._maybe_print_stats()
        self._current_stats.record_result(latency)

    def _record_failure(
        self, start_time_monotonic: float, error: Exception, latency: float
    ):
        self._output_file.write(f"failure,{start_time_monotonic},{repr(error)}\n")
        self._maybe_print_stats()
        self._current_stats.record_failure(latency)
        if self._debug:
            traceback.print_exception(
                type(error), error, error.__traceback__, file=sys.stderr
            )

    def _record_delay(self, start_time_monotonic: float, delay: float):
        self._output_file.write(f"delay,{start_time_monotonic},{delay}\n")
        self._current_stats.record_delay()

    def _maybe_print_stats(self):
        if monotonic() - self._current_stats.start_time >= 5.0:
            self._current_stats.print_stats()
            self._flush_stats()

    def _flush_stats(self):
        self._overall_stats += self._current_stats
        self._current_stats = Stats()


class AsyncioBenchmark(BaseBenchmark):
    async def setUp(self):
        self.exit_stack.enter_context(suppress(asyncio.TimeoutError))
        await self.exit_stack.enter_async_context(
            asyncio.timeout(self._expected_duration + self._timeout)
        )
        self._task_group = await self.exit_stack.enter_async_context(
            asyncio.TaskGroup()
        )
        await super().setUp()

    def spawn_request(self, timestamp: float):
        self._task_group.create_task(self._do_one_request(timestamp))

    async def _do_request_with_timeout(self):
        async with asyncio.timeout(10):
            await self.make_request()

    def run_test(self):
        asyncio.run(self.run_test_in_loop())

    async def sleep(self, delay: float):
        return await asyncio.sleep(delay)


class UvloopBenchmark(AsyncioBenchmark):
    def run_test(self):
        uvloop.run(self.run_test_in_loop())


class TrioBenchmark(BaseBenchmark):
    async def setUp(self):
        self.exit_stack.enter_context(
            trio.move_on_after(self._expected_duration + self._timeout)
        )
        self._nursery = await self.exit_stack.enter_async_context(trio.open_nursery())
        await super().setUp()

    def spawn_request(self, timestamp: float):
        self._nursery.start_soon(self._do_one_request, timestamp)

    async def _do_request_with_timeout(self):
        with trio.fail_after(self._timeout):
            await self.make_request()

    def run_test(self):
        trio.run(self.run_test_in_loop)

    async def sleep(self, delay: float):
        return await trio.sleep(delay)


class HttpxBenchmark(BaseBenchmark):
    async def setUp(self):
        self._client = await self.make_client()
        await super().setUp()

    async def make_client(self) -> httpx.AsyncClient:
        return await self.exit_stack.enter_async_context(
            httpx.AsyncClient(
                http2=True,
                verify=ssl_context,
                timeout=self._timeout,
                limits=httpx.Limits(max_connections=MAX_CONNECTION_POOL_SIZE),
            )
        )

    async def make_request(self):
        if self._body is not None:
            response = await self._client.post(self._url, json=self._body)
        else:
            response = await self._client.get(self._url)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            # 404 is the only status we expect in the test, everything else is an error
            assert response.status_code == 404
        else:
            assert response.status_code == 200
        await response.aread()
        if response.headers["Content-Type"] == "application/json":
            response.json()


class HttpxAsyncioBenchmark(HttpxBenchmark, AsyncioBenchmark):
    pass


class HttpxUvloopBenchmark(HttpxBenchmark, UvloopBenchmark):
    pass


class HttpxTrioBenchmark(HttpxBenchmark, TrioBenchmark):
    pass


class HttpxPyreqwestBenchmark(HttpxBenchmark, AsyncioBenchmark):
    async def make_client(self) -> httpx.AsyncClient:
        pyreqwest_client = await self.exit_stack.enter_async_context(
            pyreqwest.client.ClientBuilder()
            .add_root_certificate_pem(server_ca_cert_pem)
            .timeout(datetime.timedelta(seconds=self._timeout))
            .max_connections(MAX_CONNECTION_POOL_SIZE)
            .http2(True)
            .build()
        )
        return await self.exit_stack.enter_async_context(
            httpx.AsyncClient(
                transport=HttpxPyreqwestTransport(pyreqwest_client),
                timeout=self._timeout,
            )
        )


class HttpxPyreqwestUvloopBenchmark(HttpxPyreqwestBenchmark, UvloopBenchmark):
    pass


class AiohttpHttpxTransportBenchmark(HttpxBenchmark, AsyncioBenchmark):
    async def make_client(self):
        return await self.exit_stack.enter_async_context(
            httpx_aiohttp.HttpxAiohttpClient(
                verify=ssl_context,
                timeout=self._timeout,
                limits=httpx.Limits(max_connections=MAX_CONNECTION_POOL_SIZE),
            )
        )


class AiohttpHttpsTransportUvloopBenchmark(
    AiohttpHttpxTransportBenchmark, UvloopBenchmark
):
    pass


class PyreqwestBenchmark(AsyncioBenchmark):
    async def setUp(self):
        self._client = await self.exit_stack.enter_async_context(
            pyreqwest.client.ClientBuilder()
            .add_root_certificate_pem(server_ca_cert_pem)
            .timeout(datetime.timedelta(seconds=self._timeout))
            .max_connections(MAX_CONNECTION_POOL_SIZE)
            .http2(True)
            .error_for_status(True)
            .build()
        )
        await super().setUp()

    async def make_request(self):
        try:
            if self._body is not None:
                response = (
                    await self._client.post(self._url)
                    .body_json(self._body)
                    .build()
                    .send()
                )
            else:
                response = await self._client.get(self._url).build().send()
        except StatusError as e:
            # 404 is the only status we expect in the test, everything else is an error
            assert e.details["status"] == 404
            return
        else:
            assert response.status == 200
            if response.get_header("Content-Type") == "application/json":
                await response.json()
            else:
                await response.bytes()


class PyreqwestUvloopBenchmark(PyreqwestBenchmark, UvloopBenchmark):
    pass


class AiohttpBenchmark(AsyncioBenchmark):
    async def setUp(self):
        self._client = await self.exit_stack.enter_async_context(
            aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    ssl=ssl_context, limit=MAX_CONNECTION_POOL_SIZE
                ),
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            )
        )
        await super().setUp()

    async def make_request(self):
        if self._body is not None:
            req = self._client.post(self._url, json=self._body)
        else:
            req = self._client.get(self._url)
        async with req as response:
            try:
                response.raise_for_status()
            except aiohttp.ClientResponseError as e:
                # 404 is the only status we expect in the test, everything else is an error
                assert e.status == 404
            else:
                assert response.status == 200
            if response.headers["Content-Type"] == "application/json":
                await response.json()
            else:
                await response.read()


class AiohttpUvloopBenchmark(AiohttpBenchmark, UvloopBenchmark):
    pass


class NiquestsBenchmark(AsyncioBenchmark):
    async def setUp(self):
        self._client = await self.exit_stack.enter_async_context(
            niquests.AsyncSession(
                timeout=self._timeout, pool_maxsize=MAX_CONNECTION_POOL_SIZE
            )
        )
        self._client.verify = server_ca_cert_location
        await super().setUp()

    async def make_request(self):
        if self._body is not None:
            response = await self._client.post(self._url, json=self._body, stream=True)
        else:
            response = await self._client.get(self._url, stream=True)
        try:
            response.raise_for_status()
        except niquests.exceptions.RequestException as e:
            # 404 is the only status we expect in the test, everything else is an error
            assert e.response is not None
            assert e.response.status_code == 404
        else:
            assert response.status_code == 200
        if response.headers["Content-Type"] == "application/json":
            await response.json()
        else:
            await response.content


class NiquestsUvloopBenchmark(NiquestsBenchmark, UvloopBenchmark):
    pass


@dataclass(frozen=True)
class CurlFDState:
    read: bool
    write: bool


FD_STATES = {
    pycurl.POLL_IN: CurlFDState(read=True, write=False),
    pycurl.POLL_OUT: CurlFDState(read=False, write=True),
    pycurl.POLL_INOUT: CurlFDState(read=True, write=True),
    pycurl.POLL_REMOVE: CurlFDState(read=False, write=False),
}
FD_STATES_DEFAULT = CurlFDState(read=False, write=False)


class PyCurlBenchmark(AsyncioBenchmark):
    async def setUp(self):
        self._loop = asyncio.get_running_loop()
        self._multi = pycurl.CurlMulti()
        self._current_fd_states = {}
        self._multi.setopt(pycurl.M_TIMERFUNCTION, self._set_timeout)
        self._multi.setopt(pycurl.M_SOCKETFUNCTION, self._handle_socket)
        self._in_flight_curls: dict[pycurl.Curl, asyncio.Future] = {}
        self._timeout_handle: asyncio.TimerHandle | None = None
        self._curl_limit_remaining = MAX_CONNECTION_POOL_SIZE
        self._reuse_pool = asyncio.Queue(maxsize=MAX_CONNECTION_POOL_SIZE)
        self._stopped = False
        await super().setUp()

    async def tearDown(self):
        # Clean up all curl handles
        await super().tearDown()
        for curl, future in self._in_flight_curls.items():
            curl.close()
            future.cancel()
        while not self._reuse_pool.empty():
            curl = await self._reuse_pool.get()
            curl.close()
        self._multi.close()
        self._stopped = True

    def _set_timeout(self, msecs: int) -> None:
        """Called by libcurl to schedule a timeout."""
        if self._timeout_handle is not None:
            self._timeout_handle.cancel()
        if msecs >= 0:
            self._timeout_handle = self._loop.call_later(
                msecs / 1000.0, self._handle_timeout
            )

    async def _get_curl(self) -> pycurl.Curl:
        if self._reuse_pool.empty() and self._curl_limit_remaining > 0:
            curl = pycurl.Curl()
            curl.setopt(pycurl.SSL_VERIFYPEER, True)
            curl.setopt(pycurl.CAINFO, server_ca_cert_location)
            self._curl_limit_remaining -= 1
            return curl
        else:
            return await self._reuse_pool.get()

    def _handle_socket(
        self, event: int, fd: int, multi: pycurl.CurlMulti, data: bytes
    ) -> None:
        """Called by libcurl when the state of a socket changes."""
        new_state = FD_STATES[event]
        old_state = self._current_fd_states.get(fd, FD_STATES_DEFAULT)
        if event == pycurl.POLL_REMOVE:
            del self._current_fd_states[fd]
        else:
            self._current_fd_states[fd] = new_state
        if old_state.read != new_state.read:
            if new_state.read:
                self._loop.add_reader(
                    fd, self._multi.socket_action, fd, pycurl.CSELECT_IN
                )
            else:
                self._loop.remove_reader(fd)
        if old_state.write != new_state.write:
            if new_state.write:
                self._loop.add_writer(
                    fd, self._multi.socket_action, fd, pycurl.CSELECT_OUT
                )
            else:
                self._loop.remove_writer(fd)
        self._loop.call_soon(self._finish_pending_requests)

    def _handle_timeout(self) -> None:
        """Called by IOLoop when the requested timeout has passed."""
        self._timeout_handle = None
        self._multi.socket_action(pycurl.SOCKET_TIMEOUT, 0)
        self._loop.call_soon(self._finish_pending_requests)

    def _finish_pending_requests(self):
        while not self._stopped:
            num_q, ok_list, err_list = self._multi.info_read()
            for curl in ok_list:
                self._in_flight_curls[curl].set_result(None)
                del self._in_flight_curls[curl]
            for curl, errnum, errmsg in err_list:
                self._in_flight_curls[curl].set_exception(pycurl.error(errnum, errmsg))
                del self._in_flight_curls[curl]
            if num_q == 0:
                break

    async def make_request(self):
        curl = await self._get_curl()
        future = asyncio.Future()
        self._in_flight_curls[curl] = future
        curl.setopt(pycurl.URL, self._url)
        if self._body is not None:
            curl.setopt(pycurl.POST, True)
            curl.setopt(pycurl.HTTPHEADER, ["Content-Type: application/json"])
            curl.setopt(pycurl.POSTFIELDS, orjson.dumps(self._body))

        response: bytes | None = None

        def write_function(data):
            nonlocal response
            if response is None:
                response = data
            else:
                response += data
            return len(data)

        curl.setopt(pycurl.WRITEFUNCTION, write_function)

        headers = {}

        def header_function(header_line):
            header_line = header_line.decode("iso-8859-1")
            if ":" in header_line:
                name, value = header_line.split(":", 1)
                headers[name.strip()] = value.strip()

        curl.setopt(pycurl.HEADERFUNCTION, header_function)

        self._multi.add_handle(curl)
        try:
            await future
            assert curl.getinfo(pycurl.RESPONSE_CODE) in (200, 404)
            if headers.get("content-type") == "application/json":
                assert response is not None
                orjson.loads(response)
        finally:
            self._multi.remove_handle(curl)
            self._reuse_pool.put_nowait(curl)


class PyCurlUvloopBenchmark(PyCurlBenchmark, UvloopBenchmark):
    pass


TEST_CLASSES = {
    "httpx_asyncio": HttpxAsyncioBenchmark,
    "httpx_uvloop": HttpxUvloopBenchmark,
    "httpx_trio": HttpxTrioBenchmark,
    "httpx_pyreqwest": HttpxPyreqwestBenchmark,
    "httpx_pyreqwest_uvloop": HttpxPyreqwestUvloopBenchmark,
    "httpx_aiohttp": AiohttpHttpxTransportBenchmark,
    "httpx_aiohttp_uvloop": AiohttpHttpsTransportUvloopBenchmark,
    "pyreqwest": PyreqwestBenchmark,
    "pyreqwest_uvloop": PyreqwestUvloopBenchmark,
    "aiohttp": AiohttpBenchmark,
    "aiohttp_uvloop": AiohttpUvloopBenchmark,
    "niquests": NiquestsBenchmark,
    "niquests_uvloop": NiquestsUvloopBenchmark,
    "pycurl": PyCurlBenchmark,
    "pycurl_uvloop": PyCurlUvloopBenchmark,
}


class ServerTypes(StrEnum):
    HTTPS2 = "https2"
    HTTPS1 = "https1"
    HTTP1 = "http1"


WORKERS = 4


@contextmanager
def run_server(type_: ServerTypes = ServerTypes.HTTPS2):
    cmd = [
        "granian",
        "server:app",
        "--interface",
        "asginl",
        "--workers",
        f"{WORKERS}",
    ]
    ssl_opts = [
        "--ssl-keyfile",
        "server.key",
        "--ssl-certificate",
        "server.crt",
        "--port",
        "8443",
    ]
    base_url = "https://localhost:8443"
    match type_:
        case "https2":
            cmd += ssl_opts
        case "https1":
            cmd += ssl_opts + ["--http", "1"]
        case "http1":
            cmd += ["--port", "8000", "--http", "1"]
            base_url = "http://localhost:8000"
    with open("results/server.log", "wb") as log_file:
        process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        for _ in range(10):
            try:
                urlopen(base_url + "/hello", context=ssl_context, timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        else:
            process.terminate()
            raise RuntimeError("Server failed to start")
        try:
            yield base_url
        finally:
            process.terminate()

        try:
            process.wait(5)
        except subprocess.TimeoutExpired:
            subprocess.run(["pkill", "-SIGKILL", "-P", str(process.pid)])


@dataclass
class Endpoint:
    path: str
    body: dict | None


ENDPOINTS = {
    "hello": Endpoint("/hello", None),
    "json": Endpoint("/json", None),
    "chunked": Endpoint("/chunked", None),
    "post": Endpoint("/post", {f"key{i}": ["value"] * 20 for i in range(20)}),
    "latency": Endpoint("/latency", None),
    "notfound": Endpoint("/notfound", None),
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-type",
        type=ServerTypes,
        choices=list(ServerTypes),
        default=ServerTypes.HTTPS2,
    )
    parser.add_argument(
        "--endpoint", type=str, choices=ENDPOINTS.keys(), default="hello"
    )
    parser.add_argument(
        "--test-class", type=str, choices=TEST_CLASSES.keys(), default="httpx_asyncio"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Duration of the test in seconds"
    )
    parser.add_argument(
        "--initial-rate",
        type=float,
        default=1.0,
        help="Initial request rate (requests per second)",
    )
    parser.add_argument(
        "--final-rate",
        type=float,
        default=1.0,
        help="Final request rate (requests per second)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print debug information during the test"
    )
    args = parser.parse_args()

    resource.setrlimit(resource.RLIMIT_NOFILE, (100000, 100000))
    with run_server(args.server_type) as base_url:
        endpoint = ENDPOINTS[args.endpoint]
        test_class = TEST_CLASSES[args.test_class]
        benchmark = test_class(
            start_time_generator=poisson_process(
                args.duration, args.initial_rate, args.final_rate
            ),
            output_file=f"results/{args.test_class}_{args.endpoint}_{args.server_type}.csv",
            url=base_url + endpoint.path,
            expected_duration=args.duration,
            body=endpoint.body,
            debug=args.debug,
        )
        benchmark.run_test()
