import re
import argparse
import os
from collections import Counter, defaultdict
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from os.path import basename, splitext
from typing import Callable, TypedDict, cast

import numpy as np
import pygal


@dataclass
class BinnedResult:
    time_bin: float
    success_count: int
    failure_count: int
    average_latency: float
    standard_deviation_latency: float
    latency_percentiles: dict[float, float]
    delay_count: int
    bin_size: float = 1.0

    @property
    def throughput(self) -> float:
        return self.success_count / self.bin_size if self.bin_size > 0 else 0.0


PERCENTILES = (25, 50, 75, 90, 95, 99)


@dataclass
class BenchmarkResult:
    endpoint: str
    server_type: str
    client_under_test: str
    binned_results: list[BinnedResult]
    top_failure_messages: dict[str, int]
    def latency_at_rps(self, rps: float) -> float:
        for result in self.binned_results:
            if result.throughput >= rps:
                return result.average_latency
        else:
            return max((result.average_latency for result in self.binned_results), default=10.0)

    @property
    def max_throughput(self) -> float:
        return max((result.throughput for result in self.binned_results), default=0.0)

    @property
    def breaking_point(self) -> float:
        for result in self.binned_results:
                if result.failure_count > 0.05 * result.success_count:
                    return result.throughput
        else:
            return self.max_throughput


def read_and_bin_data(filename: str, bin_size: float = 1.0) -> BenchmarkResult:
    client_under_test, endpoint, server_type = splitext(basename(filename))[0].rsplit(
        "_", 2
    )
    successes = defaultdict[float, list[float]](list)
    failures_by_time = Counter[float]()
    failures_by_message = Counter[str]()
    delay_counts = Counter[float]()
    with open(filename, "r") as file_:
        for row in file_:
            type_, timestamp, data = row.strip().split(",", 2)
            time_bin = int(float(timestamp) // bin_size) * bin_size
            match type_:
                case "success":
                    latency = float(data)
                    successes[time_bin].append(latency)
                case "failure":
                    failures_by_time[time_bin] += 1
                    failures_by_message[data] += 1
                case "delay":
                    delay_counts[time_bin] += 1

    binned_results = []
    for time_bin in sorted(set(successes) | set(failures_by_time) | set(delay_counts)):
        latencies = successes.get(time_bin, [])
        success_count = len(latencies)
        failure_count = failures_by_time.get(time_bin, 0)
        delay_count = delay_counts.get(time_bin, 0)
        average_latency = float(np.mean(latencies)) if latencies else 0.0
        standard_deviation_latency = float(np.std(latencies)) if latencies else 0.0
        latency_percentiles = {
            p: float(np.percentile(latencies, p)) if latencies else 0.0
            for p in PERCENTILES
        }
        binned_results.append(
            BinnedResult(
                time_bin=time_bin,
                success_count=success_count,
                failure_count=failure_count,
                average_latency=average_latency,
                standard_deviation_latency=standard_deviation_latency,
                latency_percentiles=latency_percentiles,
                delay_count=delay_count,
                bin_size=bin_size,
            )
        )

    return BenchmarkResult(
        endpoint=endpoint,
        server_type=server_type,
        client_under_test=client_under_test,
        binned_results=binned_results,
        top_failure_messages=dict(failures_by_message.most_common(10)),
    )


class PerBenchmarkGraphs(TypedDict):
    success_failure: pygal.TimeDeltaLine
    average_latency: pygal.TimeDeltaLine
    latency_percentiles: pygal.TimeDeltaLine


def get_time_bins(benchmark_results: Iterable[BenchmarkResult]) -> list[float]:
    bins = sorted(
        set(
            result.time_bin
            for benchmark_result in benchmark_results
            for result in benchmark_result.binned_results
        )
    )
    # A few tests seem to run over, with weird sparse data at the end. Remove any bins more than 5 steps past the previous bin
    last_step = None
    last_bin = None
    filtered_bins = []
    for bin_ in bins:
        if (
            last_step is not None
            and last_bin is not None
            and bin_ - last_bin > 5 * last_step
        ):
            break
        if last_bin is not None:
            last_step = bin_ - last_bin
        last_bin = bin_
        filtered_bins.append(bin_)
    return filtered_bins


def get_test_end_time(benchmark_results: Iterable[BenchmarkResult]) -> float:
    bins = sorted(
        set(
            result.time_bin
            for benchmark_result in benchmark_results
            for result in benchmark_result.binned_results
        )
    )
    # A few tests seem to run over, with weird sparse data at the end. Remove any bins more than 5 steps past the previous bin
    last_step = None
    last_bin = None
    for bin_ in bins:
        if (
            last_step is not None
            and last_bin is not None
            and bin_ - last_bin > 5 * last_step
        ):
            return last_bin
        if last_bin is not None:
            last_step = bin_ - last_bin
        last_bin = bin_
    return last_bin if last_bin is not None else 0.0


def produce_per_benchmark_graphs(
    benchmark_result: BenchmarkResult,
) -> PerBenchmarkGraphs:
    end_time = get_test_end_time([benchmark_result])
    truncated_binned_results = [
        result
        for result in benchmark_result.binned_results
        if result.time_bin <= end_time
    ]

    success_failure_graph = pygal.TimeDeltaLine(
        title="Success and Failure Counts Over Time"
    )

    success_failure_graph.add(
        "Successes",
        [
            (result.time_bin, result.success_count)
            for result in truncated_binned_results
        ],
    )
    success_failure_graph.add(
        "Failures",
        [
            (result.time_bin, result.failure_count)
            for result in truncated_binned_results
        ],
    )
    success_failure_graph.add(
        "Delays",
        [(result.time_bin, result.delay_count) for result in truncated_binned_results],
    )

    latency_graph = pygal.TimeDeltaLine(title="Average Latency Over Time")
    latency_graph.add(
        "Average Latency",
        [
            (result.time_bin, result.average_latency)
            for result in truncated_binned_results
        ],
    )

    percentile_graph = pygal.TimeDeltaLine(title="Percentile Latency Over Time")

    for p in PERCENTILES:
        percentile_graph.add(
            f"{p}th Percentile Latency",
            [
                (result.time_bin, result.latency_percentiles[p])
                for result in truncated_binned_results
            ],
        )

    return {
        "success_failure": success_failure_graph,
        "average_latency": latency_graph,
        "latency_percentiles": percentile_graph,
    }


def save_per_benchmark_graphs(benchmark_result: BenchmarkResult) -> None:
    graphs = cast(
        dict[str, pygal.Graph], produce_per_benchmark_graphs(benchmark_result)
    )
    for graph_name, graph in graphs.items():
        graph.render_to_file(
            f"graphs/{benchmark_result.client_under_test}_{benchmark_result.endpoint}_{benchmark_result.server_type}_{graph_name}.svg"
        )


class PerServerEndpointGraphs(TypedDict):
    throughput: pygal.TimeDeltaLine
    average_latency: pygal.TimeDeltaLine
    p90_latency: pygal.TimeDeltaLine
    p99_latency: pygal.TimeDeltaLine


def produce_per_server_endpoint_graphs(
    server: str,
    endpoint: str,
    benchmark_results: Iterable[BenchmarkResult],
) -> PerServerEndpointGraphs:
    end_time = get_test_end_time(benchmark_results)

    def _make_graph(
        title: str, value_extractor: Callable[[BinnedResult], float]
    ) -> pygal.TimeDeltaLine:
        graph = pygal.TimeDeltaLine(title=title)
        for benchmark_result in sorted(
            benchmark_results, key=lambda r: r.client_under_test
        ):
            graph.add(
                benchmark_result.client_under_test,
                [
                    (result.time_bin, value_extractor(result))
                    for result in benchmark_result.binned_results
                    if result.time_bin <= end_time
                ],
            )
        return graph

    return {
        "throughput": _make_graph(
            "Throughput Over Time", lambda result: result.throughput
        ),
        "average_latency": _make_graph(
            "Average Latency Over Time",
            lambda result: result.average_latency,
        ),
        "p90_latency": _make_graph(
            "90th Percentile Latency Over Time",
            lambda result: result.latency_percentiles[90],
        ),
        "p99_latency": _make_graph(
            "99th Percentile Latency Over Time",
            lambda result: result.latency_percentiles[99],
        ),
    }


class PerServerTypeGraphs(TypedDict):
    breaking_point: pygal.Bar
    max_throughput: pygal.Bar


def produce_per_server_type_graphs(
    benchmark_results: Iterable[BenchmarkResult],
) -> PerServerTypeGraphs:
    results_by_client_and_endpoint = {
        (result.client_under_test, result.endpoint): result
        for result in benchmark_results
    }
    clients = sorted(set(result.client_under_test for result in benchmark_results))
    endpoints = sorted(set(result.endpoint for result in benchmark_results))

    def _make_graph(
        title: str, value_extractor: Callable[[BenchmarkResult], float]
    ) -> pygal.Bar:
        graph = pygal.Bar(title=title, x_labels=endpoints)
        for client in clients:
            graph.add(
                client,
                [
                    (
                        value_extractor(result)
                        if (
                            result := results_by_client_and_endpoint.get(
                                (client, endpoint)
                            )
                        )
                        else 0.0
                    )
                    for endpoint in endpoints
                ],
            )
        return graph

    return {
        "breaking_point": _make_graph(
            "Breaking Point (Throughput where failures breach 5%)",
            lambda result: result.breaking_point,
        ),
        "max_throughput": _make_graph(
            "Maximum Throughput Achieved", lambda result: result.max_throughput
        ),
    }


def write_report(
    benchmark_results: Iterable[BenchmarkResult], filename: str = "report.html", rps_for_latency: float = 500.0
) -> None:
    with open(filename, "w") as file_:
        indent = 0

        @contextmanager
        def tag(tag_name: str, **attrs) -> Generator[None, None, None]:
            nonlocal indent
            attr_str = " ".join(f'{key}="{value}"' for key, value in attrs.items())
            file_.write(f"{' ' * indent}<{tag_name} {attr_str}>\n")
            indent += 2
            yield
            indent -= 2
            file_.write(f"{' ' * indent}</{tag_name}>\n")

        def w(text: str) -> None:
            file_.write(" " * indent + text + "\n")

        w("<html><head><title>Benchmark Report</title></head><body>")
        w("<h1>Benchmark Report</h1>")
        server_types = sorted(set(result.server_type for result in benchmark_results))
        endpoints = sorted(set(result.endpoint for result in benchmark_results))
        clients_under_test = sorted(
            set(result.client_under_test for result in benchmark_results)
        )
        results_lookup = {
            (result.server_type, result.endpoint, result.client_under_test): result
            for result in benchmark_results
        }
        for server_type in server_types:
            max_max_throughputs = defaultdict[str, float](float)
            min_latencies_at_rps = defaultdict[str, float](lambda: float("inf"))
            max_latencies_at_rps = defaultdict[str, float](float)
            for result in benchmark_results:
                if result.server_type == server_type:
                    max_max_throughputs[result.endpoint] = max(
                        max_max_throughputs[result.endpoint], result.max_throughput
                    )
                    min_latencies_at_rps[result.endpoint] = min(
                        min_latencies_at_rps[result.endpoint],
                        result.latency_at_rps(rps_for_latency),
                    )
                    max_latencies_at_rps[result.endpoint] = max(
                        max_latencies_at_rps[result.endpoint],
                        result.latency_at_rps(rps_for_latency),
                    )
            with tag("h2"):
                w(f"Server Type: {server_type}")

            with tag("p"):
                w(
                    f"<object data='graphs/{server_type}_breaking_point.svg' width='800' height='600' type='image/svg+xml'></object>"
                )
            with tag("p"):
                w(
                    f"<object data='graphs/{server_type}_max_throughput.svg' width='800' height='600' type='image/svg+xml'></object>"
                )

            with tag("table", border="1", cellspacing="0", cellpadding="5"):
                with tag("tr"):
                    w("<th>Client Under Test</th>")
                    for endpoint in endpoints:
                        w(f"<th>{endpoint}</th>")
                with tag("tbody"):
                    with tag("tr"):
                        with tag("td"):
                            w("Graphs")
                        for endpoint in endpoints:
                            with tag("td"):
                                with tag("ul"):
                                    for x in [
                                        "throughput",
                                        "average_latency",
                                        "p90_latency",
                                        "p99_latency",
                                    ]:
                                        with tag("li"):
                                            w(
                                                f'<a href="graphs/{endpoint}_{server_type}_{x}.svg">{x.replace("_", " ").title()}</a>'
                                            )
                    for client_under_test in clients_under_test:
                        with tag("tr"):
                            w(f"<td>{client_under_test}</td>")
                            for endpoint in endpoints:
                                endpoint_max_throughput = max_max_throughputs[endpoint]
                                min_latency_at_rps = min_latencies_at_rps[endpoint]
                                max_latency_at_rps = max_latencies_at_rps[endpoint]
                                result = results_lookup.get(
                                    (server_type, endpoint, client_under_test)
                                )
                                with tag("td"):
                                    if result is not None:
                                        with tag("ul"):
                                            with tag(
                                                "li",
                                                style=f"color: {choose_colour_bad_to_good(result.breaking_point / endpoint_max_throughput if endpoint_max_throughput > 0 else 0.0)}",
                                            ):
                                                w(
                                                    f"Breaking Point: {result.breaking_point:.2f} rps"
                                                )
                                            with tag(
                                                "li",
                                                style=f"color: {choose_colour_bad_to_good(result.max_throughput / endpoint_max_throughput if endpoint_max_throughput > 0 else 0.0)}",
                                            ):
                                                w(
                                                    f"Max Throughput: {result.max_throughput:.2f} rps"
                                                )
                                            latency_at_rps = result.latency_at_rps(rps_for_latency)
                                            with tag(
                                                "li",
                                                style=f"color: {choose_colour_bad_to_good(1 - (latency_at_rps - min_latency_at_rps) / (max_latency_at_rps - min_latency_at_rps) if max_latency_at_rps > min_latency_at_rps else 0.0)}",
                                            ):
                                                w(
                                                    f"Latency at {rps_for_latency} rps: {latency_at_rps:.3f} s"
                                                )
                                            with tag("li"):
                                                with tag(
                                                    "a",
                                                    href=f"#details-{server_type}-{endpoint}-{client_under_test}",
                                                ):
                                                    w("Details")
                                    else:
                                        w("N/A")
        with tag("h2"):
            w("Detailed Results")
        for result in benchmark_results:
            with tag(
                "div",
                id=f"details-{result.server_type}-{result.endpoint}-{result.client_under_test}",
            ):
                with tag("h3"):
                    w(
                        f"{result.client_under_test} against {result.endpoint} ({result.server_type})"
                    )
                with tag("p"):
                    w(f"Breaking Point: {result.breaking_point:.2f} rps")
                with tag("p"):
                    w(f"Max Throughput: {result.max_throughput:.2f} rps")
                with tag("h4"):
                    w("Top Failure Messages")
                with tag("ul"):
                    for message, count in result.top_failure_messages.items():
                        with tag("li"):
                            w(f"{message}: {count} occurrences")
                with tag("h4"):
                    w("Graphs")
                with tag("ul"):
                    with tag("li"):
                        w(
                            f'<a href="graphs/{result.client_under_test}_{result.endpoint}_{result.server_type}_success_failure.svg">Success/Failure Counts</a>'
                        )
                    with tag("li"):
                        w(
                            f'<a href="graphs/{result.client_under_test}_{result.endpoint}_{result.server_type}_average_latency.svg">Average Latency</a>'
                        )
                    with tag("li"):
                        w(
                            f'<a href="graphs/{result.client_under_test}_{result.endpoint}_{result.server_type}_latency_percentiles.svg">Latency Percentiles</a>'
                        )
        w("</body></html>")


def choose_colour_bad_to_good(value: float) -> str:
    # 0.0 -> red, 1.0 -> green
    red = int((1.0 - value) * 255)
    green = int(value * 255)
    return f"rgb({red}, {green}, 0)"


def main(bin_size: float = 1.0) -> None:
    with ThreadPoolExecutor() as executor:
        benchmark_results: list[BenchmarkResult] = list(
            executor.map(
                lambda filename: read_and_bin_data(f"results/{filename}", bin_size),
                [
                    filename
                    for filename in os.listdir("results")
                    if filename.endswith(".csv")
                ],
            )
        )
        list(executor.map(save_per_benchmark_graphs, benchmark_results))

    grouped_by_server_endpoint = defaultdict[tuple[str, str], list[BenchmarkResult]](
        list
    )
    for result in benchmark_results:
        grouped_by_server_endpoint[(result.server_type, result.endpoint)].append(result)
    for (server, endpoint), group in grouped_by_server_endpoint.items():
        graphs = cast(
            dict[str, pygal.Graph],
            produce_per_server_endpoint_graphs(server, endpoint, group),
        )
        for graph_name, graph in graphs.items():
            graph.render_to_file(f"graphs/{endpoint}_{server}_{graph_name}.svg")

    grouped_by_server_type = defaultdict[str, list[BenchmarkResult]](list)
    for result in benchmark_results:
        grouped_by_server_type[result.server_type].append(result)
    for server_type, group in grouped_by_server_type.items():
        overall_graphs = cast(
            dict[str, pygal.Graph], produce_per_server_type_graphs(group)
        )
        for graph_name, graph in overall_graphs.items():
            graph.render_to_file(f"graphs/{server_type}_{graph_name}.svg")
    write_report(benchmark_results)


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-size", type=float, default=1.0, help="Bin size for data aggregation")
    args = parser.parse_args()

    main(bin_size=args.bin_size)