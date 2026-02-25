import os
from collections import Counter, defaultdict
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from os.path import basename, splitext
from typing import TypedDict, cast

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
    overall_average_latency: float

    @property
    def max_throughput(self) -> float:
        return max((result.throughput for result in self.binned_results), default=0.0)

    @property
    def breaking_point(self) -> float:
        return min(
            (
                result.throughput
                for result in self.binned_results
                if result.failure_count > 0.05 * result.success_count
            ),
            default=self.max_throughput,
        )


def read_and_bin_data(filename: str, bin_size: float = 1.0) -> BenchmarkResult:
    client_under_test, endpoint, server_type = splitext(basename(filename))[0].rsplit(
        "_", 2
    )
    successes = defaultdict[float, list[float]](list)
    failures_by_time = Counter[float]()
    failures_by_message = Counter[str]()
    delay_counts = Counter[float]()
    total_success_latency = 0.0
    total_successes_count = 0
    with open(filename, "r") as file_:
        for row in file_:
            type_, timestamp, data = row.strip().split(",", 2)
            time_bin = int(float(timestamp) // bin_size)
            match type_:
                case "success":
                    latency = float(data)
                    successes[time_bin].append(latency)
                    total_success_latency += latency
                    total_successes_count += 1
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
        overall_average_latency=(
            total_success_latency / total_successes_count
            if total_successes_count > 0
            else 0.0
        ),
    )


class PerBenchmarkGraphs(TypedDict):
    success_failure: pygal.Line
    average_latency: pygal.Line
    latency_percentiles: pygal.Line


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


def produce_per_benchmark_graphs(
    benchmark_result: BenchmarkResult,
) -> PerBenchmarkGraphs:
    time_bins = get_time_bins([benchmark_result])
    success_counts = [
        result.success_count for result in benchmark_result.binned_results
    ]
    failure_counts = [
        result.failure_count for result in benchmark_result.binned_results
    ]
    average_latencies = [
        {
            "value": result.average_latency,
            "ci": {
                "type": "continuous",
                "stddev": result.standard_deviation_latency,
                "sample_size": result.success_count,
            },
        }
        for result in benchmark_result.binned_results
    ]
    latency_percentiles = {
        p: [result.latency_percentiles[p] for result in benchmark_result.binned_results]
        for p in PERCENTILES
    }
    delay_counts = [result.delay_count for result in benchmark_result.binned_results]

    success_failure_graph = pygal.Line(title="Success and Failure Counts Over Time")
    success_failure_graph.x_labels = time_bins
    success_failure_graph.add("Successes", success_counts)
    success_failure_graph.add("Failures", failure_counts)
    success_failure_graph.add("Delays", delay_counts)

    latency_graph = pygal.Line(title="Average Latency Over Time", logarithmic=True)
    latency_graph.x_labels = time_bins
    latency_graph.add("Average Latency", average_latencies)

    percentile_graph = pygal.Line(
        title="Percentile Latency Over Time", logarithmic=True
    )
    percentile_graph.x_labels = time_bins
    for p in PERCENTILES:
        percentile_graph.add(f"{p}th Percentile Latency", latency_percentiles[p])

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
    throughput: pygal.Line
    average_latency: pygal.Line
    p90_latency: pygal.Line
    p99_latency: pygal.Line


def produce_per_server_endpoint_graphs(
    benchmark_results: Iterable[BenchmarkResult],
) -> PerServerEndpointGraphs:
    time_bins = get_time_bins(benchmark_results)
    throughput_series: defaultdict[str, dict[float, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    average_latency_series: defaultdict[str, dict[float, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    p90_latency_series: defaultdict[str, dict[float, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    p99_latency_series: defaultdict[str, dict[float, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for benchmark_result in benchmark_results:
        for binned_result in benchmark_result.binned_results:
            throughput_series[benchmark_result.client_under_test][
                binned_result.time_bin
            ] = binned_result.throughput
            average_latency_series[benchmark_result.client_under_test][
                binned_result.time_bin
            ] = binned_result.average_latency
            p90_latency_series[benchmark_result.client_under_test][
                binned_result.time_bin
            ] = binned_result.latency_percentiles[90]
            p99_latency_series[benchmark_result.client_under_test][
                binned_result.time_bin
            ] = binned_result.latency_percentiles[99]

    throughput_graph = pygal.Line(title="Throughput Over Time")
    throughput_graph.x_labels = time_bins
    for client_under_test, series in throughput_series.items():
        throughput_graph.add(
            client_under_test, [series.get(time_bin, None) for time_bin in time_bins]
        )

    average_latency_graph = pygal.Line(
        title="Average Latency Over Time", logarithmic=True
    )
    average_latency_graph.x_labels = time_bins
    for client_under_test, series in average_latency_series.items():
        average_latency_graph.add(
            client_under_test, [series.get(time_bin, None) for time_bin in time_bins]
        )

    p90_latency_graph = pygal.Line(
        title="90th Percentile Latency Over Time", logarithmic=True
    )
    p90_latency_graph.x_labels = time_bins
    for client_under_test, series in p90_latency_series.items():
        p90_latency_graph.add(
            client_under_test, [series.get(time_bin, None) for time_bin in time_bins]
        )

    p99_latency_graph = pygal.Line(
        title="99th Percentile Latency Over Time", logarithmic=True
    )
    p99_latency_graph.x_labels = time_bins
    for client_under_test, series in p99_latency_series.items():
        p99_latency_graph.add(
            client_under_test, [series.get(time_bin, None) for time_bin in time_bins]
        )

    return {
        "throughput": throughput_graph,
        "average_latency": average_latency_graph,
        "p90_latency": p90_latency_graph,
        "p99_latency": p99_latency_graph,
    }


class OverallGraphs(TypedDict):
    breaking_point: pygal.Bar
    max_throughput: pygal.Bar


def produce_overall_graphs(
    benchmark_results: Iterable[BenchmarkResult],
) -> OverallGraphs:
    x_labels = set(
        _server_and_endpoint_description(result) for result in benchmark_results
    )
    breaking_point_series: defaultdict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    max_throughput_series: defaultdict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for result in benchmark_results:
        breaking_point_series[result.client_under_test][
            _server_and_endpoint_description(result)
        ] = result.breaking_point
        max_throughput_series[result.client_under_test][
            _server_and_endpoint_description(result)
        ] = result.max_throughput

    breaking_point_graph = pygal.Bar(
        title="Breaking Point (Throughput where failures breach 1%)", x_labels=x_labels
    )
    for client_under_test, series in breaking_point_series.items():
        breaking_point_graph.add(
            client_under_test, [series.get(label, None) for label in x_labels]
        )
    max_throughput_graph = pygal.Bar(
        title="Maximum Throughput Achieved", x_labels=x_labels
    )
    for client_under_test, series in max_throughput_series.items():
        max_throughput_graph.add(
            client_under_test, [series.get(label, None) for label in x_labels]
        )

    return {
        "breaking_point": breaking_point_graph,
        "max_throughput": max_throughput_graph,
    }


def _server_and_endpoint_description(result: BenchmarkResult) -> str:
    return f"{result.endpoint}_{result.server_type}"


def write_report(
    benchmark_results: Iterable[BenchmarkResult], filename: str = "report.html"
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
            min_overall_latencies = defaultdict[str, float](lambda: float("inf"))
            max_overall_latencies = defaultdict[str, float](float)
            for result in benchmark_results:
                if result.server_type == server_type:
                    max_max_throughputs[result.endpoint] = max(
                        max_max_throughputs[result.endpoint], result.max_throughput
                    )
                    min_overall_latencies[result.endpoint] = min(
                        min_overall_latencies[result.endpoint],
                        result.overall_average_latency,
                    )
                    max_overall_latencies[result.endpoint] = max(
                        max_overall_latencies[result.endpoint],
                        result.overall_average_latency,
                    )
            with tag("h2"):
                w(f"Server Type: {server_type}")

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
                                    with tag("li"):
                                        w(
                                            f'<a href="graphs/{endpoint}_{server_type}_throughput.svg">Throughput</a>'
                                        )
                                    with tag("li"):
                                        w(
                                            f'<a href="graphs/{endpoint}_{server_type}_average_latency.svg">Average Latency</a>'
                                        )
                                    with tag("li"):
                                        w(
                                            f'<a href="graphs/{endpoint}_{server_type}_p90_latency.svg">P90 Latency</a>'
                                        )
                                    with tag("li"):
                                        w(
                                            f'<a href="graphs/{endpoint}_{server_type}_p99_latency.svg">P99 Latency</a>'
                                        )
                    for client_under_test in clients_under_test:
                        with tag("tr"):
                            w(f"<td>{client_under_test}</td>")
                            for endpoint in endpoints:
                                endpoint_max_throughput = max_max_throughputs[endpoint]
                                min_overall_latency = min_overall_latencies[endpoint]
                                max_overall_latency = max_overall_latencies[endpoint]
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
                                            with tag(
                                                "li",
                                                style=f"color: {choose_colour_bad_to_good(1 - (result.overall_average_latency - min_overall_latency) / (max_overall_latency - min_overall_latency) if max_overall_latency > min_overall_latency else 0.0)}",
                                            ):
                                                w(
                                                    f"Overall Average Latency: {result.overall_average_latency:.3f} s"
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


def main():
    with ThreadPoolExecutor() as executor:
        benchmark_results = list(
            executor.map(
                lambda filename: read_and_bin_data(f"results/{filename}"),
                [
                    filename
                    for filename in os.listdir("results")
                    if filename.endswith(".csv")
                ],
            )
        )
        executor.map(save_per_benchmark_graphs, benchmark_results)

    grouped_by_server_endpoint = defaultdict[str, list[BenchmarkResult]](list)
    for result in benchmark_results:
        grouped_by_server_endpoint[_server_and_endpoint_description(result)].append(
            result
        )
    for server_endpoint, group in grouped_by_server_endpoint.items():
        graphs = cast(dict[str, pygal.Graph], produce_per_server_endpoint_graphs(group))
        for graph_name, graph in graphs.items():
            graph.render_to_file(f"graphs/{server_endpoint}_{graph_name}.svg")
    overall_graphs = cast(
        dict[str, pygal.Graph], produce_overall_graphs(benchmark_results)
    )
    for graph_name, graph in overall_graphs.items():
        graph.render_to_file(f"graphs/overall_{graph_name}.svg")
    write_report(benchmark_results)


if __name__ == "__main__":
    main()
