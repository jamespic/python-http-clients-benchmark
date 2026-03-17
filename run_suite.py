import datetime
from argparse import ArgumentParser
from subprocess import run

from benchmark import ENDPOINTS, TEST_CLASSES, ServerTypes

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Duration to run each benchmark for (in seconds)",
    )
    argparser.add_argument(
        "--initial-rate",
        type=float,
        default=0.0,
        help="Initial request rate for benchmarks (in requests per second)",
    )
    argparser.add_argument(
        "--final-rate",
        type=float,
        default=600.0,
        help="Final request rate for benchmarks (in requests per second)",
    )
    argparser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the benchmark suite if any individual benchmark fails",
    )
    argparser.add_argument(
        "--run-threaded-benchmarks-with-cpu-quota",
        default=None,
        type=int,
        help="If set, run benchmarks with _threaded in their name with a CPU quota",
    )
    args = argparser.parse_args()

    expected_duration = (
        args.duration * len(ENDPOINTS) * len(ServerTypes) * len(TEST_CLASSES)
    )
    expected_completion_time = datetime.datetime.now() + datetime.timedelta(
        seconds=expected_duration
    )
    print(
        f"Running benchmark suite. Expected completion time: {expected_completion_time:%Y-%m-%d %H:%M:%S}"
    )
    print()

    for endpoint in ENDPOINTS:
        for server_type in ServerTypes:
            for test_class in TEST_CLASSES:
                msg = f"Testing {test_class} against {server_type} ({endpoint})"
                print(msg)
                print("-" * len(msg))
                run(
                    [
                        "python",
                        "benchmark.py",
                        "--server-type",
                        server_type.value,
                        "--endpoint",
                        endpoint,
                        "--test-class",
                        test_class,
                        "--duration",
                        str(args.duration),
                        "--initial-rate",
                        str(args.initial_rate),
                        "--final-rate",
                        str(args.final_rate),
                        *(
                            [
                                "--run-threaded-benchmarks-with-cpu-quota",
                                str(args.run_threaded_benchmarks_with_cpu_quota),
                            ]
                            if args.run_threaded_benchmarks_with_cpu_quota is not None
                            else []
                        ),
                    ],
                    check=args.stop_on_error,
                )
                print()
