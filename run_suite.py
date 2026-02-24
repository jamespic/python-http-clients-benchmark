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
    args = argparser.parse_args()
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
                    ],
                    check=args.stop_on_error,
                )
                print()
