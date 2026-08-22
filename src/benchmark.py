import statistics
import time

import httpx


URL = "http://127.0.0.1:8000/predict"

PAYLOAD = {
    "text": (
        "Patient presenting acute myocardial "
        "infarction with severe chest pain."
    )
}

N_REQUESTS = 100
WARMUP_REQUESTS = 10


def percentile(values: list[float], percentile: float) -> float:
    sorted_values = sorted(values)

    index = int(
        percentile * (len(sorted_values) - 1)
    )

    return sorted_values[index]


def main():
    latencies = []

    with httpx.Client() as client:

        print(
            f"Running {WARMUP_REQUESTS} warmup requests..."
        )

        for _ in range(WARMUP_REQUESTS):
            client.post(
                URL,
                json=PAYLOAD,
            )

        print(
            f"Running {N_REQUESTS} benchmark requests..."
        )

        for _ in range(N_REQUESTS):
            start = time.perf_counter()

            response = client.post(
                URL,
                json=PAYLOAD,
            )

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            response.raise_for_status()

            latencies.append(elapsed)

    mean_latency = statistics.mean(latencies)

    p95_latency = percentile(
        latencies,
        0.95,
    )

    print("\nBenchmark results")
    print("-----------------")
    print(f"Requests: {N_REQUESTS}")
    print(f"Mean: {mean_latency:.2f} ms")
    print(f"P95: {p95_latency:.2f} ms")
    print(f"Min: {min(latencies):.2f} ms")
    print(f"Max: {max(latencies):.2f} ms")


if __name__ == "__main__":
    main()