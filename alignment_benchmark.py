
import time
import random

from global_alignment_cpu import needleman_wunsch
from global_alignment_gpu import gpu_needleman_wunsch


def generate_dna(length, seed):
    random.seed(seed)
    return "".join(random.choices("ATGC", k=length))


lengths = [8, 32, 64, 128]

print("=" * 65)
print("          GENEWEAVER ALIGNMENT BENCHMARK")
print("=" * 65)

print(f"{'Length':<12}{'CPU (ms)':<18}{'GPU (ms)':<18}{'Match':<10}")
print("-" * 65)

for length in lengths:
    dna1 = generate_dna(length, 10)
    dna2 = generate_dna(length, 20)

    # CPU benchmark
    start = time.perf_counter()
    cpu_result1, cpu_result2, cpu_score = needleman_wunsch(dna1, dna2)
    cpu_time = (time.perf_counter() - start) * 1000

    # GPU benchmark
    start = time.perf_counter()
    gpu_result1, gpu_result2, gpu_score, gpu_name = gpu_needleman_wunsch(
        dna1, dna2
    )
    gpu_time = (time.perf_counter() - start) * 1000

    # Validate results
    matched = (
        cpu_result1 == gpu_result1
        and cpu_result2 == gpu_result2
        and cpu_score == gpu_score
    )

    print(
        f"{length:<12}"
        f"{cpu_time:<18.6f}"
        f"{gpu_time:<18.6f}"
        f"{'PASS' if matched else 'FAIL':<10}"
    )

print("-" * 65)
print("GPU:", gpu_name)
print("Benchmark completed.")
print("=" * 65)
