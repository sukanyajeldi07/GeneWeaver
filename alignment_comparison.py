
import time

from global_alignment_cpu import needleman_wunsch
from global_alignment_gpu import gpu_needleman_wunsch


dna1 = "ATGCGTAC"
dna2 = "ATGCTAC"

print("=" * 55)
print("       GENEWEAVER CPU vs GPU ALIGNMENT")
print("=" * 55)

# CPU alignment
cpu_start = time.perf_counter()
cpu_result1, cpu_result2, cpu_score = needleman_wunsch(dna1, dna2)
cpu_time = (time.perf_counter() - cpu_start) * 1000

# GPU alignment
gpu_start = time.perf_counter()
gpu_result1, gpu_result2, gpu_score, gpu_name = gpu_needleman_wunsch(
    dna1, dna2
)
gpu_time = (time.perf_counter() - gpu_start) * 1000

print("\nCPU RESULTS")
print("Sequence 1:", cpu_result1)
print("Sequence 2:", cpu_result2)
print("Score:", cpu_score)
print(f"Execution time: {cpu_time:.6f} ms")

print("\nGPU RESULTS")
print("GPU:", gpu_name)
print("Sequence 1:", gpu_result1)
print("Sequence 2:", gpu_result2)
print("Score:", gpu_score)
print(f"Execution time: {gpu_time:.6f} ms")

print("\nVALIDATION")
if (
    cpu_result1 == gpu_result1
    and cpu_result2 == gpu_result2
    and cpu_score == gpu_score
):
    print("SUCCESS: CPU and GPU alignment results match!")
else:
    print("ERROR: CPU and GPU results do not match.")

print("\nPERFORMANCE")
if gpu_time > cpu_time:
    print("CPU is faster for this small test.")
    print("GPU setup and kernel launch overhead affect performance.")
else:
    print("GPU is faster for this test.")

print("=" * 55)
