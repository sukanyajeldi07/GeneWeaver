
import time
import numpy as np
import pyopencl as cl

print("=" * 50)
print(" GENEWEAVER CPU vs GPU SEQUENCE SCORING")
print("=" * 50)

guide = "ATGCGTAC"

candidates = [
    "ATGCGTAC",
    "ATGCGTTC",
    "ATGAGTAC",
    "TTGCGTAC",
    "ATGCGTAA"
]

# CPU sequence scoring
def cpu_score(guide, candidates):
    results = []

    for sequence in candidates:
        matches = sum(
            a == b for a, b in zip(guide, sequence)
        )
        results.append(matches)

    return results


cpu_start = time.perf_counter()
cpu_results = cpu_score(guide, candidates)
cpu_time = time.perf_counter() - cpu_start

# Prepare sequences for GPU
target_array = np.array(
    [list(s.encode("ascii")) for s in candidates],
    dtype=np.uint8
)

guide_array = np.frombuffer(
    guide.encode("ascii"),
    dtype=np.uint8
)

gpu_results = np.empty(len(candidates), dtype=np.int32)

# Find Intel GPU
devices = [
    device
    for platform in cl.get_platforms()
    for device in platform.get_devices(
        device_type=cl.device_type.GPU
    )
]

if not devices:
    raise RuntimeError("No OpenCL GPU found")

device = devices[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

kernel_code = """
__kernel void score_sequences(
    __global const uchar *targets,
    __global const uchar *guide,
    __global int *scores,
    const int sequence_length
) {
    int id = get_global_id(0);
    int matches = 0;

    for (int j = 0; j < sequence_length; j++) {
        if (targets[id * sequence_length + j] == guide[j]) {
            matches++;
        }
    }

    scores[id] = matches;
}
"""

# Measure GPU setup, execution and result retrieval
gpu_start = time.perf_counter()

mf = cl.mem_flags

target_buffer = cl.Buffer(
    context,
    mf.READ_ONLY | mf.COPY_HOST_PTR,
    hostbuf=target_array
)

guide_buffer = cl.Buffer(
    context,
    mf.READ_ONLY | mf.COPY_HOST_PTR,
    hostbuf=guide_array
)

score_buffer = cl.Buffer(
    context,
    mf.WRITE_ONLY,
    gpu_results.nbytes
)

program = cl.Program(context, kernel_code).build()

program.score_sequences(
    queue,
    (len(candidates),),
    None,
    target_buffer,
    guide_buffer,
    score_buffer,
    np.int32(len(guide))
)

cl.enqueue_copy(queue, gpu_results, score_buffer).wait()

gpu_time = time.perf_counter() - gpu_start

# Display results
print("\nGPU Device:", device.name)

print("\nSEQUENCE RESULTS")

for i, sequence in enumerate(candidates):
    print(
        sequence,
        "| CPU matches:", cpu_results[i],
        "| GPU matches:", int(gpu_results[i])
    )

# Verify both implementations agree
if cpu_results == gpu_results.tolist():
    print("\nVerification: CPU and GPU results match!")
else:
    print("\nVerification FAILED: Results differ!")

print("\nPERFORMANCE")
print(f"CPU time: {cpu_time * 1000:.6f} ms")
print(f"GPU time (including setup): {gpu_time * 1000:.6f} ms")

print("\nBenchmark completed!")
