
import pyopencl as cl
import numpy as np

print("=" * 45)
print("  GENEWEAVER GPU SEQUENCE SCORING")
print("=" * 45)

# CRISPR guide sequence
guide = "ATGCGTAC"

# Candidate DNA sequences
candidates = [
    "ATGCGTAC",
    "ATGCGTTC",
    "ATGAGTAC",
    "TTGCGTAC",
    "ATGCGTAA"
]

# Validate input lengths and DNA characters
valid_bases = set("ATGC")

if not guide or any(base not in valid_bases for base in guide):
    raise ValueError("Invalid guide sequence")

for sequence in candidates:
    if len(sequence) != len(guide):
        raise ValueError("All candidates must have the same length")
    if any(base not in valid_bases for base in sequence):
        raise ValueError("Invalid candidate DNA sequence")

# Convert DNA sequences into numerical arrays
target_array = np.array(
    [list(sequence.encode("ascii")) for sequence in candidates],
    dtype=np.uint8
)

guide_array = np.frombuffer(
    guide.encode("ascii"),
    dtype=np.uint8
)

scores = np.empty(len(candidates), dtype=np.int32)

# Find Intel GPU
platforms = cl.get_platforms()
devices = [
    device
    for platform in platforms
    for device in platform.get_devices(
        device_type=cl.device_type.GPU
    )
]

if not devices:
    raise RuntimeError("No OpenCL GPU found")

device = devices[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

print("GPU:", device.name)
print("Guide:", guide)

# Allocate GPU memory
mf = cl.mem_flags

target_buffer = cl.Buffer(
    context, mf.READ_ONLY | mf.COPY_HOST_PTR,
    hostbuf=target_array
)

guide_buffer = cl.Buffer(
    context, mf.READ_ONLY | mf.COPY_HOST_PTR,
    hostbuf=guide_array
)

score_buffer = cl.Buffer(
    context, mf.WRITE_ONLY, scores.nbytes
)

# GPU scoring kernel
kernel_code = """
__kernel void score_sequences(
    __global const uchar *targets,
    __global const uchar *guide,
    __global int *scores,
    const int sequence_length
) {
    int id = get_global_id(0);
    int mismatches = 0;

    for (int j = 0; j < sequence_length; j++) {
        if (targets[id * sequence_length + j] != guide[j]) {
            mismatches++;
        }
    }

    scores[id] = sequence_length - mismatches;
}
"""

program = cl.Program(context, kernel_code).build()

# Execute scoring on GPU
program.score_sequences(
    queue,
    (len(candidates),),
    None,
    target_buffer,
    guide_buffer,
    score_buffer,
    np.int32(len(guide))
)

# Retrieve results
cl.enqueue_copy(queue, scores, score_buffer).wait()

print("\nSEQUENCE SCORING RESULTS")

for sequence, score in zip(candidates, scores):
    mismatches = len(guide) - int(score)
    print(
        "Candidate:", sequence,
        "| Matches:", int(score),
        "| Mismatches:", mismatches
    )

print("\nGPU sequence scoring completed!")
