
import time
import numpy as np
import pyopencl as cl


# Reuse OpenCL resources instead of creating them for every alignment
_context = None
_queue = None
_kernel = None
_device = None


def initialize_gpu():

    global _context, _queue, _kernel, _device

    if _context is not None:
        return

    devices = [
        device
        for platform in cl.get_platforms()
        for device in platform.get_devices(
            device_type=cl.device_type.GPU
        )
    ]

    if not devices:
        raise RuntimeError("No OpenCL GPU found")

    _device = devices[0]
    _context = cl.Context([_device])
    _queue = cl.CommandQueue(_context)

    kernel_code = """
    __kernel void fill_diagonal(
        __global const uchar *seq1,
        __global const uchar *seq2,
        __global int *matrix,
        const int rows,
        const int cols,
        const int diagonal,
        const int match_score,
        const int mismatch_score,
        const int gap_score
    ) {
        int i = get_global_id(0) + 1;
        int j = diagonal - i;

        if (i < rows && j > 0 && j < cols) {

            int diagonal_score = matrix[(i-1)*cols + (j-1)];

            if (seq1[i-1] == seq2[j-1])
                diagonal_score += match_score;
            else
                diagonal_score += mismatch_score;

            int up = matrix[(i-1)*cols + j] + gap_score;
            int left = matrix[i*cols + (j-1)] + gap_score;

            int best = max(diagonal_score, max(up, left));

            matrix[i*cols + j] = best;
        }
    }
    """

    program = cl.Program(_context, kernel_code).build()
    _kernel = cl.Kernel(program, "fill_diagonal")


def gpu_needleman_wunsch(
    sequence1, sequence2, match=2, mismatch=-1, gap=-2
):

    initialize_gpu()

    rows = len(sequence1) + 1
    cols = len(sequence2) + 1

    seq1 = np.frombuffer(sequence1.encode("ascii"), dtype=np.uint8)
    seq2 = np.frombuffer(sequence2.encode("ascii"), dtype=np.uint8)

    matrix = np.zeros((rows, cols), dtype=np.int32)

    matrix[:, 0] = np.arange(rows, dtype=np.int32) * gap
    matrix[0, :] = np.arange(cols, dtype=np.int32) * gap

    mf = cl.mem_flags

    seq1_buffer = cl.Buffer(
        _context, mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=seq1
    )

    seq2_buffer = cl.Buffer(
        _context, mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=seq2
    )

    matrix_buffer = cl.Buffer(
        _context, mf.READ_WRITE | mf.COPY_HOST_PTR,
        hostbuf=matrix
    )

    for diagonal in range(2, rows + cols - 1):

        work_items = min(rows - 1, diagonal - 1)

        _kernel(
            _queue,
            (work_items,),
            None,
            seq1_buffer,
            seq2_buffer,
            matrix_buffer,
            np.int32(rows),
            np.int32(cols),
            np.int32(diagonal),
            np.int32(match),
            np.int32(mismatch),
            np.int32(gap)
        )

    cl.enqueue_copy(_queue, matrix, matrix_buffer).wait()

    # CPU traceback
    aligned1 = []
    aligned2 = []

    i = len(sequence1)
    j = len(sequence2)

    while i > 0 or j > 0:

        if i > 0 and j > 0:
            score = (
                match if sequence1[i - 1] == sequence2[j - 1]
                else mismatch
            )

            if matrix[i, j] == matrix[i - 1, j - 1] + score:
                aligned1.append(sequence1[i - 1])
                aligned2.append(sequence2[j - 1])
                i -= 1
                j -= 1
                continue

        if i > 0 and matrix[i, j] == matrix[i - 1, j] + gap:
            aligned1.append(sequence1[i - 1])
            aligned2.append("-")
            i -= 1
        else:
            aligned1.append("-")
            aligned2.append(sequence2[j - 1])
            j -= 1

    aligned1.reverse()
    aligned2.reverse()

    return (
        "".join(aligned1),
        "".join(aligned2),
        int(matrix[-1, -1]),
        _device.name
    )


if __name__ == "__main__":

    dna1 = "ATGCGTAC"
    dna2 = "ATGCTAC"

    start = time.perf_counter()

    result1, result2, score, gpu_name = gpu_needleman_wunsch(
        dna1, dna2
    )

    elapsed = (time.perf_counter() - start) * 1000

    print("=" * 50)
    print(" GENEWEAVER GPU GLOBAL ALIGNMENT")
    print("=" * 50)
    print("GPU:", gpu_name)
    print("DNA Sequence 1:", dna1)
    print("DNA Sequence 2:", dna2)
    print("\nALIGNED SEQUENCES")
    print("Sequence 1:", result1)
    print("Sequence 2:", result2)
    print("Alignment score:", score)
    print(f"GPU execution time: {elapsed:.6f} ms")
