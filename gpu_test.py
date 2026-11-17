
import pyopencl as cl
import numpy as np

print("=" * 45)
print("       GENEWEAVER GPU TEST")
print("=" * 45)

# Detect OpenCL GPU
platforms = cl.get_platforms()
gpu_devices = []

for platform in platforms:
    gpu_devices.extend(
        platform.get_devices(device_type=cl.device_type.GPU)
    )

if not gpu_devices:
    print("No GPU device found.")
else:
    device = gpu_devices[0]

    print("GPU:", device.name)
    print("Platform:", device.platform.name)

    # Create GPU context and command queue
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Prepare sample arrays
    a = np.array([10, 20, 30, 40], dtype=np.float32)
    b = np.array([1, 2, 3, 4], dtype=np.float32)
    result = np.empty_like(a)

    # GPU memory buffers
    mf = cl.mem_flags

    a_buffer = cl.Buffer(
        context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a
    )
    b_buffer = cl.Buffer(
        context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b
    )
    result_buffer = cl.Buffer(
        context, mf.WRITE_ONLY, result.nbytes
    )

    # OpenCL GPU program
    program = cl.Program(context, """
        __kernel void add_arrays(
            __global const float *a,
            __global const float *b,
            __global float *result
        ) {
            int i = get_global_id(0);
            result[i] = a[i] + b[i];
        }
    """).build()

    # Execute calculation on GPU
    program.add_arrays(
        queue, (4,), None,
        a_buffer, b_buffer, result_buffer
    )

    # Copy result back to CPU
    cl.enqueue_copy(queue, result, result_buffer).wait()

    print("Array A:", a)
    print("Array B:", b)
    print("GPU Result:", result)
    print("GPU calculation successful!")
