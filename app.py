
"""
GeneWeaver - Flask Backend
GPU-Assisted CRISPR Alignment Engine

Provides API endpoints for the GeneWeaver dashboard.
CPU and Intel OpenCL GPU alignment benchmarking.
"""

import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from sequence_validator import validate_dna
from pam_detection import find_pam_sites
from global_alignment_cpu import needleman_wunsch
from global_alignment_gpu import gpu_needleman_wunsch


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=None)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

BENCHMARK_REPETITIONS = 3


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def error_response(message, status=400):
    return jsonify({
        "success": False,
        "error": message
    }), status


def get_json_data():
    return request.get_json(silent=True) or {}


def get_gpu_device():
    """Detect available OpenCL GPU device."""

    try:
        import pyopencl as cl

        devices = []

        for platform in cl.get_platforms():
            for device in platform.get_devices():
                devices.append(device.name)

        if devices:
            return devices[0], True

        return "No OpenCL device found", False

    except Exception as exc:
        return f"OpenCL unavailable: {exc}", False


def get_alignment_score(result):
    """Extract alignment score from CPU/GPU tuple."""

    if isinstance(result, (tuple, list)) and len(result) >= 3:
        return result[2]

    return None


def alignments_match(cpu_result, gpu_result):
    """
    Compare alignment strings and score.

    GPU returns an additional device-name value.
    CPU returns three values.
    """

    if not isinstance(cpu_result, (tuple, list)):
        return False

    if not isinstance(gpu_result, (tuple, list)):
        return False

    if len(cpu_result) < 3 or len(gpu_result) < 3:
        return False

    return tuple(cpu_result[:3]) == tuple(gpu_result[:3])


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return send_from_directory(str(BASE_DIR), "index.html")


# --------------------------------------------------
# HEALTH API
# --------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({
        "success": True,
        "project": "GeneWeaver",
        "status": "Backend API is running",
        "version": "1.0",
        "engine": "Python"
    })


# --------------------------------------------------
# DNA ANALYSIS API
# --------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():

    data = get_json_data()

    sequence = str(
        data.get("sequence", "")
    ).upper().strip()

    if not sequence:
        return error_response("Please provide a DNA sequence.")

    validation = validate_dna(sequence)

    if validation is False:
        return error_response("Invalid DNA sequence.")

    length = len(sequence)

    counts = {
        base: sequence.count(base)
        for base in "ATGC"
    }

    gc_content = (
        (counts["G"] + counts["C"]) / length
    ) * 100

    return jsonify({
        "success": True,
        "length": length,
        "gc_content": round(gc_content, 2),
        "base_composition": counts,
        "pam_sites": find_pam_sites(sequence)
    })


# --------------------------------------------------
# CPU ALIGNMENT API
# --------------------------------------------------

@app.route("/api/align", methods=["POST"])
def align():

    data = get_json_data()

    seq1 = str(
        data.get("sequence1", "")
    ).upper().strip()

    seq2 = str(
        data.get("sequence2", "")
    ).upper().strip()

    if not seq1 or not seq2:
        return error_response("Provide both sequences.")

    start = time.perf_counter()

    result = needleman_wunsch(seq1, seq2)

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    return jsonify({
        "success": True,
        "algorithm": "Needleman-Wunsch CPU",
        "result": result,
        "execution_time_ms": round(elapsed, 4)
    })


# --------------------------------------------------
# FASTA FILE PARSER API
# --------------------------------------------------

@app.route("/api/fasta", methods=["POST"])
def fasta():

    if "file" not in request.files:
        return error_response("Please upload a FASTA file.")

    uploaded = request.files["file"]

    if not uploaded.filename:
        return error_response("No file selected.")

    content = uploaded.read().decode(
        "utf-8",
        errors="replace"
    )

    records = []
    header = None
    sequence_parts = []

    for line in content.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):

            if header is not None:
                records.append({
                    "header": header,
                    "sequence": "".join(sequence_parts)
                })

            header = line[1:]
            sequence_parts = []

        else:
            sequence_parts.append(line.upper())

    if header is not None:
        records.append({
            "header": header,
            "sequence": "".join(sequence_parts)
        })

    return jsonify({
        "success": True,
        "record_count": len(records),
        "records": records
    })


# --------------------------------------------------
# GPU STATUS API
# --------------------------------------------------

@app.route("/api/gpu-status")
def gpu_status():

    device, available = get_gpu_device()

    return jsonify({
        "success": True,
        "available": available,
        "backend": "Intel OpenCL" if available else "CPU fallback",
        "device": device
    })


# --------------------------------------------------
# GPU ALIGNMENT API
# --------------------------------------------------

@app.route("/api/gpu-align", methods=["POST"])
def gpu_align():

    data = get_json_data()

    seq1 = str(
        data.get("sequence1", "")
    ).upper().strip()

    seq2 = str(
        data.get("sequence2", "")
    ).upper().strip()

    if not seq1 or not seq2:
        return error_response("Provide both sequences.")

    start = time.perf_counter()

    result = gpu_needleman_wunsch(seq1, seq2)

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    return jsonify({
        "success": True,
        "algorithm": "Needleman-Wunsch GPU",
        "result": result,
        "execution_time_ms": round(elapsed, 4)
    })


# --------------------------------------------------
# CPU VS GPU BENCHMARK API
# --------------------------------------------------

@app.route("/api/benchmark", methods=["GET"])
def benchmark():

    test_sequences = [
        ("ATGC", "ATGC"),
        (
            "ATGCGTAC" * 4,
            "ATGCGTTC" * 4
        ),
        (
            "ATGCGTAC" * 8,
            "ATGCGTTC" * 8
        ),
        (
            "ATGCGTAC" * 16,
            "ATGCGTTC" * 16
        )
    ]

    results = []

    successful = 0
    failed = 0

    gpu_device, gpu_available = get_gpu_device()

    # Warm-up call to initialize the GPU before measuring.
    gpu_initialization_ms = None

    if gpu_available:

        try:
            initialization_start = time.perf_counter()

            gpu_needleman_wunsch("ATGC", "ATGC")

            gpu_initialization_ms = round(
                (time.perf_counter() - initialization_start) * 1000,
                4
            )

        except Exception as exc:
            gpu_available = False
            gpu_device = f"GPU initialization failed: {exc}"

    # Run each benchmark.
    for sequence1, sequence2 in test_sequences:

        cpu_times = []
        gpu_times = []

        cpu_result = None
        gpu_result = None

        try:

            for repetition in range(BENCHMARK_REPETITIONS):

                # CPU benchmark
                start = time.perf_counter()

                cpu_result = needleman_wunsch(
                    sequence1,
                    sequence2
                )

                cpu_elapsed = (
                    time.perf_counter() - start
                ) * 1000

                cpu_times.append(cpu_elapsed)

                # GPU benchmark
                if gpu_available:

                    start = time.perf_counter()

                    gpu_result = gpu_needleman_wunsch(
                        sequence1,
                        sequence2
                    )

                    gpu_elapsed = (
                        time.perf_counter() - start
                    ) * 1000

                    gpu_times.append(gpu_elapsed)

            average_cpu = sum(cpu_times) / len(cpu_times)

            average_gpu = (
                sum(gpu_times) / len(gpu_times)
                if gpu_times else None
            )

            matches = (
                alignments_match(cpu_result, gpu_result)
                if gpu_available
                else False
            )

            if matches:
                successful += 1
                status = "PASS"

            else:
                failed += 1
                status = "FAIL"

            results.append({

                "sequence_length": len(sequence1),

                "cpu_time_ms": round(
                    average_cpu, 4
                ),

                "gpu_time_ms": (
                    round(average_gpu, 4)
                    if average_gpu is not None
                    else None
                ),

                "results_match": matches,

                "validation": status,

                "alignment_score": get_alignment_score(
                    cpu_result
                ),

                "repetitions": BENCHMARK_REPETITIONS,

                "speedup": (
                    round(average_cpu / average_gpu, 4)
                    if average_gpu is not None and average_gpu > 0
                    else None
                )

            })

        except Exception as exc:

            failed += 1

            results.append({

                "sequence_length": len(sequence1),

                "validation": "ERROR",

                "results_match": False,

                "error": str(exc)

            })

    return jsonify({

        "success": (
            successful == len(test_sequences)
        ),

        "project": "GeneWeaver",

        "algorithm": "Needleman-Wunsch",

        "gpu_backend": (
            "Intel OpenCL"
            if gpu_available
            else "GPU unavailable"
        ),

        "gpu_device": gpu_device,

        "gpu_initialization_ms": gpu_initialization_ms,

        "total_tests": len(test_sequences),

        "successful_tests": successful,

        "failed_tests": failed,

        "repetitions": BENCHMARK_REPETITIONS,

        "results": results

    })


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(404)
def not_found(error):

    return error_response(
        "Endpoint not found.",
        404
    )


@app.errorhandler(413)
def too_large(error):

    return error_response(
        "Uploaded file is too large.",
        413
    )


@app.errorhandler(500)
def internal_error(error):

    return error_response(
        "Internal server error.",
        500
    )


# --------------------------------------------------
# APPLICATION STARTUP
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 55)
    print("       GENEWEAVER FLASK BACKEND")
    print("=" * 55)

    print("Website:   http://127.0.0.1:5000")
    print("Health:    http://127.0.0.1:5000/api/health")
    print("GPU:       http://127.0.0.1:5000/api/gpu-status")
    print("Benchmark: http://127.0.0.1:5000/api/benchmark")

    print("=" * 55)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
