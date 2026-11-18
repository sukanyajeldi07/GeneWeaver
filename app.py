
"""
GeneWeaver - Flask Backend
GPU-Assisted CRISPR Alignment Engine

Provides API endpoints for the GeneWeaver browser dashboard.
Existing Python modules are reused without modifying them.
"""

import math
import tempfile
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from sequence_validator import validate_dna
from alignment import needleman_wunsch
from crispr_matching import find_guide_matches
from pam_detection import find_pam_sites
from genome_parser import parse_fasta_file, get_fasta_summary


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

CORS(app)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def error_response(message, status_code=400):
    return jsonify({
        "success": False,
        "error": message
    }), status_code


def get_json_data():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return None

    return data


# --------------------------------------------------
# WEBSITE ROUTE
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return send_from_directory(BASE_DIR, "index.html")


# --------------------------------------------------
# 1. HEALTH CHECK
# --------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "project": "GeneWeaver",
        "status": "Backend API is running",
        "version": "1.0",
        "engine": "Python",
        "gpu_backend": "Intel OpenCL (if available)"
    })


# --------------------------------------------------
# 2. DNA VALIDATION AND CRISPR ANALYSIS
# --------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze_dna():

    data = get_json_data()

    if data is None:
        return error_response("Send valid JSON data.")

    dna = data.get("sequence", "")
    guide = data.get("guide", "")

    if not isinstance(dna, str):
        return error_response("DNA sequence must be text.")

    if not isinstance(guide, str):
        return error_response("Guide sequence must be text.")

    dna = dna.strip().upper()
    guide = guide.strip().upper()

    if len(dna) > 100000:
        return error_response("DNA sequence exceeds 100,000 bases.")

    start = time.perf_counter()

    valid, message = validate_dna(dna)

    if not valid:
        return jsonify({
            "success": False,
            "valid": False,
            "message": message
        }), 400

    # DNA composition

    composition = {
        "A": dna.count("A"),
        "T": dna.count("T"),
        "G": dna.count("G"),
        "C": dna.count("C")
    }

    # PAM detection

    pam_results = find_pam_sites(dna)

    # Optional guide matching

    guide_matches = []

    if guide:

        guide_valid, guide_message = validate_dna(guide)

        if not guide_valid:
            return error_response(
                "Invalid guide sequence: " + guide_message
            )

        if len(guide) > 10000:
            return error_response("Guide sequence is too long.")

        guide_matches = find_guide_matches(dna, guide)

    elapsed = (time.perf_counter() - start) * 1000

    return jsonify({
        "success": True,
        "valid": True,
        "message": message,
        "sequence_length": len(dna),
        "composition": composition,
        "pam_sites": pam_results,
        "pam_count": len(pam_results),
        "guide_matches": guide_matches,
        "guide_match_count": len(guide_matches),
        "processing_time_ms": round(elapsed, 4)
    })


# --------------------------------------------------
# 3. NEEDLEMAN-WUNSCH CPU ALIGNMENT
# --------------------------------------------------

@app.route("/api/align", methods=["POST"])
def align_sequences():

    data = get_json_data()

    if data is None:
        return error_response("Send valid JSON data.")

    sequence1 = data.get("sequence1", "")
    sequence2 = data.get("sequence2", "")

    if not isinstance(sequence1, str) or not isinstance(sequence2, str):
        return error_response("Both sequences must be text.")

    sequence1 = sequence1.strip().upper()
    sequence2 = sequence2.strip().upper()

    if not sequence1 or not sequence2:
        return error_response("Both DNA sequences are required.")

    valid1, message1 = validate_dna(sequence1)
    valid2, message2 = validate_dna(sequence2)

    if not valid1:
        return error_response("Sequence 1: " + message1)

    if not valid2:
        return error_response("Sequence 2: " + message2)

    # Protect the server from excessive matrix allocation.
    if len(sequence1) > 1000 or len(sequence2) > 1000:
        return error_response(
            "CPU alignment is limited to 1,000 bases per sequence."
        )

    start = time.perf_counter()

    aligned1, aligned2, score = needleman_wunsch(
        sequence1,
        sequence2
    )

    elapsed = (time.perf_counter() - start) * 1000

    return jsonify({
        "success": True,
        "algorithm": "Needleman-Wunsch",
        "backend": "CPU",
        "aligned_sequence1": aligned1,
        "aligned_sequence2": aligned2,
        "alignment_score": score,
        "processing_time_ms": round(elapsed, 4)
    })


# --------------------------------------------------
# 4. FASTA FILE UPLOAD AND PARSING
# --------------------------------------------------

@app.route("/api/fasta", methods=["POST"])
def analyze_fasta():

    if "file" not in request.files:
        return error_response("Please upload a FASTA file.")

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return error_response("No file selected.")

    extension = Path(uploaded_file.filename).suffix.lower()

    if extension not in {".fasta", ".fa", ".fna"}:
        return error_response(
            "Supported formats: .fasta, .fa and .fna"
        )

    try:
        chunk_size = int(request.form.get("chunk_size", 100000))
        overlap = int(request.form.get("overlap", 0))
    except (TypeError, ValueError):
        return error_response("Chunk size and overlap must be integers.")

    if not 1 <= chunk_size <= 1000000:
        return error_response("Chunk size must be between 1 and 1,000,000.")

    if not 0 <= overlap < chunk_size:
        return error_response(
            "Overlap must be at least 0 and smaller than chunk size."
        )

    try:

        with tempfile.TemporaryDirectory() as temp_dir:

            file_path = Path(temp_dir) / ("upload" + extension)
            uploaded_file.save(file_path)

            # Parse records

            records = list(parse_fasta_file(file_path))

            # Calculate summary

            summary = get_fasta_summary(file_path)

            record_results = []
            total_chunks = 0

            step = chunk_size - overlap

            for record in records:

                length = record["length"]

                chunk_count = (
                    math.ceil(length / step) if length else 0
                )

                total_chunks += chunk_count

                sequence = record["sequence"]

                record_results.append({
                    "id": record["id"],
                    "description": record["description"],
                    "length": length,
                    "gc_percentage": round(
                        (
                            (
                                sequence.count("G")
                                + sequence.count("C")
                            ) / length * 100
                        ),
                        2
                    ) if length else 0,
                    "chunk_count": chunk_count
                })

        return jsonify({
            "success": True,
            "message": "FASTA file parsed successfully.",
            "summary": {
                "records": summary["records"],
                "total_bases": summary["total_bases"],
                "gc_percentage": summary["gc_percentage"],
                "chunk_size": chunk_size,
                "overlap": overlap,
                "total_chunks": total_chunks
            },
            "records_detail": record_results
        })

    except Exception as error:
        return error_response(
            "FASTA processing failed: " + str(error)
        )


# --------------------------------------------------
# 5. GPU DEVICE STATUS
# --------------------------------------------------

@app.route("/api/gpu-status", methods=["GET"])
def gpu_status():

    try:

        import pyopencl as cl

        devices = []

        for platform in cl.get_platforms():

            for device in platform.get_devices(
                device_type=cl.device_type.GPU
            ):

                devices.append({
                    "name": device.name.strip(),
                    "vendor": device.vendor.strip(),
                    "version": device.version.strip(),
                    "platform": platform.name.strip()
                })

        if not devices:
            return jsonify({
                "success": True,
                "available": False,
                "backend": "OpenCL",
                "message": "No OpenCL GPU device detected."
            })

        return jsonify({
            "success": True,
            "available": True,
            "backend": "OpenCL",
            "devices": devices
        })

    except Exception as error:

        return jsonify({
            "success": True,
            "available": False,
            "backend": "OpenCL",
            "message": str(error)
        })


# --------------------------------------------------
# 6. GPU GLOBAL ALIGNMENT
# --------------------------------------------------

@app.route("/api/gpu-align", methods=["POST"])
def gpu_align():

    data = get_json_data()

    if data is None:
        return error_response("Send valid JSON data.")

    sequence1 = data.get("sequence1", "")
    sequence2 = data.get("sequence2", "")

    if not isinstance(sequence1, str) or not isinstance(sequence2, str):
        return error_response("Both sequences must be text.")

    sequence1 = sequence1.strip().upper()
    sequence2 = sequence2.strip().upper()

    if not sequence1 or not sequence2:
        return error_response("Both DNA sequences are required.")

    valid1, message1 = validate_dna(sequence1)
    valid2, message2 = validate_dna(sequence2)

    if not valid1:
        return error_response("Sequence 1: " + message1)

    if not valid2:
        return error_response("Sequence 2: " + message2)

    if len(sequence1) > 1000 or len(sequence2) > 1000:
        return error_response(
            "GPU alignment is limited to 1,000 bases per sequence."
        )

    try:

        from global_alignment_gpu import gpu_needleman_wunsch

        start = time.perf_counter()

        aligned1, aligned2, score, device_name = (
            gpu_needleman_wunsch(sequence1, sequence2)
        )

        elapsed = (time.perf_counter() - start) * 1000

        return jsonify({
            "success": True,
            "algorithm": "Needleman-Wunsch",
            "backend": "Intel OpenCL",
            "device": device_name,
            "aligned_sequence1": aligned1,
            "aligned_sequence2": aligned2,
            "alignment_score": score,
            "processing_time_ms": round(elapsed, 4),
            "traceback": "CPU"
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "backend": "OpenCL",
            "error": str(error)
        }), 503


# --------------------------------------------------
# 7. ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(404)
def not_found(error):

    return error_response("API route not found.", 404)


@app.errorhandler(413)
def file_too_large(error):

    return error_response(
        "Uploaded file exceeds the 5 MB limit.",
        413
    )


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 55)
    print("       GENEWEAVER FLASK BACKEND")
    print("=" * 55)
    print("Website: http://127.0.0.1:5000")
    print("Health:  http://127.0.0.1:5000/api/health")
    print("GPU:     http://127.0.0.1:5000/api/gpu-status")
    print("=" * 55)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
