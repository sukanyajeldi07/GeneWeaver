# GeneWeaver: GPU-Accelerated CRISPR Alignment Engine

## 1. Project Overview

GeneWeaver is a Python-based DNA sequence analysis project that combines CRISPR-related sequence utilities with GPU-assisted sequence alignment.

The project implements DNA validation, PAM site detection, CRISPR target matching, and global sequence alignment using both CPU and GPU computation.

It uses OpenCL to perform GPU-based alignment scoring on supported hardware.

## 2. Project Objectives

- Validate DNA sequences containing A, T, G, and C.
- Identify SpCas9 NGG PAM sites.
- Find exact CRISPR target sequence matches.
- Perform Needleman-Wunsch global sequence alignment.
- Compare CPU and GPU alignment results.
- Benchmark alignment performance across different sequence lengths.

## 3. Technologies Used

- Python
- NumPy
- PyOpenCL
- OpenCL
- Biopython
- Pytest

## 4. Project Structure

```text
GeneWeaver/
│
├── app.py
├── sequence_validator.py
├── alignment.py
├── crispr_matching.py
├── pam_detection.py
├── geneweaver_engine.py
│
├── gpu_test.py
├── gpu_sequence_scoring.py
├── cpu_gpu_comparison.py
│
├── global_alignment_cpu.py
├── global_alignment_gpu.py
├── alignment_comparison.py
├── alignment_benchmark.py
│
├── benchmark_results.txt
├── requirements.txt
├── .gitignore
│
└── tests/
    ├── test_alignment.py
    ├── test_pam_detection.py
    └── test_validator.py
```

## 5. Main Features

### DNA Sequence Validation
Checks whether a DNA sequence contains only valid DNA bases: A, T, G, and C.

### PAM Detection
Detects SpCas9 NGG PAM motifs in DNA sequences and reports their positions.

### CRISPR Target Matching
Searches for exact target sequence matches in a DNA sequence.

### CPU Global Alignment
Implements the Needleman-Wunsch algorithm using dynamic programming.

### GPU Global Alignment
Uses OpenCL to calculate the alignment scoring matrix on a supported GPU. The traceback step is performed on the CPU.

### Performance Benchmarking
Compares CPU and GPU alignment execution times for different sequence lengths.

## 6. Hardware Configuration

- Processor: 12th Gen Intel Core i5-1235U
- GPU: Intel UHD Graphics
- GPU Framework: OpenCL
- Programming Language: Python

## 7. Installation

Clone the repository:

```bash
git clone https://github.com/sukanyajeldi07/GeneWeaver.git
```

Navigate into the project directory:

```bash
cd GeneWeaver
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 8. Running the Project

Run the application:

```bash
python app.py
```

Run DNA validation:

```bash
python sequence_validator.py
```

Run PAM detection:

```bash
python pam_detection.py
```

Run GPU sequence scoring:

```bash
python gpu_sequence_scoring.py
```

Run CPU global alignment:

```bash
python global_alignment_cpu.py
```

Run GPU global alignment:

```bash
python global_alignment_gpu.py
```

Run CPU/GPU comparison:

```bash
python alignment_comparison.py
```

Run performance benchmarking:

```bash
python alignment_benchmark.py
```

## 9. Testing

GeneWeaver uses Pytest for automated testing.

Execute the complete test suite:

```bash
python -m pytest -v
```

Current test results:

**21 tests passed successfully.**

- DNA validation: 7 tests
- PAM detection: 7 tests
- CPU/GPU alignment: 7 tests

## 10. Benchmarking Notes

The project compares CPU and GPU execution times across multiple sequence lengths.

GPU execution includes OpenCL overhead, such as kernel launch and device interaction. For small sequences, CPU execution may be faster.

The GPU implementation calculates the scoring matrix on the GPU, while traceback is performed on the CPU. Therefore, the benchmark measures the end-to-end alignment operation, not GPU kernel execution alone.

## 11. Current Limitations

- CRISPR matching currently supports exact forward-strand matches.
- PAM detection currently focuses on the SpCas9 NGG motif.
- GPU acceleration currently applies to alignment scoring.
- CPU traceback is used to reconstruct the final alignment.
- GPU performance depends on sequence length and hardware.

## 12. Future Enhancements

- Reverse-complement sequence matching.
- Mismatch-tolerant CRISPR target identification.
- Additional PAM motif support.
- Improved GPU optimization for larger sequences.
- Interactive DNA sequence analysis dashboard.
- Detailed performance visualization.

## 13. Author

**Sukanya Jeldi**

Project: GeneWeaver – GPU-Accelerated CRISPR Alignment Engine

GitHub: https://github.com/sukanyajeldi07/GeneWeaver
