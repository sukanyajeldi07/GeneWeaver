# GENEWEAVER
## GPU-Accelerated CRISPR Alignment Engine

**Project Report**

**Project:** GeneWeaver – GPU-Accelerated CRISPR Alignment Engine  
**Author:** Sukanya Jeldi  
**Organization:** Infotact Solutions  
**Domain:** Python Development  
**Duration:** 1 Month  

---

## 1. Abstract

GeneWeaver is a Python-based DNA sequence analysis project designed to combine CRISPR-related sequence analysis with GPU-assisted global sequence alignment.

The system provides DNA sequence validation, CRISPR target matching, SpCas9 PAM detection, and Needleman-Wunsch global alignment using CPU and GPU implementations.

OpenCL is used to perform GPU-based alignment scoring. The CPU implementation serves as a baseline for comparing alignment results and execution times.

## 2. Introduction

DNA sequence analysis is an important part of bioinformatics and CRISPR-related research. Sequence validation, target identification, PAM recognition, and sequence alignment are useful operations in this domain.

GeneWeaver demonstrates how Python and GPU computing can be combined to implement these operations in a modular software project.

## 3. Project Objectives

1. Validate DNA sequences containing A, T, G, and C.
2. Identify SpCas9 NGG PAM sites.
3. Detect exact CRISPR target matches.
4. Implement Needleman-Wunsch global alignment.
5. Implement GPU-based alignment scoring using OpenCL.
6. Compare CPU and GPU alignment results.
7. Evaluate execution time using benchmark tests.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical computing |
| PyOpenCL | GPU computing interface |
| OpenCL | GPU execution framework |
| Biopython | Bioinformatics library |
| Pytest | Automated testing |

## 5. System Architecture

The project is divided into the following functional modules:

- **Input validation:** Checks DNA sequence characters and empty input.
- **PAM detection:** Identifies NGG motifs.
- **CRISPR matching:** Searches for exact