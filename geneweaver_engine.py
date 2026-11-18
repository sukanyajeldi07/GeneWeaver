
# GeneWeaver - Main Analysis Engine

from sequence_validator import validate_dna
from alignment import needleman_wunsch
from crispr_matching import find_guide_matches
from pam_detection import find_pam_sites

from genome_parser import (
    process_fasta_in_chunks,
    get_fasta_summary
)


# --------------------------------------------------
# ORIGINAL DNA ANALYSIS
# --------------------------------------------------

def run_analysis(dna, guide):

    dna = dna.upper()
    guide = guide.upper()

    print("\n================================")
    print("       GENEWEAVER ENGINE")
    print("================================")

    # Step 1: DNA validation

    valid, message = validate_dna(dna)

    print("\n1. DNA VALIDATION")
    print("Status:", message)

    if not valid:
        print("Analysis stopped.")
        return

    # Step 2: Guide matching

    print("\n2. CRISPR GUIDE MATCHING")

    matches = find_guide_matches(dna, guide)

    if isinstance(matches, list):
        print("Matching positions:", matches)
        print("Total matches:", len(matches))
    else:
        print("Error:", matches)
        return

    # Step 3: PAM detection

    print("\n3. PAM DETECTION")

    pam_results = find_pam_sites(dna)

    if isinstance(pam_results, list):

        for site in pam_results:
            print(
                "PAM:", site["pam"],
                "Position:", site["position"]
            )

        print("Total PAM sites:", len(pam_results))

    # Step 4: Alignment

    print("\n4. DNA ALIGNMENT")

    aligned1, aligned2, score = needleman_wunsch(
        dna, guide
    )

    print("DNA:  ", aligned1)
    print("Guide:", aligned2)
    print("Alignment score:", score)


# --------------------------------------------------
# NEW: FASTA GENOME CHUNK ANALYSIS
# --------------------------------------------------

def run_fasta_chunk_analysis(
    file_path,
    guide,
    chunk_size=100000,
    overlap=0
):

    guide = guide.upper()

    print("\n========================================")
    print(" GENEWEAVER FASTA CHUNK ANALYSIS")
    print("========================================")

    print("FASTA file:", file_path)
    print("Chunk size:", chunk_size)
    print("Overlap:", overlap)

    # Get file statistics

    try:

        summary = get_fasta_summary(file_path)

    except Exception as error:

        print("\nFASTA processing error:", error)
        return

    print("\nFASTA SUMMARY")
    print("Records:", summary["records"])
    print("Total base pairs:", summary["total_bases"])
    print("GC content:", summary["gc_percentage"], "%")

    print("\nPROCESSING GENOMIC CHUNKS")

    total_chunks = 0
    valid_chunks = 0
    skipped_chunks = 0
    total_pam_sites = 0

    try:

        for chunk in process_fasta_in_chunks(
            file_path,
            chunk_size=chunk_size,
            overlap=overlap
        ):

            total_chunks += 1

            sequence = chunk["sequence"]

            print("\n----------------------------------------")
            print(
                f"Record: {chunk['record_id']} | "
                f"Chunk: {chunk['chunk_number']}"
            )

            print(
                f"Position: {chunk['start']} - {chunk['end']}"
            )

            print("Chunk length:", chunk["length"])

            # Validate chunk for the existing DNA engine

            valid, message = validate_dna(sequence)

            print("DNA validation:", message)

            if not valid:

                skipped_chunks += 1

                print(
                    "Chunk skipped by the existing "
                    "ACGT-only analysis engine."
                )

                continue

            valid_chunks += 1

            # Guide matching

            print("\nCRISPR GUIDE MATCHING")

            matches = find_guide_matches(
                sequence,
                guide
            )

            if isinstance(matches, list):

                print("Matching positions:", matches)
                print("Total matches:", len(matches))

            else:

                print("Guide matching result:", matches)

            # PAM detection

            print("\nPAM DETECTION")

            pam_results = find_pam_sites(sequence)

            if isinstance(pam_results, list):

                for site in pam_results:

                    print(
                        "PAM:", site["pam"],
                        "Position:", site["position"]
                    )

                total_pam_sites += len(pam_results)

                print(
                    "PAM sites in this chunk:",
                    len(pam_results)
                )

            # Alignment

            print("\nGLOBAL ALIGNMENT")

            aligned1, aligned2, score = needleman_wunsch(
                sequence,
                guide
            )

            print("Sequence:", aligned1)
            print("Guide:   ", aligned2)
            print("Alignment score:", score)

    except Exception as error:

        print("\nChunk processing error:", error)
        return

    # Final summary

    print("\n========================================")
    print(" FASTA CHUNK PROCESSING SUMMARY")
    print("========================================")

    print("Total chunks:", total_chunks)
    print("Valid chunks:", valid_chunks)
    print("Skipped chunks:", skipped_chunks)
    print("Total PAM sites:", total_pam_sites)

    print("\nFASTA chunk analysis completed.")


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    # Original sample analysis

    dna = "ATGCGTACGGTAGCTAGG"
    guide = "CGGT"

    run_analysis(dna, guide)

    # To process a FASTA file, call:
    #
    # run_fasta_chunk_analysis(
    #     "sample.fasta",
    #     "CGGT",
    #     chunk_size=100000
    # )
