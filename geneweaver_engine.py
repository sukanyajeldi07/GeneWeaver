
# GeneWeaver - Main Analysis Engine

from sequence_validator import validate_dna
from alignment import needleman_wunsch
from crispr_matching import find_guide_matches
from pam_detection import find_pam_sites


def run_analysis(dna, guide):

    print("\n================================")
    print("       GENEWEAVER ENGINE")
    print("================================")

    # Step 1: Validate DNA
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
            print("PAM:", site["pam"],
                  "Position:", site["position"])

        print("Total PAM sites:", len(pam_results))

    # Step 4: Alignment
    print("\n4. DNA ALIGNMENT")

    aligned1, aligned2, score = needleman_wunsch(
        dna, guide
    )

    print("DNA:  ", aligned1)
    print("Guide:", aligned2)
    print("Alignment score:", score)


# Sample analysis
dna = "ATGCGTACGGTAGCTAGG"
guide = "CGGT"

run_analysis(dna, guide)
