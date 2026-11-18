
# GeneWeaver - SpCas9 PAM Detection
# Detects NGG motifs in a DNA sequence


def find_pam_sites(dna_sequence):
    """Detect SpCas9 NGG PAM sites in a DNA sequence."""

    dna_sequence = dna_sequence.upper()

    valid_bases = {"A", "T", "G", "C"}

    if not dna_sequence:
        return "DNA sequence cannot be empty"

    if not set(dna_sequence).issubset(valid_bases):
        return "Invalid DNA sequence"

    pam_sites = []

    for i in range(len(dna_sequence) - 2):

        pam = dna_sequence[i:i + 3]

        if pam[1:] == "GG":
            pam_sites.append({
                "position": i,
                "pam": pam
            })

    return pam_sites


# Sample test runs only when this file is executed directly.
if __name__ == "__main__":

    dna = "ATGCGTACGGTAGCTAGG"

    results = find_pam_sites(dna)

    print("\nGeneWeaver - PAM Detection")
    print("-------------------------")
    print("DNA Sequence:", dna)

    if isinstance(results, list):

        for site in results:
            print(
                "PAM:", site["pam"],
                "| Position:", site["position"]
            )

        print("Total PAM sites:", len(results))

    else:
        print("Error:", results)
