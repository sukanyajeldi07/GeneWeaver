
# GeneWeaver - DNA Sequence Alignment
# Needleman-Wunsch Global Alignment Algorithm

def needleman_wunsch(seq1, seq2):

    match = 2
    mismatch = -1
    gap = -2

    rows = len(seq1) + 1
    cols = len(seq2) + 1

    matrix = [[0] * cols for _ in range(rows)]

    # Initialize first column and row
    for i in range(rows):
        matrix[i][0] = i * gap

    for j in range(cols):
        matrix[0][j] = j * gap

    # Fill scoring matrix
    for i in range(1, rows):
        for j in range(1, cols):

            if seq1[i - 1] == seq2[j - 1]:
                score = match
            else:
                score = mismatch

            diagonal = matrix[i - 1][j - 1] + score
            up = matrix[i - 1][j] + gap
            left = matrix[i][j - 1] + gap

            matrix[i][j] = max(diagonal, up, left)

    # Traceback
    aligned1 = ""
    aligned2 = ""

    i = len(seq1)
    j = len(seq2)

    while i > 0 or j > 0:

        if i > 0 and j > 0:
            score = match if seq1[i-1] == seq2[j-1] else mismatch

            if matrix[i][j] == matrix[i-1][j-1] + score:
                aligned1 = seq1[i-1] + aligned1
                aligned2 = seq2[j-1] + aligned2
                i -= 1
                j -= 1
                continue

        if i > 0 and matrix[i][j] == matrix[i-1][j] + gap:
            aligned1 = seq1[i-1] + aligned1
            aligned2 = "-" + aligned2
            i -= 1
        else:
            aligned1 = "-" + aligned1
            aligned2 = seq2[j-1] + aligned2
            j -= 1

    return aligned1, aligned2, matrix[-1][-1]


# Test alignment
sequence1 = "ATGCGT"
sequence2 = "ATGCAT"

a, b, score = needleman_wunsch(sequence1, sequence2)

print("\nGeneWeaver - DNA Alignment")
print("--------------------------")
print("Sequence 1:", a)
print("Sequence 2:", b)
print("Alignment Score:", score)
