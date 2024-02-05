from Bio.SeqUtils.ProtParam import ProteinAnalysis

x = ProteinAnalysis("cc.fasta")

print(x.count_amino_acids())

import os
import sys

path = r"C:\Users\captain\Desktop\frequencies\candidatus_magnetomorum_sp_hk_1\pep\Candidatus_magnetomorum_sp_hk_1.SAG5.pep.all.fa"
with open(path) as f:
    Acid_frequencies = []
    Dipeptide_frequencies = {}
    for each_line in f:
        if each_line[0] == '>':
            continue

        for acid in each_line:
            Acid_frequencies.append(acid)

        Length = len(each_line)
        for i in range(Length - 1):
            dipeptide = (each_line[i] + each_line[i + 1])
            if each_line[i + 1] != '\n':
                if dipeptide in Dipeptide_frequencies:
                    Dipeptide_frequencies[dipeptide] += 1
                else:
                    Dipeptide_frequencies[dipeptide] = 1

Acid_Type = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
for acid in Acid_Type:
    print('{}:{}'.format(acid, Acid_frequencies.count(acid)), end=' ')
print()
print(sorted(Dipeptide_frequencies.items()))