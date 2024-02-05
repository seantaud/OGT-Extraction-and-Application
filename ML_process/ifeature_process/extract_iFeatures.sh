mkdir Extracted_iFeatures

cd iFeature

for tpe in  {AAC,,CKSAAP,DPC,DDE,GAAC,CKSAAGP,GDPC,GTPC,NMBroto,Moran,Geary,CTDC,CTDT,CTDD,CTriad,KSCTriad,SOCNumber,QSOrder,PAAC,APAAC}
do python iFeature.py --type $tpe --out ../Extracted_iFeatures/$tpe.csv --file ../Topt_fasta.fasta
done

