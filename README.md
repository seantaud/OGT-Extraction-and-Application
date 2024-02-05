# OGT-Extraction-and-Application

This is a project for OGT(Optimal Growth Temperature) data extraction in journals and the prediction of Enzyme's Topt(Temperature optimum) based on the OGT and features of protein.

The data collection and pre-processing codes for OGT are stored in OGT-Extraction-and-Application/Literature_Acquisition_and_Context_Locating and those for Topt are in OGT-Extraction-and-Application/Get_topt_from_brenda_and_deal. The final database OGT_journal is OGT-Extraction-and-Application/OGT_journal.xlsx.

### Requirements for Extraction based on Language Models

```
cd OGT_extraction
conda create -n torch python=3.9 -y
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

### OGT Extraction using Extractive Question Anwering and Generative Question Answering

```
cd OGT_extraction
bash scripts/BERT_search
# For GPT models, change the bash code accordingly
```

### Machine Learning Experiments for Prediction of Topt using OGT and Protein Features
