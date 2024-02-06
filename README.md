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

The OGT extracted was subjected to relevant analysis experiments and machine learning experiments, all of which are stored in the directory named "OGT-Extraction-and-Application/ML_process". These contents include code and experimental data corresponding to each process.

- "1_Data processing.ipynb" is used to process the data after preliminary handling by the language model. It systematically carries out operations such as deduplication, correction, and merging on the raw data, ultimately resulting in a data file named "OGT_journal.xlsx". Intermediate data generated at each step is stored in the directory "OGT-Extraction-and-Application/ML_process/dataset".

- "2_Data analysis.ipynb" is utilized to analyze the final data obtained and to plot corresponding graphs.

- "3_Database construction.ipynb" is employed to process intermediate files generated in the first step to obtain data suitable for machine learning experiments. The relevant data is stored in "OGT-Extraction-and-Application/ML_process/ML_data".

- "4_ML_experiments.ipynb" contains the code for conducting machine learning experiments, including the setup of hyperparameters for each model. The code can be modified according to different experimental requirements. Additionally, this file includes the necessary data processing steps for the experiments.

- "5_ML_plot.ipynb" is responsible for generating graphs depicting the results of the machine learning experiments. The generated images are stored in the "graph" directory.

Furthermore, all generated images can be stored in the "graph" directory. The "ifeature_process" directory is used to obtain the ifeature feature set corresponding to protein sequences.
