# OGT-Extraction-and-Application
This is a project for OGT(Optimal Growth Temperature) data extraction in journals and the prediction of Enzyme's Topt(Temperature optimum) based on the OGT and features of protein.

### Requirements for Extraction based on Language Models 
```
cd OGT_extraction
conda create -n torch python=3.9 -y
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```
