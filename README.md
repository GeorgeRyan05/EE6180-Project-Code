# EE6180 Project
# Initial Setup
## 1. Clone the repository
## 2. Install the required packages
Code should work with Python 3.12 or 3.13 (I used 3.13).
```bash
# conda env create -f environment.yml
conda activate ryan
pip install -e .
# image_captioning_with_blip does not mean the BLIP model, I am simply using it as that was the initial name of the project.
# Please note that Phi 4 Multimodal will not run with transformers 4.53, it does run with version 4.51. Phi 4 may also be incompatible with Qwen 2.5.
```
If possible, also install flash_attn (`pip install flash_attn --no-build-isolation`).
## 3. Set up data and folders
```bash
# mastani is the folder containing Frames.json and get_dataset.py
python ./mastani/get_dataset.py
mkdir data_subset
mkdir .logs
python split_data.py
```
# Running the code
Running inference for a single model on a folder of images. Change the model and dataset classes as needed.
```bash
python inference.py
```
Running inference for an ensemble after obtaining .tsv or .json files for each of the models. (Should be in the outputs folder by default. The logs will also contain the path to the .tsv files (the JSON files use the same stem, and the .tsv files may not actually be readable if newlines are present in captions, they are a backup.))
```bash
python inference-ensemble.py
```
If you want to get readable files for comparisons, run `human_eval [-h] image_set ground_truth_captions generated_captions [markdown_file]`. These will work with VSCode's preview. Ignore formatting issues, as lines are wrapped (poorly) to ensure that the image remains visible.
# Relevant Output Files
Blip2__Salesforce__blip2-flan-t5-xxl.tsv  
Llama3.2__meta-llama__Llama-3.2-11b-Vision-Instruct.json (Llama 3.2)  
Qwen2.5__Qwen__Qwen-2.5-Omni-7B.tsv  
Phi4Ensemble__microsoft__Phi-4-mini-instruct__8.tsv (Ensemble with Llama)  
Phi4Ensemble__microsoft__Phi-4-mini-instruct__2.tsv (Ensemble without Llama)  
Phi4__microsoft__Phi-4-multimodal-instruct.tsv (Phi 4)  
Phi4__microsoft__Phi-4-multimodal-instruct__1.tsv (Used in ensembles instead of above)  
Phi4Sam__microsoft__Phi-4-multimodal-instruct.json (Phi 4 with SAM)  
captions_GSAM.csv (captions from Llama 3.2 with SAM)  
Phi4Ensemble__microsoft__Phi-4-mini-instruct__17.json (didn't work well, attempt at   ensembling with SAM)
Phi4Ensemble__microsoft__Phi-4-mini-instruct__18.json (didn't work well, attempt at   ensembling with SAM)
batch_descriptors.tsv (outputs of SAM 2.0)  