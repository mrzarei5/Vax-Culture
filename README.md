# Vax-Culture: A Dataset for Studying Vaccine Discourse on Twitter
This is the dataset and code repository for ['Vax-Culture: A Dataset for Studying Vaccine Discourse on Twitter'](address will be added). In this paper, we present Vax-Culture, a data set of vaccine-related Tweets, manually annotated with the help of a team of annotators with a background in communications and journalism. Please refer to [dataset](https://github.com/mrzarei5/Vax-Culture/tree/main/dataset) for more information and to download dataset files. This repository also hosts the codes for the five baseline tasks introduced in our paper including four classification and one sequence generation tasks on Vax-Culture dataset.

## Requirements
### Installation
Create a conda environment and install dependencies:
```bash
conda create --name Vax_Culture --file requirements.txt
conda activate Vax_Culture
```

### Dataset
1. Copy the contents of Vax-Culture dataset directory [(link)](https://github.com/mrzarei5/Vax-Culture/tree/main/dataset) to `filelists`. 

2. Pull the tweets using their provided unique identifiers from Twitter and append their texts to `filelists/Vax_Culture.csv` under the column name `tweet_text`. (Any subset of the tweets in `Vax_Culture.csv` can be dropped. Just remember not to alter the header of columns.)

3. Create train, validation and test subsets by running `filelists/prepare_datasets.py`:
```bash
python prepare_datasets.py
```

### Classification Tasks
Run `classification.py` with the desired model and problem task:
```bash
python classification.py --model_name Bert_base --problem communicated_message
```
Parameter `model_name` can be selected from `Bertweet`, `Bert_base`, `Bert_large`, `Roberta_base` and `Roberta_large` for any of the four classification problem tasks.
Parameter `problem` can be selected from any of the values: `inaccurate_or_misleadning` for misleading or inaccurate information detection task, `communicated_message` for communicated message prediction task, `multilabel_criticism` for subjects of criticism prediction task, `multilabel_support` for subjects of support/promote prediction task.

### Text Generation Task
Run `generation.py` with the desired model:
```bash
python generation.py --model_name Bart_large --problem gen
```
Parameter `model_name` can be `Bart_large` or `T5_large`.







