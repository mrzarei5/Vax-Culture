# Vax-Culture: A Dataset for Studying Vaccine Discourse on Twitter
This is the code repository for ['Vax-Culture: A Dataset for Studying Vaccine Discourse on Twitter'](address will be added). In this paper, we presented Vax-Culture, a data set of vaccine-related Tweets and annotated them with the help of a team of 
annotators with a background in communications and journalism. Vax-Culture consists of 6373 vaccine-related tweets accompanied by an extensive 
set of human-provided annotations including vaccine-hesitancy stance, indication of any misinformation in tweets, 
the entities criticized and supported in each tweet and the communicated message of each tweet. This dataset can be downloaded from the project website [link](address will be added). This repository hosts the codes for the five baseline tasks introduced in the paper including four classification and one sequence generation tasks on Vax-Culture dataset.

## Requirements
### Installation
Create a conda environment and install dependencies:
```conda create --name Vax_Culture --file requirements.txt`
conda activate Vax_Culture```

### Dataset
1. Download the dataset from [link](address will be added) and extract it in `dataset_dir`. 

2. Pull the tweets using their provided unique identifiers from Twitter and append their texts to `dataset_dir/Vax_Culture.csv` under the column name `tweet_text`. (Any subset of the tweets in `Vax_Culture.csv` can be dropped. Just remember not to alter the header of columns.)

3. Create train, validation and test subsets by running `dataset_dir/prepare_datasets.py`:

`python prepare_datasets.py`

###Classification Tasks
Run `classification.py` with the desired model and problem task:

`python classification.py --model_name Bert_base --problem communicated_message`

Parameter `model_name` can be selected from `Bertweet`, `Bert_base`, `Bert_large`, `Roberta_base` and `Roberta_large` for any of the four classification problem tasks.
Parameter `problem` can be selected from any of the values: `inaccurate_or_misleadning` for misleading or inaccurate information detection task, `communicated_message` for communicated message prediction task, `multilabel_criticism` for subjects of criticism prediction task, `multilabel_support` for subjects of support/promote prediction task.

###Text Generation Task
Run `generation.py` with the desired model:

`python generation.py --model_name Bart_large --problem gen`

Parameter `model_name` can be `Bart_large` or `T5_large`.







