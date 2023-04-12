import torch
import pandas as pd
from TweetNormalizer import normalizeTweet_bertweet, normalizeTweet_other
from datasets import Dataset as Dataset_org
import os
from transformers.models.bart.modeling_bart import shift_tokens_right
from datasets.utils import disable_progress_bar
disable_progress_bar()

multilabel_tasks_labels = {'multilabel_criticism':['criticism_politicians','criticism_pharmaceutical_companies',
        'criticism_public_health_officials', 'criticism_anti-vaxxers',
        'criticism_vaccine_mandates', 'criticism_vaccine_safety',
        'criticism_conservative_media', 'criticism_mainstream_media',
        'criticism_public_health_policy', 'criticism_democrats_or_liberals',
        'criticism_government', 'criticism_vaccine_effectiveness'],
        'multilabel_support': ['support_science', 'support_choice_freedom', 'support_natural_health',
        'support_vaccines', 'support_small_business',
        'support_alternative_remedies', 'support_relaxed_approach',
        'support_more_information', 'support_public_health_interventions',
        'support_global_response', 'support_religious_beliefs']}

class Dataset(torch.utils.data.Dataset):
   def __init__(self, texts, tokenizer, labels=None):
       self.encodings = tokenizer(texts, truncation=True, max_length=128)
       self.labels = labels
   def __getitem__(self, idx):
       item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
       if self.labels:
           item["labels"] = torch.tensor(self.labels[idx])
       return item
   def __len__(self):
       return len(self.encodings["input_ids"])
   
class Dataset_gen(torch.utils.data.Dataset):
   def __init__(self, texts, tokenizer, labels=None, max_input_length = 130, max_target_length = 150):
       self.encodings = tokenizer(texts, truncation=True, max_length=max_input_length)
       with tokenizer.as_target_tokenizer():
            self.labels = tokenizer(labels, truncation=True, max_length=max_target_length)
   def __getitem__(self, idx):
       item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
       if self.labels:
           item["labels"] = torch.tensor(self.labels['input_ids'][idx])
       return item
   def __len__(self):
       return len(self.encodings["input_ids"])

def normalize_tweet(texts, model_name):
    if model_name == 'Bertweet':
        normalizer = normalizeTweet_bertweet
    else:
        normalizer = normalizeTweet_other
    return [normalizer(t) for t in texts]

#convert categorila string labels to categorical numbers for communicated message prediction task
def convert_labels(data, labels_dic): 
    for label, label_id in labels_dic.items():
        data.loc[data['label']==label,['label']] = label_id
    return data

def prepare_multiclass_task(data, label_name, model_name, label_convertor = None):
    data.rename(columns={label_name:'label'}, inplace=True)
    if label_convertor:
        data = convert_labels(data, label_convertor)
    texts = normalize_tweet(data["tweet_text"].tolist(), model_name)
    labels = data['label'].to_list()
    return texts, labels

def prepare_multilabel_task(data, problem, model_name):
    label_names = multilabel_tasks_labels[problem]
        
    sample_nums = len(data)
    labels = [[] for t in range(sample_nums)] 
    for label_name in label_names:
        label_this_list = data[label_name].to_list()
        for index, label in enumerate(label_this_list):
            labels[index].append(float(label))
    texts = normalize_tweet(data["tweet_text"].tolist(), model_name)
    return texts, labels

def prepare_datasets(problem, model_name, dataset_dir, tokenizer):
    df_train = pd.read_pickle(os.path.join(dataset_dir,'df_train'))
    df_val = pd.read_pickle(os.path.join(dataset_dir,'df_val'))
    df_test = pd.read_pickle(os.path.join(dataset_dir,'df_test'))

    data_dic={'train':df_train,'val':df_val,'test':df_test}

    for data_name, data in data_dic.items():
        if problem == 'multilabel_criticism' or problem == 'multilabel_support': 
            texts, labels = prepare_multilabel_task(data, problem, model_name)
        else:
            label_convertor = {'Anti-vaccine':0, 'Pro-vaccine':1, 'Unsure about the vaccine':2} if problem == 'communicated_message' else None
            texts, labels = prepare_multiclass_task(data, problem, model_name, label_convertor)
        if data_name == 'train':
            dataset_train = Dataset(texts, tokenizer, labels)
        elif data_name == 'val':
            dataset_val = Dataset(texts, tokenizer, labels)
        elif data_name == 'test':
            dataset_test = Dataset(texts, tokenizer, labels)
    return dataset_train, dataset_val, dataset_test

def make_dataset_bart(dataset,tokenizer, model):
    input_encodings = tokenizer.batch_encode_plus(dataset['text'], truncation=True, max_length=128, padding="max_length")
    target_encodings = tokenizer.batch_encode_plus(dataset['label'], truncation=True, max_length=150, padding="max_length")

    labels = target_encodings['input_ids']
    decoder_input_ids = shift_tokens_right(torch.tensor(labels), model.config.pad_token_id, model.config.decoder_start_token_id)

    labels = [[-100 if token == tokenizer.pad_token_id else token for token in l] for l in labels]
    
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': decoder_input_ids,
        'labels': labels,
    }
    return encodings
    
def prepare_gen_task(data, model_name, tokenizer, model):
    if model_name == 'Bart_large':
        data = Dataset_org.from_pandas(pd.DataFrame.from_dict({'text':[normalizeTweet_other(t) for t in data['tweet_text'].to_list()],
                                                               'label':data['meaning'].to_list()})).map(lambda x: make_dataset_bart(x,tokenizer,model), batched=True)
        data = data.remove_columns(['text','label'])
    elif model_name == 'T5_large':
        data = Dataset_gen(['generate description: ' + normalizeTweet_other(t) for t in data['tweet_text'].to_list()], tokenizer, labels = data['meaning'].to_list())
        
    return data

def prepare_datasets_gen(dataset_dir, model_name, tokenizer, model):
    df_train = pd.read_pickle(os.path.join(dataset_dir,'df_train'))
    df_val = pd.read_pickle(os.path.join(dataset_dir,'df_val'))
    df_test = pd.read_pickle(os.path.join(dataset_dir,'df_test'))

    data_dic={'train':df_train,'val':df_val,'test':df_test}

    for data_name, data in data_dic.items():
        if data_name == 'train':
            dataset_train = prepare_gen_task(data, model_name, tokenizer, model)
        elif data_name == 'val':
            dataset_val = prepare_gen_task(data, model_name, tokenizer, model)
        elif data_name == 'test':
            dataset_test = prepare_gen_task(data, model_name, tokenizer, model)
    return dataset_train, dataset_val, dataset_test
