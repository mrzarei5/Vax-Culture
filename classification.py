import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
import os
from utils import *
from dataset import prepare_datasets

import pprint
import configs

class Dataset(torch.utils.data.Dataset):
   def __init__(self, texts, labels=None):
       self.encodings = tokenizer(texts, truncation=True, max_length=128)
       self.labels = labels
   def __getitem__(self, idx):
       item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
       if self.labels:
           item["labels"] = torch.tensor(self.labels[idx])
       return item
   def __len__(self):
       return len(self.encodings["input_ids"])
   
if __name__=='__main__':
    args_ = parse_args()

    num_workers = args_.num_workers
    dataset_dir = args_.dataset_dir
    model_name = args_.model_name
    epochs = args_.epochs
    problem = args_.problem

    random_state = configs.random_state
    learning_rate = configs.learning_rate_class_tasks
    weight_decay = configs.weight_decay_class_tasks
    batch_size = configs.batch_size

    problem_type, num_labels  = get_problem_settings(problem)


    model = AutoModelForSequenceClassification.from_pretrained(configs.models_dic[model_name], problem_type = problem_type, num_labels = num_labels)
    tokenizer = AutoTokenizer.from_pretrained(configs.models_dic[model_name])

    tokens_added = ['HTTPURL', '@USER']
    
    for se in tokens_added:
        tokenizer.add_tokens(se)
        model.resize_token_embeddings(len(tokenizer))
    
    data_collator = DataCollatorWithPadding(tokenizer)

    dataset_train, dataset_val, dataset_test = prepare_datasets(problem, model_name, dataset_dir, tokenizer)

    
    output_dir = os.path.join('./results', model_name + '_' + problem)
    logs_dir = os.path.join('./logs', model_name + '_' + problem)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    
    if problem == 'inaccurate_or_misleadning':
        compute_met = compute_metrics_binary
    elif problem == 'communicated_message':
        compute_met = compute_metrics
    else:
        compute_met = compute_metrics_multilabel

    args_train = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logs_dir,
        logging_strategy='epoch',
        evaluation_strategy="epoch",
        metric_for_best_model="f1",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        seed=random_state,
        load_best_model_at_end=True,
        learning_rate= learning_rate,
        weight_decay= weight_decay,
        overwrite_output_dir = True,
        save_total_limit=1,
        save_strategy='epoch',
        fp16=True)
    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_met,
        data_collator=data_collator
        )


    trainer.train()
    trainer.save_model(output_dir+'/'+'best_model')

    val_results = trainer.evaluate()
    test_results = trainer.evaluate(dataset_test)

    

    with open(output_dir+'/results.txt', 'w') as results_file:
        results_file.write(pprint.pformat(val_results))
        results_file.write('\n\n\n')
        results_file.write(pprint.pformat(test_results))
