import numpy as np
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right

import os
from utils import parse_args
from TweetNormalizer import normalizeTweet_other
import pprint
import evaluate
import nltk
from dataset import prepare_datasets_gen
import configs




metric = evaluate.load("rouge")

def postprocess_text_rouge(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds_ = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels_ = tokenizer.batch_decode(labels, skip_special_tokens=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

    decoded_preds_rouge, decoded_labels_rouge = postprocess_text_rouge(decoded_preds_, decoded_labels_)
    result_rouge = metric.compute(predictions=decoded_preds_rouge, references=decoded_labels_rouge, use_stemmer=True)
    result_rouge = {k: round(v * 100, 4) for k, v in result_rouge.items()}
    result_rouge["gen_len"] = np.mean(prediction_lens)
    return result_rouge

if __name__=='__main__':
    args_ = parse_args()
    num_workers = args_.num_workers
    dataset_dir = args_.dataset_dir
    model_name = args_.model_name
    epochs = args_.epochs
    problem = args_.problem

    random_state = configs.random_state
    batch_size = configs.batch_size


    
    toekns_added = ['HTTPURL', '@USER']


    if model_name == 'T5_large':
        tokenizer = T5Tokenizer.from_pretrained(configs.models_dic[model_name])
        model = T5ForConditionalGeneration.from_pretrained(configs.models_dic[model_name])
        learning_rate = configs.learning_rate_T5
    elif model_name == 'Bart_large':
        tokenizer = BartTokenizer.from_pretrained(configs.models_dic[model_name])
        model = BartForConditionalGeneration.from_pretrained(configs.models_dic[model_name])
        learning_rate= configs.learning_rate_Bart

    for se in toekns_added:
        tokenizer.add_tokens(se)
        model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    dataset_train, dataset_val, dataset_test = prepare_datasets_gen(dataset_dir, model_name, tokenizer, model)
    
    output_dir = os.path.join('./results', model_name + '_' + problem)
    logs_dir = os.path.join('./logs', model_name + '_' + problem)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    args_train = Seq2SeqTrainingArguments(output_dir,
    evaluation_strategy = "epoch",
    save_strategy='epoch',
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=1,
    num_train_epochs=epochs,
    predict_with_generate=True,
    fp16=True,
    seed=random_state,
    load_best_model_at_end=True,
    overwrite_output_dir = True,
    logging_dir=logs_dir,
    logging_strategy='epoch',
    )

    compute_met = compute_metrics

    trainer = Seq2SeqTrainer(
    model,
    args_train,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output_dir+'/'+'best_model')

    val_results = trainer.evaluate()
    test_results = trainer.evaluate(dataset_test)

    with open(output_dir+'/results.txt', 'w') as results_file:
        results_file.write(pprint.pformat(val_results))
        results_file.write('\n\n\n')
        results_file.write(pprint.pformat(test_results))
