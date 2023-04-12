
models_dic = {"Bart_large":"facebook/bart-large", "T5_large":"t5-large", 
              "Bertweet":"vinai/bertweet-covid19-base-cased", "Bert_base":"bert-base-cased", 
              "Bert_large":"bert-large-cased", "Roberta_base":"roberta-base",
              "Roberta_large":"roberta-large"}


random_state = 10
batch_size = 4
learning_rate_class_tasks=1e-5
weight_decay_class_tasks=0.01

learning_rate_T5 = 3e-4
learning_rate_Bart = 1e-5
