import argparse
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for datasets')
    parser.add_argument('--dataset_dir', default='./dataset_dir', help='dataset location')
    parser.add_argument('--model_name',default='', help= 'select model name from Bertweet, Bert_base, Bert_large, Roberta_base, Roberta_large, Bart_large, T5_large')
    parser.add_argument('--epochs', type = int, default=40, help='Number of training epochs')
    parser.add_argument('--problem', default='', help='select problem name from inaccurate_or_misleadning, communicated_message, multilabel_criticism, multilabel_support, gen')
    
    return parser.parse_args()

def get_problem_settings(problem):
     problem_dic = {'communicated_message': ("single_label_classification", 3), 'inaccurate_or_misleadning': ("single_label_classification", 2), 
                      'multilabel_criticism': ('multi_label_classification', 12), 'multilabel_support': ('multi_label_classification', 11)}
     return problem_dic[problem]
        

def compute_metrics(p):
   pred, labels = p
   pred = np.argmax(pred, axis=1)
   accuracy = accuracy_score(y_true=labels, y_pred=pred)
   recall = recall_score(y_true=labels, y_pred=pred, average='macro')
   precision = precision_score(y_true=labels, y_pred=pred, average='macro')
   f1 = f1_score(y_true=labels, y_pred=pred, average='macro')
   return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def compute_metrics_binary(p):
   pred, labels = p
   pred = np.argmax(pred, axis=1)
   accuracy = accuracy_score(y_true=labels, y_pred=pred)
   recall = recall_score(y_true=labels, y_pred=pred, pos_label=1, average='binary')
   precision = precision_score(y_true=labels, y_pred=pred, pos_label=1, average='binary')
   f1 = f1_score(y_true=labels, y_pred=pred, pos_label=1, average='binary')
   return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def multi_label_metrics(predictions, labels, threshold=0.5):
   sigmoid = torch.nn.Sigmoid()
   probs = sigmoid(torch.Tensor(predictions))
   y_pred = np.zeros(probs.shape)
   y_pred[np.where(probs >= threshold)] = 1
   average_acc = (y_pred == labels).mean(axis=1).mean()*100
   recall = recall_score(y_true=labels, y_pred=y_pred, average=None)
   precision = precision_score(y_true=labels, y_pred=y_pred, average=None)
   f1 = f1_score(y_true=labels, y_pred=y_pred, average=None)
   metric_dic = {"accuracy":average_acc}
   
   counter = 1
   for re in recall:
        metric_dic['recall_'+str(counter)] = re
        counter += 1
   counter = 1
   for pr in precision:
        metric_dic['precision_'+str(counter)] = pr
        counter += 1
   counter = 1
   for f in f1:
        metric_dic['f1_'+str(counter)] = f
        counter += 1
   metric_dic['f1']=np.mean(f1)
   metric_dic['recall']=np.mean(recall)
   metric_dic['precision']=np.mean(precision)

   return metric_dic

def compute_metrics_multilabel(p):
   preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
   result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
   return result

