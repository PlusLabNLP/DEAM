import argparse
import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr
import json

from transformers import (
    Trainer,
    TrainingArguments,
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizerFast)
from sklearn.metrics import f1_score, accuracy_score

random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
device='cpu'
if torch.cuda.is_available():
    device='cuda'
    torch.cuda.manual_seed_all(1000)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        '''
        Param:
            encodings: encodings of inputs
            labels: labels of inputs
        '''
        self.encodings = encodings
        print("self.encodings.keys() in Dataset", self.encodings.keys())
        self.labels = labels

    def __getitem__(self, idx):
        '''get the specifeid item's encodings and label 
        Param:
            idx: index of an input
        '''
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        '''number of data
        '''
        if not self.labels:
            return 0
        return len(self.labels)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        '''
        Param:
            encodings: encodings of inputs
        '''
        self.encodings = encodings

    def __getitem__(self, idx):
        '''get the specifeid item's encodings and label 
        Param:
            idx: index of an input
        '''
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        return item

    def __len__(self):
        '''number of data
        '''
        return len(self.encodings.input_ids)


def load_data(data_path):
    '''load data
        Param:
            data_path: path of the input data
    '''
    fr = open(data_path, 'r')
    lines=fr.readlines()
    convs=[]
    orig_convs=[]
    labels=[]
    for ind, line in enumerate(lines):
        line=line.split('\n')[0]
        if not line:
            print('error line {} is an empty conversation'.format(ind))
        else:
            parts=line.split('</UTT>')
            label=round(float(parts[-1]))
            conv=' '.join(parts[:-1])
            convs.append(conv)
            orig_convs.append('</UTT>'.join(parts[:-1]))
            labels.append(label)
    return convs, orig_convs, labels

def load_test_data(data_path):
    '''load test data
        Param:
            data_path: path of the input test data
    '''
    fr = open(data_path, 'r')
    lines=fr.readlines()
    convs=[]
    orig_convs=[]
    for ind, line in enumerate(lines):
        line=line.split('\n')[0]
        if not line:
            print('error line {} is an empty conversation'.format(ind))
        else:
            conv=line.split('\n')[0]
            parts=line.split('</UTT>')
            conv=' '.join(parts[:-1])
            convs.append(conv)
            orig_convs.append(' '.join(parts[:-1]))
    return convs, orig_convs

def load_and_cache_examples(args, data, labels=None, type_data="train", additional_filename_postfix=""):
    '''load  and cache input data
        Param:
            data: input data
            labels: label of data
            type_data: whether it is train/valid/test
            additional_filename_postfix: which test set
    '''
    if not os.path.exists(args.model_path):
        os.mkdir(os.path.join('./',args.model_path))
    cached_features_file = os.path.join(args.model_path,"cached_{}_{}_{}{}".format(type_data, args.model_type, args.max_length, additional_filename_postfix))
    if os.path.exists(cached_features_file):
        encodings = torch.load(cached_features_file)
    else:
        encodings = tokenizer(data, truncation=True, padding=True, max_length=args.max_length)
        torch.save(encodings, cached_features_file)
    if labels:
        dataset=Dataset(encodings, labels)
    else:
        dataset=TestDataset(encodings)
    print("len(dataset) is {} in load_and_cache_examples with type_data {}".format(len(dataset), type_data))
    return dataset


def get_metrics(output):
    '''return accuracy and f1 scores
        Param:
            output: ground-truth and predicted scores
    '''
    labels = output.label_ids
    preds = output.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1=f1_score(labels,preds)
    return {'accuracy': acc, 'f1': f1}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='./data/topical_chat/train_amr_manamr_cont_coref_pirel_eng.txt', required=True, help="path of train conversations")
    parser.add_argument('--valid_data_path', default='./data/topical_chat/valid_amr_manamr_cont_coref_pirel_eng.txt', required=True, help="path of valid conversations")
    parser.add_argument('--model_path', default='coh_models/', required=True, help="path of trained model")
    parser.add_argument('--max_length', default=512, type=int, required=False, help="maximum length of input conversations")
    parser.add_argument('--num_labels', default=2, type=int, required=False, help="number of labels for classifying conversations")
    parser.add_argument('--num_epochs', default=3, type=int, required=False, help="number of training epochs")
    parser.add_argument('--train_batch_size', default=8, type=int, required=False, help="batch size for training")
    parser.add_argument('--valid_batch_size', default=8, type=int, required=False, help="batch size for evaluating")
    parser.add_argument('--warmup_steps', default=500, type=int, required=False, help="number of warmup steps for lr scheduler")
    parser.add_argument('--warmup_decay', default=0.01, required=False, help="weight decay value")
    parser.add_argument('--model_name', default='roberta-large', required=False, help="name of the model")
    parser.add_argument('--model_type', default='roberta', required=False, help="type of the model")
    parser.add_argument('--logging_steps', default=100, type=int, required=False, help="logging model weights")
    parser.add_argument('--save_steps', default=200, type=int, required=False, help="save model weights")
    parser.add_argument('--learning_rate', default=0.0001, type=float, required=False, help="learning rate")
    parser.add_argument('--mode', default='valid',  required=False, help="mode (train/valid/predict)")
    args=parser.parse_args()

    MODEL_CLASSES = {
       "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizerFast),
	}
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


    if args.mode=='train':
        tokenizer = tokenizer_class.from_pretrained(args.model_name, do_lower_case=True)
        model = model_class.from_pretrained(args.model_name, num_labels=args.num_labels).to(device)

    else:
        tokenizer = tokenizer_class.from_pretrained(args.model_path)
        model = model_class.from_pretrained(args.model_path, num_labels=args.num_labels).to(device)
        print('load model {}'.format(args.model_path))


    print('loading train/valid data ...')
    train_convs, train_orig_convs, train_labels = load_data(args.train_data_path)
    valid_convs, valid_orig_convs, valid_labels = load_data(args.valid_data_path)
    print('Done!')

    #Create dataset including features for the model
    train_dataset=load_and_cache_examples(args, train_convs, train_labels, "train")
    valid_dataset=load_and_cache_examples(args, valid_convs, valid_labels, "valid")


    training_args = TrainingArguments(output_dir=args.model_path+'/'+args.model_type, num_train_epochs=args.num_epochs,  per_device_train_batch_size=args.train_batch_size,
                                          per_device_eval_batch_size=args.valid_batch_size, warmup_steps=args.warmup_steps, weight_decay=args.warmup_decay,learning_rate=args.learning_rate,
                                          logging_dir=args.model_path+'/logs', load_best_model_at_end=True, metric_for_best_model='loss', #use the evaluation loss to save best models
                                          logging_steps=args.logging_steps, evaluation_strategy="steps", save_total_limit=1)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=valid_dataset, tokenizer=tokenizer, compute_metrics=get_metrics)

    if args.mode=='train':
        trainer.train()
        model.save_pretrained(args.model_path)
        tokenizer.save_pretrained(args.model_path)
        output=trainer.evaluate()
        print('the results for evaluation set is')
        print(output)

    elif args.mode=='valid':
        print('check performance on eval set')
        output=trainer.evaluate()
        print(output)

    elif args.mode=='predict':
        tests = {
            'fed_test_coherence_orig': ("data/fed", 'fed_test_coherence.txt'),
            'fed_test_overall_orig': ("data/fed", 'fed_test_overall.txt'),
            'dstc9_test_coherence_orig': ("data/dstc9", 'dstc9_test_coherence_averaged.txt'),
            'dstc9_test_overall_orig': ("data/dstc9", 'dstc9_test_overall_averaged.txt'),
        }
        for which_test in tests:
            test_convs, test_orig_convs = load_test_data(os.path.join(*tests[which_test]))
            test_dataset=load_and_cache_examples(args, test_convs, labels=None, type_data="test", additional_filename_postfix="_"+which_test)
            print('Predicting {} which contains {} sentences...'.format(which_test, len(test_convs)))
            output=trainer.predict(test_dataset)
            prob=torch.nn.Softmax(dim=1)
            scores=prob(torch.tensor(output.predictions))
            fn=args.model_path+'{}_preds.txt'.format(which_test)
            fw=open(fn, 'w')
            for ind, conv in enumerate(test_orig_convs):
                fw.write(conv+'</UTT>'+str(scores[ind][1].item())+'\n')
            print("Prediction result saved to {}.\n-----------------------------------".format(fn))
        