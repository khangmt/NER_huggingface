
import numpy as np
import pickle
import torch
import logging
from datasets import load_metric
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
logger = logging.getLogger(__name__)
import pandas as pd
import json
import evaluate
def compute_metrics(args, data, model, id2tag, deviceType):
    model.eval()
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)
    logger.info("***** Running eval *****")
    all_predictions = []
    all_labels = []
    for b_i, (input_ids, segment_ids, input_mask, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
        
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)
        labels = labels.to(args.device)
        with torch.no_grad():
            output = model(input_ids= input_ids, attention_mask = input_mask, labels = labels)
            logits = output.logits
        if deviceType == "cuda":
            logits = logits.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        else:
            logits = logits.detach().numpy()
            labels = labels.detach().numpy()
        predictions = np.argmax(logits, axis=-1)
        for prediction, label in zip(predictions, labels):
            assert len(prediction) == len(label) == 256
            for predicted_idx, label_idx in zip(prediction, label):
                if label_idx == -100:
                    continue
                all_predictions.append(id2tag[predicted_idx])
                all_labels.append(id2tag[label_idx])

          

    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=[all_predictions], references=[all_labels])
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
def align_labels_with_tokens(labels, word_ids, tag2id):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            
            label = -100 if word_id is None else tag2id[labels[word_id]]
            
            new_labels.append(label)

            
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            #same word as previous tokens
            label = tag2id[labels[word_id]]
            # If the label is B-XXX we change it to I-XXX
            if labels[word_id].startswith("B-"):
                temp = "I-" + labels[word_id][2:]
                label = tag2id[temp]
            
            new_labels.append(label)
    assert len(word_ids) == len (new_labels)
    assert new_labels is not None
    for n in new_labels:
        if n > len(tag2id):
            print("error")
            print(n)
    return new_labels

def get_label_map(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    tag2id = data["tag2id"]
    id2tag = data["id2tag"]
    num_tags = data["num_tags"]
    return tag2id, id2tag, num_tags

class InputFeatures(object):
    # """A single set of features of data."""
    def __init__(self, input_ids, segment_ids, input_mask, labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels

def get_feature_from_sentence(tokenizer, text ,labels, tag2id, maxlength = 512):

    tokenized_inputs = tokenizer(text, truncation=True, is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()
    labels = align_labels_with_tokens(labels, word_ids= word_ids, tag2id = tag2id)
    assert labels is not None
    assert len(tokenized_inputs["input_ids"]) == len(labels)
    if len(tokenized_inputs["input_ids"]) < maxlength:
        padding_length = maxlength - len(tokenized_inputs["input_ids"])
        tokenized_inputs["input_ids"].extend([0]*padding_length)
        tokenized_inputs["token_type_ids"].extend([0]*padding_length)
        tokenized_inputs["attention_mask"].extend([0]*padding_length)
        labels.extend([-100]* padding_length )
    for l in labels:
        if l == 101:
            print("error")
            print(l)
    # assert input_ids is None
    return InputFeatures(tokenized_inputs["input_ids"], tokenized_inputs["token_type_ids"],tokenized_inputs["attention_mask"], labels = labels)
def get_features_from_files(tokenizer, filepath, tag2id , maxlength = 512):# for csv file
    features = []
    raw_data = pd.read_csv(filepath, sep=",")      
    tag_lb = [i.split() for i in raw_data['tags'].values.tolist()]
    txts = raw_data['text'].values.tolist()
    for text, labels in zip (txts,tag_lb):
        feature = get_feature_from_sentence(tokenizer,text,labels,tag2id= tag2id)
        features.append(feature)
    return features
def get_Dataset(tokenizer, file, tag2id, maxlength = 512):
    print(tag2id)
    features = get_features_from_files2(tokenizer,file, tag2id= tag2id, maxlength = maxlength)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.labels for f in features], dtype=torch.long)
    data = DataSequence(all_input_ids,  all_segment_ids, all_input_mask, all_label_ids)
    return data

#text file
def get_features_from_files2(tokenizer, filepath, tag2id, maxlength = 512):
    with open(filepath,"r",encoding="utf-8") as f:
        lines = f.readlines()
    features = []
    texts = []
    labels = []
    txt = []
    label = []
    for l in lines:
        split = l.split()
        if len(split)<2: # this is the end of the sentence
            assert len(txt) == len(label)
            texts.append(txt)
            labels.append(label)
            txt = [] #reset txt and label
            label = []
        else:
            txt.append(split[0].strip())
            label.append(split[1].strip())
    
    for text, label in zip (texts,labels):
        feature = get_feature_from_sentence(tokenizer,text,label,tag2id= tag2id, maxlength= maxlength)
        features.append(feature)
    return features

data = get_features_from_files
save = r"C:\Users\Oblivion\Working Space\NER_huggingface_pipeline\input\mapper.pkl"
with open(save, "wb") as f:
    pickle.dump(data,f, protocol= pickle.HIGHEST_PROTOCOL)

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, input_ids, segment_ids, mask_ids, labels):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.mask_ids = mask_ids
        self.labels = labels
    
    def __len__(self):

        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.input_ids[idx],self.segment_ids[idx],self.mask_ids[idx],self.labels[idx]