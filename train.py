from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import sys
import datetime
import time
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from sklearn.metrics import f1_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, BertConfig, AutoTokenizer , AutoModelForTokenClassification, TrainingArguments, Trainer , DataCollatorForTokenClassification)
from transformers import AdamW, get_scheduler
logger = logging.getLogger(__name__)
from utils import compute_metrics, get_Dataset, get_label_map

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default="train.txt", type=str)
    parser.add_argument("--eval_file", default="eval.txt", type=str)
    parser.add_argument("--test_file", default="test.txt", type=str)
    parser.add_argument("--mapper_file", default="mapper.pickle", type=str)
    parser.add_argument("--model_checkpoint", default="bert-base-uncased", type=str)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--input_dir", default="input", type=str)
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="bert-base-uncased", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_eval", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--num_train_epochs", default=1000, type=float)
    parser.add_argument("--warmup_proprotion", default=0.1, type=float)
    parser.add_argument("--use_weight", default=1, type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--fp16", default=False)
    parser.add_argument("--loss_scale", type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--logging_steps", default=5, type=int)
    parser.add_argument("--clean", default=False, action="store_true", help="clean the output dir")
    args = parser.parse_args()
    print(args)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        logger.info(f"device: {device} n_gpu: {n_gpu}")
    else :
        device = torch.device("cpu")
    args.device = device
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(),"output")
    if args.clean and args.do_train:
        # logger.info("清理")
            if os.path.exists(args.output_dir):
                def del_file(path):
                    ls = os.listdir(path)
                    for i in ls:
                        c_path = os.path.join(path, i)
                        print(c_path)
                        if os.path.isdir(c_path):
                            del_file(c_path)
                            os.rmdir(c_path)
                        else:
                            os.remove(c_path)
            try:
                del_file(args.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_checkpoint, 
                    do_lower_case=args.do_lower_case)
    if not os.path.exists(os.path.join(args.output_dir, "eval")):
        os.makedirs(os.path.join(args.output_dir, "eval"))

    tag2id, id2tag, num_tags = get_label_map(os.path.join(args.input_dir,args.mapper_file))
    print(tag2id)
    print(id2tag)
    if args.do_train:
        model = AutoModelForTokenClassification.from_pretrained(args.model_checkpoint, id2label = id2tag, label2id = tag2id )
        model.to(device)
        
        if device.type =="cuda" and n_gpu > 1:
            model = torch.nn.DataParallel(model)
        writer = SummaryWriter(log_dir = os.path.join(args.output_dir, "eval"), comment="Linear")
        #return pytorch Dataset
        train_data = get_Dataset(tokenizer, os.path.join(args.input_dir,args.train_file), tag2id, maxlength=256)
        print(train_data[0:1])
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        if args.do_eval:
            eval_data = get_Dataset(tokenizer,os.path.join(args.input_dir,args.eval_file), tag2id, maxlength=256)
        if args.max_steps > 0:
            num_training_steps = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        loss_fn = torch.nn.CrossEntropyLoss()
        scheduler = get_scheduler(name="cosine", optimizer= optimizer, num_warmup_steps= args.warmup_steps, num_training_steps= num_training_steps)
        #save infor
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Total optimization steps = %d", num_training_steps)
        
        model.train()
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        best_f1 = 0.0
        for ep in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids,  segment_ids,input_mask, labels = batch
                # print(input_ids[0])
                # print(segment_ids[0])
                # print(input_mask[0])
                # print(labels[0])
                ouput = model(input_ids= input_ids, attention_mask = input_mask, labels = labels)
                loss = ouput.loss
                if device.type =="cuda" and n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                    
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        tr_loss_avg = (tr_loss-logging_loss)/args.logging_steps
                        writer.add_scalar("Train/loss", tr_loss_avg, global_step)
                        logging_loss = tr_loss
            if args.do_eval:
                score = compute_metrics(args, eval_data, model, id2tag, device.type)
                
                # add eval result to tensorboard
                f1_score = score["f1"]
                writer.add_scalar("Eval/precision", score["precision"], ep)
                writer.add_scalar("Eval/recall", score["recall"], ep)
                writer.add_scalar("Eval/f1_score", score["f1"], ep)
                # save the best performs model
                if f1_score > best_f1 :
                    logger.info(f"----------the best f1 score is {f1_score}")
                    best_f1 = f1_score
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)

                # Good practice: save your training arguments together with the trained model
                    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

            # logger.info(f'epoch {ep}, train loss: {tr_loss}')
        # writer.add_graph(model)
        writer.close()

if __name__ == "__main__":
    main()
    pass