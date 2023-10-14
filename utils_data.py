import select

import torch
import random
import copy
import re
import os
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.utils.rnn as rnn_utils
import pdb


def prepare_tokenizer(args):
    if args.encoder_type == 'bert':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args.plm_path, do_lower_case=False, cache_dir='')
    if args.encoder_type == 'bertu':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args.plm_path, do_lower_case=False, cache_dir='')
    return tokenizer


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.tokenizer = prepare_tokenizer(args)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

    def tokenize(self, texts, text_type):
        tokens = []
        input_ids = []
        for text in tqdm(texts, desc=f'[tokenize {text_type}]', leave=True):
            tokens.append(self.tokenizer.tokenize(str(text)))
            input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens[-1]))
        return tokens, input_ids

    def padding(self, input_ids, max_len, text_type):
        _input_ids = list(input_ids)
        token_type_ids = []
        for i, item in enumerate(_input_ids):
            if max_len == -1:
                _input_ids[i] = [self.cls_token_id] + item[:510] + [self.sep_token_id]
                token_type_ids.append([1] * (len(item[:510]) + 2))
            else:
                _input_ids[i] = [self.cls_token_id] + item[:max_len - 2] + [self.sep_token_id]
                token_type_ids.append([1] * (len(item[:max_len - 2]) + 2))

        max_len = max([len(s) for s in _input_ids])
        input_ids = np.array([item + [self.pad_token_id] * (max_len - len(item)) for item in _input_ids], dtype=np.int)
        attention_mask = np.array(
            [[1] * len(item) + [self.pad_token_id] * (max_len - len(item)) for item in _input_ids], dtype=np.int)
        token_type_ids = np.array([item + [self.pad_token_id] * (max_len - len(item)) for item in token_type_ids],
                                  dtype=np.int)
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        token_type_ids = torch.LongTensor(token_type_ids)

        return input_ids.to(self.device), attention_mask.to(self.device), token_type_ids.to(self.device)



class LTRRaceDataset(BaseDataset):
    def __init__(self, args, split='train'):
        super(LTRRaceDataset, self).__init__(args)
        self.split = split
        self.version = args.dataset
        self.data_folder = ''
        self.process()

    def process(self):

        cached_dir = f"./cached_data/{self.args.dataset}"
        if not os.path.exists(cached_dir):
            os.makedirs(cached_dir)
        cached_dataset_file = os.path.join(cached_dir, self.split)

        logging.info("Creating instances from dataset file at %s", self.data_folder)
        with open(self.data_folder + f"/{self.split}_data.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if self.split == 'test':
            self.texts = raw_data['texts']
            self.questions = raw_data['questions']
            self.answers = raw_data['answers']
            self.distractors = raw_data['candidates']
            self.ground_truths = raw_data['ground_truths']
            self.answer_ids = raw_data['answer_ids']
        else:
            self.texts = raw_data['texts']
            self.questions = raw_data['questions']
            self.answers = raw_data['answers']
            self.distractors = raw_data['distractors']
            self.question_ids = raw_data['question_ids']
        
        if os.path.exists(cached_dataset_file):
            logging.info("Loading dataset from cached file %s", cached_dataset_file)
            data_dict = torch.load(cached_dataset_file)
            text_tokens = data_dict["text_tokens"]
            self.text_input_ids = data_dict["text_input_ids"]
            question_tokens = data_dict["question_tokens"]
            self.question_input_ids = data_dict["question_input_ids"]
            answer_tokens = data_dict["answer_tokens"]
            self.answer_input_ids = data_dict["answer_input_ids"]
            distractor_tokens = data_dict["distractor_tokens"]
            self.distractor_input_ids = data_dict["distractor_input_ids"]

        else:
            text_tokens, self.text_input_ids = self.tokenize(self.texts, 'texts')
            question_tokens, self.question_input_ids = self.tokenize(self.questions, 'questions')
            answer_tokens, self.answer_input_ids = self.tokenize(self.answers, 'answers')
            distractor_tokens, self.distractor_input_ids = self.tokenize(self.distractors, 'distractors')

            while len(self.question_input_ids) < len(self.distractor_input_ids):
                self.text_input_ids.append([0])
                self.question_input_ids.append([0])
                self.answer_input_ids.append([0])


            saved_data = {
                'text_input_ids': self.text_input_ids,
                'question_input_ids': self.question_input_ids,
                'answer_input_ids': self.answer_input_ids,
                'distractor_input_ids': self.distractor_input_ids,
                'text_tokens': text_tokens,
                'question_tokens': question_tokens,
                'answer_tokens': answer_tokens,
                'distractor_tokens': distractor_tokens
            }

            logging.info("Saving processed dataset to %s", cached_dataset_file)
            torch.save(saved_data, cached_dataset_file)

        logging.info(f"question: {self.questions[0]}")
        logging.info(f'question tokens: {question_tokens[0]}')
        logging.info(f'question input ids: {self.question_input_ids[0]}')
        logging.info('')
        logging.info(f"distractor: {self.distractors[0]}")
        logging.info(f'distractor tokens: {distractor_tokens[0]}')
        logging.info(f'distractor input ids: {self.distractor_input_ids[0]}')
        logging.info('')

        

    def __len__(self):
        return len(self.question_input_ids)

    def __getitem__(self, idx):
        if self.split == 'test':
            return (self.args,
                    self.text_input_ids[idx],
                    self.question_input_ids[idx],
                    self.answer_input_ids[idx],
                    self.distractor_input_ids[idx])
        else:
            return (self.args,
                    self.text_input_ids[idx],
                    self.question_input_ids[idx],
                    self.answer_input_ids[idx],
                    self.distractor_input_ids[idx],
                    self.question_ids[idx])

    def collate_fn(self, raw_batch):
        args = raw_batch[-1][0]
        batch = dict()
        if self.split == 'test':
            _, text_input_ids, question_input_ids, answer_input_ids, distractor_input_ids = list(zip(*raw_batch))
            batch['src_input_ids'], batch['src_attention_mask'], batch['src_token_type_ids'] = self.padding_tqa(
                text_input_ids, question_input_ids, answer_input_ids, -1, 'question_answers')
            batch['tgt_input_ids'], batch['tgt_attention_mask'], batch['tgt_token_type_ids'] = self.padding_d(
                distractor_input_ids, -1, 'distractors')
        else:
            _, text_input_ids, question_input_ids, answer_input_ids, distractor_input_ids, question_ids = list(
                zip(*raw_batch))
            batch['src_input_ids'], batch['src_attention_mask'], batch['src_token_type_ids'] = self.padding_tqa(
                text_input_ids, question_input_ids, answer_input_ids, self.args.max_text_question_answer_len, 'question_answers')
            batch['tgt_input_ids'], batch['tgt_attention_mask'], batch['tgt_token_type_ids'] = self.padding_d(
                distractor_input_ids, self.args.max_distractor_len, 'distractors')
            batch['src_ids'] = torch.LongTensor(list(question_ids)).to(self.device)
        return batch


    def padding_d(self, input_ids, max_len, text_type):
        if max_len == -1:
            max_len = 512
        _input_ids = list(input_ids)
        token_type_ids = []
        for i, item in enumerate(_input_ids):
                _input_ids[i] = [self.cls_token_id] + item[0 : min(len(item), max_len - 2)] + [self.sep_token_id]
                token_type_ids.append([1] * (len(item[0 : min(len(item), max_len - 2)]) + 2))

        max_len = max([len(s) for s in _input_ids])
        input_ids = np.array([item + [self.pad_token_id] * (max_len - len(item)) for item in _input_ids], dtype=np.int)
        attention_mask = np.array(
            [[1] * len(item) + [self.pad_token_id] * (max_len - len(item)) for item in _input_ids], dtype=np.int)
        token_type_ids = np.array([item + [self.pad_token_id] * (max_len - len(item)) for item in token_type_ids],
                                  dtype=np.int)
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        token_type_ids = torch.LongTensor(token_type_ids)

        return input_ids.to(self.device), attention_mask.to(self.device), token_type_ids.to(self.device)


    def padding_tqa(self, t_input_ids, q_input_ids, a_input_ids, max_len, text_type):
        if max_len == -1:
            max_len = 512
        _t_input_ids = list(t_input_ids)
        _q_input_ids = list(q_input_ids)
        _a_input_ids = list(a_input_ids)
        _input_ids = []
        for i in range(len(_q_input_ids)):
            _input_ids.append(_t_input_ids[i] + [self.sep_token_id] +  _q_input_ids[i] + [self.sep_token_id] + _a_input_ids[i])
        token_type_ids = []
        for i, item in enumerate(_input_ids):
            _input_ids[i] = [self.cls_token_id] + item[max(len(item) - (max_len - 2), 0) :] + [self.sep_token_id]
            token_type_ids.append([1] * (len(item[max(len(item) - (max_len - 2), 0) :]) + 2))

        max_len = max([len(s) for s in _input_ids])
        input_ids = np.array([item + [self.pad_token_id] * (max_len - len(item)) for item in _input_ids], dtype=np.int)
        attention_mask = np.array(
            [[1] * len(item) + [self.pad_token_id] * (max_len - len(item)) for item in _input_ids], dtype=np.int)
        token_type_ids = np.array([item + [self.pad_token_id] * (max_len - len(item)) for item in token_type_ids],
                                  dtype=np.int)
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        token_type_ids = torch.LongTensor(token_type_ids)

        return input_ids.to(self.device), attention_mask.to(self.device), token_type_ids.to(self.device)





