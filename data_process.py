import argparse
import json
import logging
from itertools import combinations
import random
from rank_bm25 import BM25Okapi, BM25L
from tqdm import tqdm
from transformers import BertTokenizer
import numpy as np


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_valid_dis(dis, dataset):
    if dataset == 'mcql':
        if not dis:
            return False
        if '?' in dis:  # parsing error for formula
            return False
        wds = dis.lower().split()
        if len(wds) == 0 or wds[0] == 'all' or wds[0] == 'none' or wds[0] == 'both':
            return False
        return True
    elif dataset == 'sciq':
        if not dis or dis in set(['a and b', 'b and c', 'a and c']):
            return False
        return True
    elif dataset == 'race':
        return True
    else:
        return True



def gen_t_q_a_d_uncased(args):
    for split in ["train", "dev"]:
        with open(f"/home/wjy/newdisk/project/MCQ/pre-rank/data/{args.dataset}/{split}.json") as data_file:
            all_items = json.load(data_file)
        all_texts = []
        all_questions = []
        all_answers = []
        all_distractors = []
        all_question_ids = []
        for idx, item in enumerate(all_items):
            if not is_valid_dis(item['answer'], args.dataset):
                continue
            for d in item["distractors"]:
                if not is_valid_dis(d, args.dataset):
                    continue
                all_texts.append(item["article"].lower())
                all_questions.append(item["sentence"].lower())
                all_answers.append(item["answer"].lower())
                all_distractors.append(d.lower())
                all_question_ids.append(idx)

        logger.info(f"[{split} dataset] number of questions: {len(all_questions)}")

        data = {
            'texts': all_texts,
            'questions': all_questions,
            'answers': all_answers,
            'distractors': all_distractors,
            'question_ids': all_question_ids,
        }
        with open(f'/home/wjy/newdisk/project/MCQ/pre-rank/data/{args.dataset}_uncased_tqad/{split}_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f)

    with open(f"/home/wjy/newdisk/project/MCQ/pre-rank/data/{args.dataset}/test.json") as data_file:
        all_items = json.load(data_file)
    all_texts = []
    all_questions = []
    all_answers = []
    all_candidates = []
    all_ground_truths = []
    all_answer_ids = []

    for item in all_items:
        if not is_valid_dis(item['answer'], args.dataset):
            continue
        cnt_dis = 0
        for d in item["distractors"]:
            if not is_valid_dis(d, args.dataset):
                continue
            cnt_dis += 1
            all_candidates.append(d.lower())
        if cnt_dis == 0:
            continue
        all_texts.append(item["article"].lower())
        all_questions.append(item["sentence"].lower())
        all_answers.append(item['answer'].lower())
        all_candidates.append(item["answer"].lower())
    all_candidates = sorted(list(set(all_candidates)))

    for item in all_items:
        if not is_valid_dis(item['answer'], args.dataset):
            continue
        ground_truths = []
        for d in item["distractors"]:
            if not is_valid_dis(d, args.dataset):
                continue
            ground_truths.append(all_candidates.index(d.lower()))
        if len(ground_truths) > 0:
            all_ground_truths.append(ground_truths)
            all_answer_ids.append(all_candidates.index(item['answer'].lower()))

    logger.info(f"[test dataset] number of questions: {len(all_questions)}")

    test_data = {
        'texts': all_texts,
        'questions': all_questions,
        'answers': all_answers,
        'candidates': all_candidates,
        'ground_truths': all_ground_truths,
        'answer_ids': all_answer_ids
    }
    with open(f'/home/wjy/newdisk/project/MCQ/pre-rank/data/{args.dataset}_uncased_tqad/test_data.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='race')
    args = parser.parse_args()

    gen_t_q_a_d_uncased(args)



