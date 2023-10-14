'''
This script handling the training process.
'''
import warnings

import pytrec_eval

warnings.filterwarnings("ignore")

import os
import argparse
import math
import time
import logging
import json
import copy
import random
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, BertTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_data_loader, optimizer, scheduler, epoch_i, args):
    model.train()
    total_tr_loss = 0.0
    total_train_batch = 0
    total_acc = 0.0
    start = time.time()

    for step, batch in enumerate(tqdm(train_data_loader, desc='  -(Training)', leave=False)):
        # forward
        tr_loss, tr_acc = model(**batch)

        # backward
        tr_loss.backward()

        # record
        total_acc += tr_acc
        total_tr_loss += tr_loss.item()
        total_train_batch += 1

        # update parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    logger.info('[Epoch{epoch: d}] - (Train) loss ={train_loss: 8.5f}, acc ={acc: 3.2f} %, ' \
                'elapse ={elapse: 3.2f} min'.format(epoch=epoch_i,
                                                    train_loss=total_tr_loss / total_train_batch,
                                                    acc=100 * total_acc / total_train_batch,
                                                    elapse=(time.time() - start) / 60))


def dev_epoch(model, dev_data_loader, epoch_i, args):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_dev_loss = []
    total_dev_acc = []
    start = time.time()

    with torch.no_grad():
        for batch in tqdm(dev_data_loader, desc='  -(Dev)', leave=False):
            # forward
            dev_loss, dev_acc = model(**batch)
            # record
            total_dev_loss += [dev_loss.item()]
            total_dev_acc += [dev_acc]

    dev_loss = np.mean(total_dev_loss)
    dev_acc = 100 * np.mean(total_dev_acc)

    logger.info('[Epoch{epoch: d}] - (Dev  ) loss ={loss: 8.5f}, acc ={acc: 3.2f} %, ' \
                'elapse ={elapse: 3.2f} min'.format(epoch=epoch_i,
                                                    loss=dev_loss,
                                                    acc=dev_acc,
                                                    elapse=(time.time() - start) / 60))
    return dev_loss, dev_acc



def obtain_test_embedding(model, pca_whitening, test_data_loader, args):
    model.eval()
    question_embeddings = []
    candidate_embeddings = []

    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc='[encoding test QA pairs]', leave=False):
            batch['inference'] = True
            # forward
            question_embedding, answer_embedding = model(**batch)
            question_embeddings.append(question_embedding.cpu().numpy())
            candidate_embeddings.append(answer_embedding.cpu().numpy())

    test_ground_truth = test_data_loader.dataset.ground_truths
    test_answer_ids = test_data_loader.dataset.answer_ids
    question_embeddings = np.concatenate(question_embeddings, 0)[:len(test_ground_truth)]
    candidate_embeddings = np.concatenate(candidate_embeddings, 0)

    if pca_whitening is not None:
        question_embeddings = pca_whitening.transform(question_embeddings)
        candidate_embeddings = pca_whitening.transform(candidate_embeddings)
    return question_embeddings, candidate_embeddings, test_ground_truth, test_answer_ids


def test(args, original_question_embeddings, original_candidate_embeddings, test_ground_truth, test_answer_ids,
         test_dataset, used_dimension=-1):
    if used_dimension != -1:
        question_embeddings = original_question_embeddings[:, :used_dimension]
        candidate_embeddings = original_candidate_embeddings[:, :used_dimension]
    else:
        question_embeddings = original_question_embeddings
        candidate_embeddings = original_candidate_embeddings

    question_embeddings /= np.linalg.norm(question_embeddings, axis=1, keepdims=True)
    candidate_embeddings /= np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    predict_logits = np.matmul(question_embeddings, candidate_embeddings.T)

    rankat = [1, 5, 10]
    r_counts = defaultdict(float)
    for rank in rankat:
        r_counts[rank] = 0
    r_rank = 0

    eval_dict = {}
    pred_results = {"pred_results": []}

    for num in tqdm(range(len(test_ground_truth))):
        pred = np.argsort(-predict_logits[num]).tolist()
        pred_result = {"T": test_dataset.texts[num],
                       "Q": test_dataset.questions[num],
                       "A": test_dataset.answers[num],
                       "GT": [test_dataset.distractors[i] for i in test_ground_truth[num]],
                       "Pos": [pred.index(i) for i in test_ground_truth[num]],
                       "Rank": [test_dataset.distractors[i] for i in pred[:20]]}
        pred_results["pred_results"].append(pred_result)
        if args.rm_ans:
            pred.remove(test_answer_ids[num])
        for rank in rankat:
            r_count = 0
            for gt in test_ground_truth[num]:
                if gt in pred[:rank]:
                    r_count += 1
            r_count /= len(test_ground_truth[num])
            r_counts[rank] += r_count

        for idx, p in enumerate(pred):
            if p in test_ground_truth[num]:
                r_rank += 1 / (idx + 1)
                break

        qrel_dict = {}
        run_dict = {}
        for idx in range(len(predict_logits[num])):
            qrel_dict[str(idx)] = 0 if idx not in test_ground_truth[num] else 1
            run_dict[str(idx)] = float(predict_logits[num][idx])
        if args.rm_ans:
            qrel_dict.pop(str(test_answer_ids[num]))
            run_dict.pop(str(test_answer_ids[num]))

        qrel = {'q1': qrel_dict}
        run = {'q1': run_dict}
        evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'P_1', 'P_3', 'map_cut_10', 'ndcg_cut_10', 'recall_10', 'recall_30',
                                                          'recall_50', 'recall_100', 'recall_200'})
        eval_result = evaluator.evaluate(run)
        for key in eval_result['q1']:
            eval_dict[key] = eval_dict.get(key, 0) + eval_result['q1'][key]

    for key in eval_dict:
        eval_dict[key] = np.round(eval_dict[key] / len(test_ground_truth), 4)

    mrr = np.round(r_rank / len(test_ground_truth), 4)
    r_at_k = [np.round(v / len(test_ground_truth), 4) for k, v in sorted(r_counts.items(), key=lambda item: item[0])]

    test_mrr, test_r1, test_r5, test_r10 = mrr * 100, r_at_k[0] * 100, r_at_k[1] * 100, r_at_k[2] * 100

    logger.info('[       ] - (Test ) P_1 ={P_1: 3.2f} %, P_3 ={P_3: 3.2f} %,' \
                ' MAP_10 ={map_cut_10: 3.2f} %, NDCG_10 ={ndcg_cut_10: 3.2f} %,' \
                ' mrr ={mrr: 3.2f} % R_10 ={recall_10: 3.2f} %, R_30 ={recall_30: 3.2f} %,'
                'R_50 ={recall_50: 3.2f} %, R_100 ={recall_100: 3.2f} %, R_200 ={recall_200: 3.2f} %, '.format(
                                             P_1=eval_dict['P_1'] * 100,
                                             P_3=eval_dict['P_3'] * 100,
                                             map_cut_10=eval_dict['map_cut_10'] * 100,
                                             ndcg_cut_10=eval_dict['ndcg_cut_10'] * 100,
                                             mrr=mrr * 100,
                                             recall_10=eval_dict['recall_10'] * 100,
                                             recall_30=eval_dict['recall_30'] * 100,
                                             recall_50=eval_dict['recall_50'] * 100,
                                             recall_100=eval_dict['recall_100'] * 100,
                                             recall_200=eval_dict['recall_200'] * 100,
    ))

    result_dict = { "p1": eval_dict['P_1'] * 100, "p3": eval_dict['P_3'] * 100,
                   "map10": eval_dict['map_cut_10'] * 100, "ndcg10": eval_dict['ndcg_cut_10'] * 100,
                   "mrr": mrr * 100, "r10": eval_dict['recall_10'] * 100, "r30": eval_dict['recall_30'] * 100,
                    "r50": eval_dict['recall_50'] * 100, "r100": eval_dict['recall_100'] * 100, "r200": eval_dict['recall_200'] * 100,}
    json.dump(result_dict, open(os.path.join(args.save_model, "eval_result.json"), "w"))
    json.dump(pred_results, open(os.path.join(args.save_model, "pred_results.json"), "w"), indent=4)
    np.save(os.path.join(args.save_model, "pred_logits.npy"), predict_logits)



def run(model, train_data_loader, dev_data_loader, test_data_loader, args, results):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * args.num_training_steps),
        num_training_steps=args.num_training_steps
    )

    if args.do_train:
        best_metrics = 0
        best_epoch = 0
        for epoch_i in range(args.epoch):
            logger.info('[Epoch {}]'.format(epoch_i))

            train_epoch(model, train_data_loader, optimizer, scheduler, epoch_i, args)

            dev_loss, dev_acc = dev_epoch(model, dev_data_loader, epoch_i, args)

            current_metrics = dev_acc

            if args.save_model:
                model_name = args.save_model + f'/best.pt'
                if not os.path.exists(args.save_model):
                    os.makedirs(args.save_model)
                if current_metrics > best_metrics:
                    best_epoch = epoch_i
                    best_metrics = current_metrics
                    model_state_dict = model.state_dict()
                    torch.save(model_state_dict, model_name)
                    logger.info('  - [Info] The checkpoint file has been updated.')
        logger.info(f'Got best test performance on epoch{best_epoch}')

    if args.do_test:
        logger.info(f'Conduct evaluation on test dataset')
        model.load_state_dict(torch.load(args.save_model + f'/best.pt'))
        model.to(args.device)
        logger.info('reload best checkpoint')
        # no-whitening evaluation
        pca_whitening = None
        question_embeddings, candidate_embeddings, test_ground_truth, test_answer_ids = obtain_test_embedding(model,
                                                                                                              pca_whitening,
                                                                                                              test_data_loader,
                                                                                                              args)
        test(args, copy.deepcopy(question_embeddings),
                                                    copy.deepcopy(candidate_embeddings),
                                                    test_ground_truth,
                                                    test_answer_ids,
                                                    test_data_loader.dataset,
                                                    -1)

    if args.rm_saved_model:
        import shutil
        shutil.rmtree(args.save_model)

    return results


def prepare_dataloaders(args):
    from utils_data import LTRRaceDataset as Dataset
    # initialize datasets
    train_dataset = Dataset(args, split='train')
    dev_dataset = Dataset(args, split='dev')
    test_dataset = Dataset(args, split=args.test_dataset)

    args.num_train_instances = train_dataset.__len__()
    args.num_training_steps = int(args.num_train_instances / args.batch_size * args.epoch)
    logger.info(f"train data size: {train_dataset.__len__()}")
    logger.info(f"dev data size: {dev_dataset.__len__()}")
    logger.info(
        f"test question size: {len(test_dataset.ground_truths)} candidate size: {len(test_dataset.distractor_input_ids)}")

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn)

    dev_data_loader = torch.utils.data.DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=dev_dataset.collate_fn)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=test_dataset.collate_fn)

    return train_data_loader, dev_data_loader, test_data_loader


def prepare_model(args):
    from models.dual_encoder import RankRaceModel
    model = RankRaceModel(args)
    if args.load_model != '':
        # pretrained_dict = torch.load(args.load_model)['model']
        pretrained_dict = torch.load(args.load_model)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict)
        logger.info('load model successfully!')
    model.to(args.device)
    return model


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--max_text_question_answer_len', type=int, default=256)
    parser.add_argument('--max_distractor_len', type=int, default=64)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument('--test_dataset', type=str, default='test')

    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument('--model_type', type=str)
    parser.add_argument('--encoder_type', type=str)
    parser.add_argument('--plm_path', type=str)
    parser.add_argument('--pooler_type', type=str)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--save_model', type=str, default='')
    parser.add_argument("--rm_saved_model", action="store_true")
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--temperature2', type=float)
    parser.add_argument('--temperature3', type=float)
    parser.add_argument('--ns', action="store_true")
    parser.add_argument('--rm_ans', action="store_true")
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--sim_type', type=str, default="cos")
    parser.add_argument('--kd', action='store_true')
    parser.add_argument('--kd_w', type=float, default=1)
    parser.add_argument('--hn', action='store_true')
    parser.add_argument('--hn_ss', type=float, default=0)
    parser.add_argument('--hn_tt', type=float, default=0)
    parser.add_argument('--hn_st', type=float, default=0)
    parser.add_argument('--hn_w', type=float, default=1)
    parser.add_argument('--sl_s', type=float, default=0)
    parser.add_argument('--sl_t', type=float, default=0)
    parser.add_argument('--ce_w', type=float, default=1)



    args = parser.parse_args()

    logger.info(args)

    # set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set seed
    set_seed(42)

    # record results
    results = {
        'no-whitening': defaultdict(list),
        'train-whitening': defaultdict(list),
    }

    # loading dataset
    train_data_loader, dev_data_loader, test_data_loader = prepare_dataloaders(args)

    # preparing model
    model = prepare_model(args)

    # running
    results = run(model, train_data_loader, dev_data_loader, test_data_loader, args, results)


if __name__ == '__main__':
    main()
