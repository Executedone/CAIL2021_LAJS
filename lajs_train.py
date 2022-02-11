''' 
_*_ coding: utf-8 _*_
Date: 2021/3/13
Author: 
Intent:
'''

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

import torch
from torch.utils.data import DataLoader
from lajs_model import FirstModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import MSELoss
from lajs_datasets import SegmentSelectionDataset, select_collate
from lajs_utils import ndcg
from tensorboardX import SummaryWriter
from tqdm import tqdm
import logging, os, random, json
import numpy as np
import gc, os

logger = logging.getLogger(__name__)


class SegmentMatch(object):
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FirstModel(config)
        self.model.to(self.device)

    def set_train_parameters(self):
        self.train_set = SegmentSelectionDataset(self.config, self.config['train_file'])
        self.dev_data = json.load(open(self.config['dev_file'], 'r', encoding='utf8'))
        self.num_training_steps = self.config['train_epoch'] * (len(self.train_set) // self.config['train_batch_size'])
        self.num_warm_steps = self.config['warm_ratio'] * self.num_training_steps

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_params = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.config['weight_decay']},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_params, lr=self.config['learning_rate'])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=self.num_warm_steps,
                                                         num_training_steps=self.num_training_steps)
        self.optimizer.zero_grad()
        self.Loss = MSELoss()

    def train(self):
        self.set_train_parameters()

        epochs = [i for i in range(self.config['train_epoch'])]
        global_step = 0
        start_epoch = 0
        best_auc = 0.0
        writer = SummaryWriter(self.config['model_dir'])

        if self.config['init_checkpoint']:
            logger.info(f'loading init_checkpoint from `{self.config["init_checkpoint"]}`')
            ckpt_dict = torch.load(self.config['init_checkpoint'], map_location=self.device)
            self.model.load_state_dict(ckpt_dict['model'])
            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            self.scheduler.load_state_dict(ckpt_dict['schedular'])
            global_step = ckpt_dict['global_step']
            start_epoch = ckpt_dict['epoch']

            del ckpt_dict
            gc.collect()

        torch.cuda.empty_cache()
        self.model.train()

        for epoch in tqdm(epochs[start_epoch:]):
            train_loader = DataLoader(self.train_set, batch_size=self.config['train_batch_size'], shuffle=True,
                                      num_workers=2, collate_fn=select_collate)
            train_batch = 0.0
            train_loss = 0.0
            for batch in tqdm(train_loader):
                global_step += 1
                batch = _move_to_device(batch, self.device)
                scores = self.model(batch)
                labels = batch['rel_labels'].to(self.device)
                loss = self.Loss(scores, labels)

                if self.config['accum_steps'] > 1:
                    loss = loss / self.config['accum_steps']

                train_loss += loss.item() * self.config['accum_steps']
                train_batch += 1

                if global_step % self.config['print_steps'] == 0:
                    logger.info(f'logits: {scores}, labels: {labels}...')
                    logger.info(f'current loss is {round(train_loss/train_batch, 6)} '
                                f'at {global_step} step on {epoch} epoch...')
                    writer.add_scalar('train_loss', train_loss/train_batch, global_step)
                    train_batch = 0.0
                    train_loss = 0.0

                if global_step > 500 and global_step % self.config['save_ckpt_steps'] == 0:
                    logger.info(f'saving ckpt at {global_step} step on {epoch} epoch...')
                    ckpt_dict = {'model': self.model.state_dict(),
                                 'optimizer': self.optimizer.state_dict(),
                                 'schedular': self.scheduler.state_dict(),
                                 'global_step': global_step,
                                 'epoch': epoch}
                    torch.save(ckpt_dict, os.path.join(self.config['model_dir'], f'ckpt-temp.bin'))

                if global_step % self.config['eval_steps'] == 0:
                    acc_avg = self.evaluate()
                    writer.add_scalar('acc_avg', acc_avg, global_step)
                    logger.info(f'acc_avg: {acc_avg} at global step {global_step} in epoch {epoch}...')
                    if acc_avg > best_auc:
                        logger.info(f'from {best_auc} -> {acc_avg}')
                        logger.info('saving models...')
                        torch.save(self.model.state_dict(), os.path.join(self.config['model_dir'], 'best_model.pt'))
                        best_auc = acc_avg

                loss.backward()

                if global_step % self.config['accum_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad'])
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

    def evaluate(self):
        self.model.eval()
        all_preds = {}
        with torch.no_grad():
            for i, s in enumerate(tqdm(self.dev_data)):
                qidx, q, q_crime, docs = s['qidx'], s['q'], s['crime'], s['docs']
                preds, dids = [], []
                for d in docs:
                    didx = d['didx']
                    batch = self.train_set.get_predict_samples(q, q_crime, d)
                    batch = _move_to_device(batch, self.device)
                    num_q, num_d = batch['num_q'], batch['num_d']
                    score = self.model(batch).squeeze()
                    crime_score, chunk_score = score[0], score[1:]
                    # print(crime_score, chunk_score)
                    chunk_score = chunk_score.reshape(num_q, num_d)
                    chunk_score = torch.sum(torch.max(chunk_score, dim=-1).values) / num_q
                    # print(chunk_score)
                    merge_score = (0.2 * crime_score + 0.8 * chunk_score).item()
                    preds.append(merge_score)
                    dids.append(didx)
                sorted_r = sorted(list(zip(dids, preds)), key=lambda x: x[1], reverse=True)
                pred_ids = [x[0] for x in sorted_r]
                all_preds[qidx] = pred_ids[:30]

        all_labels = {}
        for s in self.dev_data:
            all_labels[s['qidx']] = s['labels']

        ndcg_30 = self.cal_ndcg(all_preds, all_labels)

        torch.cuda.empty_cache()
        self.model.train()
        return ndcg_30

    def cal_ndcg(self, all_preds, all_labels):
        ndcgs = []
        for qidx, pred_ids in all_preds.items():
            did2rel = all_labels[qidx]
            ranks = [did2rel[idx] if idx in did2rel else 0 for idx in pred_ids]
            ndcgs.append(ndcg(ranks, 30))
            print(f'********** qidx: {qidx} **********')
            print(f'top30 pred_ids: {pred_ids}')
            print(f'ranks: {ranks}')
        print(ndcgs)
        return sum(ndcgs) / len(ndcgs)


def _move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinidtic = True
    random.seed(seed)
    np.random.seed(seed)


def main():
    from lajs_config import LajsConfig
    seed = 2022
    setup_seed(seed)
    match_model = SegmentMatch(LajsConfig)
    if LajsConfig['do_train']:
        logger.info('-----------start training-----------')
        match_model.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S')
    main()
    pass