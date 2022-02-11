''' 
_*_ coding: utf-8 _*_
Date: 2021/9/4
Author: 
Intent:
'''

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

import torch
from lajs_model import FirstModel
from lajs_datasets import SegmentSelectionDataset
from tqdm import tqdm
import os, json

class Predict(object):
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = SegmentSelectionDataset(config)
        self.predict_data = json.load(open(self.config['predict_file'], 'r', encoding='utf8'))
        self.model = FirstModel(config)
        state_dict = torch.load(os.path.join(config['pretrain_model_dir'], 'pytorch_model.bin'),
                                map_location=self.device)
        print('------', list(self.model.state_dict().keys())[:6], '-------')
        print('------', list(state_dict.keys())[:6], '-------')
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def predict(self):
        all_preds = {}
        with torch.no_grad():
            for i, s in enumerate(tqdm(self.predict_data)):
                qidx, q, q_crime, docs = s['qidx'], s['q'], s['crime'], s['docs']
                preds, dids = [], []
                for d in docs:
                    didx = d['didx']
                    batch = self.dataset.get_predict_samples(q, q_crime, d)
                    batch = _move_to_device(batch, self.device)
                    num_q, num_d = batch['num_q'], batch['num_d']
                    score = self.model(batch).squeeze()
                    crime_score, chunk_score = score[0], score[1:]
                    # print(crime_score, chunk_score)
                    chunk_score = chunk_score.reshape(num_q, num_d)
                    chunk_score = torch.sum(torch.max(chunk_score, dim=-1).values) / num_q
                    # print(chunk_score)
                    merge_score = (0.2 * crime_score + 0.8 * chunk_score).item()
                    print(crime_score.item(), chunk_score.item())
                    preds.append(merge_score)
                    dids.append(didx)
                sorted_r = sorted(list(zip(dids, preds)), key=lambda x: x[1], reverse=True)
                pred_ids = [x[0] for x in sorted_r]
                all_preds[qidx] = pred_ids
        # print(all_preds)
        return all_preds

def _move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

if __name__ == "__main__":
    pass