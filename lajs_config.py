''' 
_*_ coding: utf-8 _*_
Date: 2021/8/23
Author: 
Intent:
'''

LajsConfig = {
              "pretrain_model_dir": "./model_rbt3_seg2",  # 预测时加载训练好的模型；训练时加载预训练模型
              "model_dir": "/kaggle/working/model/",
              "init_checkpoint": "",

              "do_train": True,
              "warm_ratio": 0.1,
              "max_len": 512,
              "chunk_size": 254,
              "random_k": 4,
              "train_file": "./data/large_train_segment_selection.json",
              "dev_file": "./data/small_dev.json",
              "train_epoch": 500,
              "train_batch_size": 1,
              "weight_decay": 0.01,
              "learning_rate": 3e-5,
              "accum_steps": 4,
              "print_steps": 20,
              "eval_steps": 500,
              "save_ckpt_steps": 40000000,
              "max_grad": 1,

              "predict_file": "./data.json",
              "predict_batch_size": 1
              }

if __name__ == "__main__":
    pass