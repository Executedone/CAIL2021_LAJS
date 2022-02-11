import argparse
import json
import os, time, logging
from lajs_utils import select_relavence
from lajs_predict_rbt3_seg import Predict
from lajs_config import LajsConfig

parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('--input', type=str, default='./data/small/', help='input path of the dataset directory.')
parser.add_argument('--output', type=str, default='./', help='output path of the prediction file.')

args = parser.parse_args()
input_path = args.input
input_query_path = os.path.join(input_path, 'query.json')
input_candidate_path = os.path.join(input_path, 'candidates')
output_path = args.output
stw_path = os.path.join(os.path.dirname(__file__), 'stopword.txt')
new_data_path = os.path.join(os.path.dirname(__file__), 'data.json')
LajsConfig['predict_file'] = new_data_path

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S')
    print('begin...')
    if not os.path.exists(new_data_path):
        select_relavence(input_query_path, input_candidate_path, stw_path, new_data_path, sent_group=6, select=1)
    time.sleep(1)
    print('temp data converting finished...')

    lp = Predict(LajsConfig)
    print('prediction starting...')
    result = lp.predict()
    json.dump(result, open(os.path.join(output_path, 'prediction.json'), "w", encoding="utf8"), indent=2,
              ensure_ascii=False, sort_keys=True)
    print('output done.')


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    main()

    # import time, random
    # import pyautogui
    #
    # time.sleep(10)  # 延迟8秒
    #
    # y_list = [500, 520, 540, 560, 580]
    # x_list = [800,820,840,860,880,900,920,940,960,980,1000]
    # while True:
    #     time.sleep(30)
    #     x = random.sample(x_list, k=1)[0]
    #     y = random.sample(y_list, k=1)[0]
    #     pyautogui.moveTo(x, y, duration=0.3)
    #     pyautogui.click()
    pass
