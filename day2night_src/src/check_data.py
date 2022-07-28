# @Time : 9/9/21 4:27 PM 
# @Author : xiangtao.wang@inceptio.ai
# @File : multiprocess_checkdata.py
# @Software: PyCharm

import json
import os
import argparse
import time

import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import multiprocessing as mp


def worker(q, moment_id, data_path, camera_type, dataversion):
    print("Loading moment to be check")
    error_log = []
    error_moment = set()
    print("Currently processing moment ", moment_id)
    img_dir = os.path.join(data_path, moment_id, 'camera_images', camera_type)
    imgs_name = os.listdir(img_dir)
    l40_count = 0
    g40_count = 0
    invalid_count = 0
    for img_name in imgs_name:
        img_path = os.path.join(img_dir, img_name)
        json_path = img_path.replace('.jpg', '_update_amodel_ctr.json')
        json_path = json_path.replace('camera_images', 'split_json')
        json_path = json_path.replace('raw_data', 'parsed_groundtruth'+'/'+dataversion)
        try:
            frame_objects = json.load(open(json_path, 'r'))
        except:
           # error_log.append(f'The json file is invalid {json_path} \n')
            continue

        if len(frame_objects) == 0:
           # print(f"There is no 2d object in current frame! {json_path}")
            continue

        for frame_object in frame_objects:
            if 0 < frame_object['depth'] < 30:
                l40_count = l40_count + 1
            elif frame_object['depth'] >= 30:
                g40_count = g40_count + 1
            elif frame_object['depth'] == -10:
                invalid_count += 1

    error_log.append(f'{moment_id} has l30_count:{l40_count} g30_count:{g40_count} invalid_count:{invalid_count} \n')

    q.put(error_log)
    return error_log


def listener(q, error_log_path):
    '''listens for messages on the q, writes to file. '''

    with open(error_log_path, 'w') as f:
        while 1:
            m = q.get()
            if m == 'kill':
                break
            if len(m) != 0:
                f.write(''.join(m))
            f.flush()


# luci
if __name__ == '__main__':
    moment_set = set([moment_id.split('\n')[0] for moment_id in open(
        '/mnt/luci-logs/luci-home/xinjing/xiangtao/data_moments/hirain/2021/v3.9.0_night_aug/night_data_v3.9.0_train.txt',
        'r').readlines()])
    root_dir = '/mnt/exterior-perception-cpfs/training_data/training_dataset__from_2020_02_01__v2.01/raw_data'
    error_log_path = '/mnt/luci-home/songyang/day2night_1129/UNIT/src/data_stat_train.log'
    data_channel = 'narrow'
    dataset_version = 'v3.9.0'
    # must use Manager queue here, or will not work
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() - 1)

    # put listener to work first
    watcher = pool.apply_async(listener, (q, error_log_path))

    # fire off workers
    jobs = []
    for moment_id in moment_set:
        job = pool.apply_async(worker, (q, moment_id, root_dir, data_channel, dataset_version))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    # now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

    print('Finished checking data!!')

# local
# if __name__ == '__main__':
#     moment_set = set([moment_id.split('\n')[0] for moment_id in open(
#         '/data/dataset/baidu_tracking_hiran_6night/badiu_tracking_hiran/mon_list/valid_63_night_6.txt',
#         'r').readlines()])
#     root_dir = '/data/dataset/baidu_tracking_hiran_6night/badiu_tracking_hiran/inceptio-track-squece-test/raw_data'
#     error_log_path = '/data/dataset/baidu_tracking_hiran_6night/badiu_tracking_hiran/total_train_error.log'
#     error_moment_path = '/data/dataset/baidu_tracking_hiran_6night/badiu_tracking_hiran/total_train_error_moment.txt'
#     data_channel = 'narrow'
#     dataset_version = 'v3.9.0'
#     # must use Manager queue here, or will not work
#     manager = mp.Manager()
#     q = manager.Queue()
#     pool = mp.Pool(mp.cpu_count() - 1)
#
#     # put listener to work first
#     watcher = pool.apply_async(listener, (q, error_log_path))
#
#     # fire off workers
#     jobs = []
#     for moment_id in moment_set:
#         job = pool.apply_async(worker, (q, moment_id, root_dir, data_channel, dataset_version))
#         jobs.append(job)
#
#     # collect results from the workers through the pool result queue
#     for job in jobs:
#         job.get()
#
#     # now we are done, kill the listener
#     q.put('kill')
#     pool.close()
#     pool.join()

    # print('Finished checking data!!')
