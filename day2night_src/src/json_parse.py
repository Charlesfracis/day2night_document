#!/usr/bin/env python
# coding=utf-8
#@Information:
#@Author       : liyue
#@Date         : 2021-07-07 19:47:20
#@FilePath     : /my_test/json_parse.py
#@Email        : yue.li@inceptio.ai
#@LastEditTime : 2021-10-18 14:56:37


from posixpath import dirname, pardir
from typing import Text
import pycocotools.coco as coco
import os
from collections import defaultdict, Counter
import json
from tqdm import tqdm
import cv2
import numpy as np
# from absmv_metric_utils import ABSMVeval
import PIL.Image as pil
import matplotlib.pyplot as plt 

class Json:
    def __init__(self, img_dir, ann_file):
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.my_coco = coco.COCO(self.ann_file)
        self.images = self.my_coco.getImgIds()
        self.num_samples = len(self.images)


    def get_moments(self):
        moment_list = []
        video_lists = self.my_coco.dataset['videos']
        for video in video_lists:
            moment_name = video['file_name'][: video['file_name'].find('_')]
            moment_list.append(moment_name)

        moment_new = set(moment_list)
        return moment_new


    def statistic_movement_state(self):
        movement_state_list = [] # 存储所有目标的movement_state
        movement_state_dict_list = defaultdict(list)  # 存储每张图片中的movement_state为1的目标框, key is image_id, value is list of bbox
        movement_state_and_other_attributes = []
        anns = self.my_coco.dataset['annotations']
        print(anns[0])
        # for ann in anns:
        #     movement_state = ann['movement_state']
        #     velocity = ann['abs_velocity']
        #     bbox = ann['bbox']
        #     bbox[2] += bbox[0]
        #     bbox[3] += bbox[1]
        #     movement_state_list.append(movement_state)
        #     movement_state_and_other_attributes.append((movement_state, bbox[2] * bbox[3] / (1920 * 1080), velocity[0]))
        #     if movement_state == 1:
        #         movement_state_dict_list[ann['image_id']].append(bbox)
        
        # statistic = Counter(movement_state_list)
        # print(statistic)
        # # for element in movement_state_and_other_attributes:
        # #     if element[0] == 2 and element[2] > -100.0:
        # #         print(element)

        # for img_id in movement_state_dict_list.keys():
        #     bbox_list = movement_state_dict_list[img_id]
        #     self.show_bbox_on_image(img_id, bbox_list)


    def show_bbox_on_image(self, img_id, bbox_list1 = None, bbox_list2 = None):
        img_info = self.my_coco.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']
        # moment_name = file_name[: file_name.find('_')]
        # print(moment_name)
        img_path = os.path.join(self.img_dir, file_name)
        img = cv2.imread(img_path)
        if bbox_list1 is not None:
            for bbox in bbox_list1:
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 1) # 红色框
        if bbox_list2 is not None:
            for bbox in bbox_list2:
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,255), 1) # 黄色框
        cv2.imshow('img', img)
        cv2.waitKey(0)

    
    def show_image(self, folder_name):
        for img_id in range(self.num_samples):
            img_info = self.my_coco.loadImgs(ids=[img_id + 1])[0]
            file_name = img_info['file_name']
            img_path = os.path.join(self.img_dir, file_name) # original image 
            # depth_map_path = img_path.replace('images', 'depth_map_v{}'.format(depth_map_version)).replace('.jpg', '.png')
            # if not os.path.exists(depth_map_path):
            #     depth_map_path = depth_map_path.replace('raw_data', 'depth_map_v{}'.format(depth_map_version))
            # if os.path.exists(img_path) and os.path.exists(depth_map_path):

            # fs_map_path_human = img_path.replace('images', 'drivable_region_vhuman').replace('.jpg', '.png') # 
            # fs_map_path_auto = img_path.replace('images', 'drivable_region_vauto')
            fs_map_path = os.path.join(folder_name, file_name).replace('images', 'drivable_region').replace('.jpg', '.png')
            img = pil.open(img_path).convert('RGB')
            fs_seg_img = pil.open(fs_map_path).convert('RGB')
            # fs_seg_img_human = pil.open(fs_map_path_human).convert('RGB')
            combine_img = pil.blend(img, fs_seg_img, 0.5)
            fs_map_path_bak = os.path.join(os.path.dirname(fs_map_path), os.path.basename(fs_map_path).split('.')[0] + '_bak.jpg')
            # print(fs_map_path_bak)
            combine_img.save(fs_map_path_bak)
            # plt.figure(figsize=(50, 50))
            # plt.ion()
            # plt.subplot(311)
            # plt.imshow(img)
            # plt.subplot(312)
            # plt.imshow(fs_seg_img)
            # plt.subplot(313)
            # plt.imshow(combine_img)
            # plt.pause(3)
            # plt.close()
            # size = (192 * 4, 108 * 4)
            # alpha = 0.5
            # beta = 1 - alpha
            # gamma = 0
            # img = cv2.resize(cv2.imread(img_path), size)
            # fs_seg_img_human = cv2.resize(cv2.imread(fs_map_path_human), size) * 255
            # fs_seg_img_auto = cv2.resize(cv2.imread(fs_map_path_auto), size)
            # combine_img = cv2.addWeighted(img, alpha, fs_map_path_auto, beta, gamma)
            # # depth_img_gray = np.expand_dims(depth_img, -1)
            # # depth_img_gray = np.clip(depth_img_gray / 65535 * 2 * 255, 0, 255).astype(np.uint8)
            # # depth_img_color = cv2.applyColorMap(depth_img_gray, cv2.COLORMAP_RAINBOW)
            # # print(np.sum((depth_img > 0)), np.sum((depth_img)))
            # cv2.imshow('img', combine_img)
            # cv2.imshow('fs_seg_img_human', fs_seg_img_human)
            # cv2.imshow('fs_seg_img_auto', fs_seg_img_auto)
            # cv2.waitKey(0)


    def get_new_moments(self, moment_txt):
        moment_new = self.get_moments()
        # if moment_txt is not None:
        #     with open(moment_txt, 'w') as fm:
        #         for moment_id in moment_new:
        #             fm.write(moment_id + '\n')
                # moment_exclude = [line.strip() for line in fm.readlines()]
                # moment_new = moment_new - set(moment_exclude)
        
        moment_vaild = []
        for moment_name in tqdm(moment_new):
            print(moment_name)
            moment_folder = os.path.join(self.img_dir, moment_name)
        #     is_valid = False
        #     if os.path.exists(moment_folder):
        #         for lidar_name in os.listdir(moment_folder):
        #             lidar_folder = os.path.join(moment_folder, lidar_name)
        #             depth_folder = os.path.join(lidar_folder, 'depth_map_v1.0.0/obstacle')
        #             if os.path.exists(depth_folder) and len(os.listdir(depth_folder)) > 0:
        #                 is_valid = True
        #                 break
        #     else:
        #         print(moment_folder)
        #         is_valid = False
            
        #     if is_valid is True:
        #         moment_vaild.append(moment_name)
            
        # with open(moment_txt, 'w') as fm:
        #     for moment_name in moment_vaild:
        #         fm.write(moment_name + '\n')
        # print(len(moment_new), len(moment_vaild))

        
    def download_data_by_moment(self, moment_txt=None):
        moment_new = self.get_moments()
   
        moment_vaild = []
        for moment_name in tqdm(moment_new):
            moment_folder = os.path.join(self.img_dir, moment_name)
            if not os.path.exists(moment_folder):
                print(moment_name)
            else:
                moment_vaild.append(moment_name)
        print(len(moment_new), len(moment_vaild))

        # scp_command = 'scp -r '
        # for moment_name in moment_vaild:
        #     scp_command += moment_name + ' ' 
        
        # scp_command += 'yue@10.8.27.234:/home/yue/my_code/centertrack/data/baidu_tracking_hiran/inceptio-track-squece-test/raw_data'
        # os.system(scp_command)


    def absmv_analyse(self, results_dir, show_results=False):
        absmv_coco_dets = self.my_coco.loadRes(results_dir + '/results_absmv.json')
        absmv_coco_eval = ABSMVeval(self.my_coco, absmv_coco_dets, 'bbox')
        fpr, tpr, thresholds = absmv_coco_eval.evaluate()
        results_absmv_file = results_dir + '/results_absmv_bak.txt'
        with open(results_absmv_file, 'w') as fabsmv:
            for i in range(len(thresholds)):
                fabsmv.write(str(thresholds[i]) + ' ' + 'tpr: ' + str(tpr[i]) + ' ' + '1 - fpr:' + str(1 - fpr[i]) + '\n')
        
        movement_state_dict_list = defaultdict(list)  # 存储每张图片中的movement_state为1的目标框, key is image_id, value is list of bbox
        anns = self.my_coco.dataset['annotations']
        for ann in anns:
            movement_state = ann['movement_state']
            bbox = ann['bbox']
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            if movement_state == 1:
                movement_state_dict_list[ann['image_id']].append(bbox)
        # print(movement_state_dict_list)
        if show_results:
            image_id_to_bbox = absmv_coco_eval.image_id_to_bbox
            for img_id in image_id_to_bbox.keys():
                bbox_list = image_id_to_bbox[img_id]
                self.show_bbox_on_image(img_id, bbox_list)
            # for img_id in movement_state_dict_list.keys():
            #     # print(img_id)
            #     gt_bbox_list = movement_state_dict_list[img_id]
            #     if img_id in image_id_to_bbox.keys():
            #         pred_bbox_list = image_id_to_bbox[img_id]
            #         self.show_bbox_on_image(img_id, gt_bbox_list, pred_bbox_list)
            #     else:
            #         self.show_bbox_on_image(img_id, gt_bbox_list, None)
            #         print('img_id {} does not exists!'.format(img_id))


    def show_detection_results(self, results_dir):
        image_id_to_bbox = defaultdict(list)
        det_results = json.load(open(results_dir + '/results_2d.json', 'r'))
        for result in det_results:
            if result['score'] > 0.3:
                image_id_to_bbox[result['image_id']].append(result['bbox'])
        
        for img_id in image_id_to_bbox.keys():
            bbox_list = image_id_to_bbox[img_id]
            self.show_bbox_on_image(img_id, bbox_list)


    def get_mono_depth2_train_or_val_filename(self, data_filename):
        moment_new = self.get_moments()
        with open(data_filename, 'w') as fd:
            for moment_name in tqdm(moment_new):
                print(moment_name)
                moment_folder = os.path.join(self.img_dir, moment_name)
                if os.path.exists(moment_folder):
                    lidar_name_list = os.listdir(moment_folder)
                    lidar_name_list.sort()
                    for i in range(1, len(lidar_name_list) - 1):
                        fd.write(moment_name + ' ' + lidar_name_list[i - 1] + ' ' + lidar_name_list[i] + ' ' + lidar_name_list[i + 1] + ' l' + '\n')                    
                else:
                    continue

                
if __name__ == '__main__':
    root_dir = '/mnt/liyue/data/baidu_tracking_hiran'
    img_dir = os.path.join(root_dir, 'inceptio-track-squece-test/raw_data')
    ann_json_file = 'scenario/day/day.json'
    ann_path = os.path.join(root_dir, ann_json_file)
    # ann_path = os.path.join(root_dir, 'annotations', ann_json_file)
    # img_dir = '/mnt/luci-home/xinjing/liyue/data/raw_data'
    # ann_path = '/mnt/luci-home/xinjing/liyue/data/annotations/tracking_val_hiran_v0.1.0_freespace.json'
    moment_txt = '/mnt/luci-home/xinjing/liyue/catspaw/VLS128_moments_train_0.2.0.txt'
    # ann_path = '/mnt/3dvision-cpfs/ziwei/inceptio_discrete/annotations_new/tracking_train_hiran_v3.1.0_all_feature_update_amodel_ctr_with_vel_smooth.json'
    result_dir = '/home/yue/my_code/centertrack/exp'
    result_dir = os.path.join(result_dir, 'ddd,absmv,velocity/hiran_v3.1.0_512*960_lfrf/results_baidu_tracking_v3.1.0_all_feature_update_amodel_ctr_with_vel_smooth_rfgt_left-forward_right-forward')
    json_parse = Json(img_dir, ann_path)
    # json_parse.statistic_movement_state()
    # json_parse.absmv_analyse(result_dir)
    # json_parse.show_detection_results(result_dir)
    json_parse.download_data_by_moment()
    # folder_name = '/home/yue/my_code/centertrack/exp/fs_seg/hiran_v0.1.0_human_fs_seg_352*672/results_baidu_tracking_v0.1.0_freespace/free_space_seg_result/result_model_last_human_auto'
    #$ json_parse.show_image(folder_name)
    # data_filename = '/home/yue/my_code/monodepth2/splits/eigen_zhou/hirain_val.txt'
    # json_parse.get_mono_depth2_train_or_val_filename(data_filename)
