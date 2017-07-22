import tensorflow as tf
import os
import numpy as np
import random
from open_video import open_video
from datetime import datetime
import cv2
import time

# now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
# root_logdir = "tf_logs"
# logdir = "{}/run-{}/".format(root_logdir, now)

SAMPLING_WIDTH = 160
SAMPLING_HEIGHT = 120

TEMPORAL_DEPTH = 16
INPUT_CHANNELS = 3
NUM_CLASSES = 101
INPUT_WIDTH = 112
INPUT_HEIGHT = 112

TESTPATH1 =  './ucfTrainTestlist/testlist01.txt'
TRAINPATH1 = './ucfTrainTestlist/trainlist01.txt'

with open("./ucfTrainTestlist/classInd.txt") as file:
    class_list = file.read().split()[1::2]
    

def one_hot(index):
    encoding = np.zeros(NUM_CLASSES)
    encoding[index] = 1
    return encoding


def create_crops_from_video(filename, num_crops):
    video_array = open_video(filename, SAMPLING_WIDTH, SAMPLING_HEIGHT)
    num_video_frames = video_array.shape[0]
    
    if num_video_frames < TEMPORAL_DEPTH:
        num_repeat_ops = int(TEMPORAL_DEPTH / num_video_frames)
        repeated_video = video_array
        for i in range(num_repeat_ops):
            repeated_video = np.concatenate((repeated_video, video_array), 0)
        video_array = repeated_video
        num_video_frames = video_array.shape[0]
    
    crops = np.zeros((num_crops, TEMPORAL_DEPTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS), dtype=np.uint8)
    
    for i in range(num_crops):
        start_frame = np.random.randint(num_video_frames - TEMPORAL_DEPTH)
        end_frame = start_frame + TEMPORAL_DEPTH
        start_width = np.random.randint(SAMPLING_WIDTH - INPUT_WIDTH)
        end_width = start_width + INPUT_WIDTH
        start_height = np.random.randint(SAMPLING_HEIGHT - INPUT_HEIGHT)
        end_height = start_height + INPUT_HEIGHT
        crops[i, :, :, :, :] = video_array[start_frame:end_frame, start_height:end_height, start_width:end_width, :]
        
        if not crops[i, :, :, :, :].any():
            print("(EE) --- empty crops!!!")
            print(start_frame, end_frame, video_array.shape)
            input()
    return crops


def get_folder_filename_tuples(filepath):
    with open(filepath) as file:
        tuples = file.read().splitlines()
        tuples = list(map(lambda s: tuple(s.split(" ")[0].split("/")), tuples))
    return tuples


train_file_tuples = get_folder_filename_tuples(TRAINPATH1)
random.shuffle(train_file_tuples)
batch_called = 0

def get_next_batch(batch_size):
    global batch_called
    
    batch = np.zeros((batch_size, TEMPORAL_DEPTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS), dtype=np.uint8)
    labels = np.zeros((batch_size, NUM_CLASSES))
    
    start = batch_called * batch_size
    end = start + batch_size
    
    if end <= len(train_file_tuples):
        batch_called = batch_called + 1
        for i in range(batch_size):
            foldername, filename = train_file_tuples[start + i]
            complete_path = './UCF-101/{}/{}'.format(foldername, filename)
            batch[i, :, :, :, :] = create_crops_from_video(complete_path, 1)
            labels[i, :] = one_hot(class_list.index(foldername))
        return batch, labels, False
    else:
        batch_called = 0
        # pad last returned batch with crops from random videos
        for i in range(batch_size):
            if start + i < len(train_file_tuples):
                video_index = start + i
            else:
                video_index = np.random.randint(len(train_file_tuples))
            foldername, filename = train_file_tuples[video_index]
            complete_path = './UCF-101/{}/{}'.format(foldername, filename)
            batch[i, :, :, :, :] = create_crops_from_video(complete_path, 1)
            labels[i, :] = one_hot(class_list.index(foldername))
        random.shuffle(train_file_tuples)
        return batch, labels, True
    
    
test_file_tuples = get_folder_filename_tuples(TESTPATH1)

def get_test_set():
    num_test_videos = len(test_file_tuples)
    testset = np.zeros((num_test_videos, TEMPORAL_DEPTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS), dtype=np.uint8)
    labels = np.zeros((num_test_videos, NUM_CLASSES))
    for i in range(num_test_videos):
        foldername, filename = test_file_tuples[i]
        complete_path = './UCF-101/{}/{}'.format(foldername, filename)
        testset[i] = create_crops_from_video(complete_path, 1)
        labels[i] = one_hot(class_list.index(foldername))
    return testset, labels
    
    
def play_clip(clip):
    for i in range(clip.shape[0]):
        cv2.imshow('clip', clip[i])
        cv2.waitKey(10)
    
    
if __name__ == '__main__':
    print(batch_called)
    b, l, ended = get_next_batch(10)
    print(batch_called)
    b, l, ended = get_next_batch(10)
    print(batch_called)
    b, l, ended = get_next_batch(10)
    print(batch_called)
    b, l, ended = get_next_batch(10)
    print(batch_called)
    # before = time.time()
    # batch, labels = get_test_set()
    # duration = time.time() - before
    # print("Creating Test set took:", duration)
    # for i in range(50):
    #     play_clip(batch[i])
    # cv2.destroyAllWindows()