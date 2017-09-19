from abc import ABC, abstractmethod
import numpy as np
import random
from videotools import open_video, one_hot, enough_motion_present, permute_clip
import csv
import os


class GenericDataProvider(ABC):
    TEMPORAL_DEPTH = 16
    INPUT_CHANNELS = 3
    INPUT_WIDTH = 112
    INPUT_HEIGHT = 112

    def __init__(self, batch_size, tov_pretraining, debug_mode, current_split):
        """
        videofile_label_tuples... list of self.num_splits lists containing tuples
            of format (path_to_filename, onehot_encoded_label)
        """
        self.batch_size = batch_size
        self.debug_mode = debug_mode
        self.tov_pretraining = tov_pretraining
        if tov_pretraining:
            self.NUM_CLASSES = 2

        self.current_batch = 0
        self.current_test_video = 0
        self.epochs_finished = 0
        if self.debug_mode:
            self.delivered_batches = []
        
        self.training_vidpath_label_tuples = []
        self.test_vidpath_label_tuples = []
        self.current_split = current_split
        self.set_training_tuples()
        self.set_test_tuples()
        
    @abstractmethod
    def set_training_tuples(self):
        pass
    
    @abstractmethod
    def set_test_tuples(self):
        pass
    
    def reset_to_split(self, split_index):
        # Todo: Check for validity according to provided number of splits
        self.current_split = split_index
        self.current_batch = 0
        self.epochs_finished = 0
        self.current_test_video = 0
        
    def __create_crops_from_video(self, vidfile_dict, num_crops=1):
        video_array = open_video(presampling_depth=self.TEMPORAL_DEPTH, **vidfile_dict)
        # video_array = open_video(presampling_depth=None, **vidfile_dict)
        
        num_video_frames = video_array.shape[0]
        video_height = video_array.shape[1]
        video_width = video_array.shape[2]
        
        if num_video_frames < self.TEMPORAL_DEPTH:
            num_repeat_ops = int(self.TEMPORAL_DEPTH / num_video_frames)
            repeated_video = video_array
            for i in range(num_repeat_ops):
                repeated_video = np.concatenate((repeated_video, video_array), 0)
            video_array = repeated_video
            num_video_frames = video_array.shape[0]
        
        crops = np.zeros(
            (num_crops, self.TEMPORAL_DEPTH, self.INPUT_HEIGHT, self.INPUT_WIDTH, self.INPUT_CHANNELS),
            dtype=np.uint8)
        
        for i in range(num_crops):
            start_frame = random.randint(0, num_video_frames - self.TEMPORAL_DEPTH)
            end_frame = start_frame + self.TEMPORAL_DEPTH
            start_width = random.randint(0, video_width - self.INPUT_WIDTH)
            end_width = start_width + self.INPUT_WIDTH
            start_height = random.randint(0, video_height - self.INPUT_HEIGHT)
            end_height = start_height + self.INPUT_HEIGHT
            crops[i, :, :, :, :] = video_array[start_frame:end_frame, start_height:end_height, start_width:end_width, :]
            
            if not crops[i, :, :, :, :].any():
                print("(EE) --- empty crops!!!")
                print(start_frame, end_frame, video_array.shape)
                print(vidfile_dict)
                # input()
        return crops
        
    def get_next_training_batch(self, lock):
        batch = np.zeros(
            (self.batch_size, self.TEMPORAL_DEPTH, self.INPUT_HEIGHT, self.INPUT_WIDTH, self.INPUT_CHANNELS),
            dtype=np.float32)
        if self.tov_pretraining:
            labels = np.zeros((self.batch_size, 2))
        else:
            labels = np.zeros((self.batch_size, self.NUM_CLASSES), dtype=np.float32)

        tuples = self.training_vidpath_label_tuples[self.current_split]
        
        lock.acquire()
        start = self.current_batch * self.batch_size
        end = start + self.batch_size
        self.current_batch += 1
        
        if end <= len(tuples):
            lock.release()
            for i in range(self.batch_size):
                vidfile_dict, action_label = tuples[start + i]
                if self.tov_pretraining:
                    clip = self.__create_crops_from_video(vidfile_dict)
                    count = 0
                    while not enough_motion_present(clip) and count <= 10:
                        clip = self.__create_crops_from_video(vidfile_dict)
                        count += 1
                    permute = np.random.rand() > 0.20
                    if not enough_motion_present(clip):
                        permute = False
                    if permute:
                        clip = permute_clip(clip)
                    batch[i, :, :, :, :] = clip
                    labels[i] = one_hot(int(permute), 2)
                else:
                    batch[i, :, :, :, :] = self.__create_crops_from_video(vidfile_dict)
                    labels[i, :] = one_hot(action_label, self.NUM_CLASSES)
            epoch_ended = False
        else:
            # pad last returned batch with crops from random videos
            for i in range(self.batch_size):
                if start + i < len(tuples):
                    indx = start + i
                else:
                    indx = np.random.randint(len(tuples))
                vidfile_dict, action_label = tuples[indx]
                if self.tov_pretraining:
                    clip = self.__create_crops_from_video(vidfile_dict)
                    count = 0
                    while not enough_motion_present(clip) and count <= 10:
                        clip = self.__create_crops_from_video(vidfile_dict)
                        count += 1
                    permute = np.random.rand() > 0.20
                    if not enough_motion_present(clip):
                        permute = False
                    if permute:
                        clip = permute_clip(clip)
                    batch[i, :, :, :, :] = clip
                    labels[i] = one_hot(int(permute), 2)
                else:
                    batch[i, :, :, :, :] = self.__create_crops_from_video(vidfile_dict)
                    labels[i, :] = one_hot(action_label, self.NUM_CLASSES)
            random.shuffle(tuples)
            epoch_ended = True
            self.current_batch = 0
            lock.release()
        if self.debug_mode:
            self.delivered_batches.append([batch, labels, epoch_ended])
        return batch, labels, epoch_ended
    
    def get_next_test_video_clips(self, offset=1):
        if self.tov_pretraining:
            print("ERROR Testing tov testing not supported")
            return
        tuples = self.test_vidpath_label_tuples[self.current_split]
        vidfile_dict, action_index = tuples[self.current_test_video]
        video_array = open_video(**vidfile_dict)
        num_video_frames = video_array.shape[0]
        video_height = video_array.shape[1]
        video_width = video_array.shape[2]
        num_clips = int((num_video_frames - self.TEMPORAL_DEPTH) / offset) + 1
        
        clips = np.zeros((num_clips, self.TEMPORAL_DEPTH, self.INPUT_HEIGHT, self.INPUT_WIDTH, self.INPUT_CHANNELS))
        
        # Get all center crop clips from the video
        y_offset = int(self.INPUT_HEIGHT / 2)
        x_offset = int(self.INPUT_WIDTH / 2)
        y_start = int(video_height / 2) - y_offset
        y_end = int(video_height / 2) + y_offset
        x_start = int(video_width / 2) - x_offset
        x_end = int(video_width / 2) + x_offset
        
        for i in range(num_clips):
            start = i * offset
            end = start + self.TEMPORAL_DEPTH
            clips[i] = video_array[start:end, y_start:y_end, x_start:x_end, :]
        
        if self.current_test_video == len(tuples) -1:
            # this was the last video
            self.current_test_video = 0
            test_ended = True
        else:
            self.current_test_video += 1
            test_ended = False
        return clips, one_hot(action_index, self.NUM_CLASSES), test_ended


class UCF101Provider(GenericDataProvider):
    def __init__(self, batch_size=40, tov_pretraining=False, debug_mode=False, current_split=0):
        self.SAMPLING_WIDTH = 160
        self.SAMPLING_HEIGHT = 120
        self.NUM_CLASSES = 101
        self.num_splits = 3
        self.basedir = './datasets/UCF-101'
        super().__init__(batch_size, tov_pretraining, debug_mode, current_split)
        
    def set_training_tuples(self):
        def dict_label_tuple_from_string(actionstr):
            path_end, class_str = actionstr.split(' ')
            dict = {'filename' : self.basedir + '/' + path_end,
                    'size': (self.SAMPLING_WIDTH, self.SAMPLING_HEIGHT)}
            # 1 has to be subtracted from class index, because datasets starts enumerating action classes at 1
            action_index = int(class_str) -1
            return (dict, action_index)
            
        for i in range(self.num_splits):
            with open(self.basedir + '/ucfTrainTestlist' + '/trainlist0{}.txt'.format(i + 1)) as file:
                tuples = file.read().splitlines()
            tuples = list(map(dict_label_tuple_from_string, tuples))
            random.shuffle(tuples)
            self.training_vidpath_label_tuples.append(tuples)
    
    def set_test_tuples(self):
        with open(self.basedir + '/ucfTrainTestlist' + '/classInd.txt') as file:
            classes = file.read().splitlines()
        classes = map(lambda s: s.split(' '), classes)
        class_index_dict = {}
        for indx, classstring in classes:
            class_index_dict[classstring] = int(indx) - 1
            
        def dict_label_tuple_from_string(actionstr):
            foldername, filename = actionstr.split('/')
            dict = {'filename' : self.basedir + '/' + actionstr,
                    'size' : (self.SAMPLING_WIDTH, self.SAMPLING_HEIGHT)}
            action_index = class_index_dict[foldername]
            return (dict, action_index)
        
        for i in range(self.num_splits):
            with open(self.basedir + '/ucfTrainTestlist' + '/testlist0{}.txt'.format(i + 1)) as file:
                tuples = file.read().splitlines()
            # tuples = map(lambda s: s.split('/'), tuples)
            tuples = list(map(dict_label_tuple_from_string, tuples))
            self.test_vidpath_label_tuples.append(tuples)
    
    
class CharadesProvider(GenericDataProvider):
    """
    Provider of the charades dataset.
    NOTE: vidfile_label_tuples needs to contain tuples of format (path_to_vidfile, onehot_action_label, begin_time, end_time)
        since the videos in charades dataset contain multiple actions now!!!
    """
    def __init__(self, batch_size=40, tov_pretraining=False, debug_mode=False):
        self.num_splits = 1
        self.base_dir = './datasets/charades'
        self.NUM_CLASSES = 157  # can be overwritten in base-class constructor according to tov_pretraining
        self.SAMPLING_WIDTH = None
        self.SAMPLING_HEIGHT = None
        with open(self.base_dir + '/charades_meta/Charades_v1_classes.txt') as file:
            classstrings = file.read().splitlines()
        self.class_index_dict = {}
        self.class_text_dict = {}
        for i in range(len(classstrings)):
            split = classstrings[i].split(' ', 1)
            self.class_index_dict[split[0]] = i
            self.class_text_dict[split[0]] = split[1]
        self.testIDs = []
        self.current_test_id = 0
        super().__init__(batch_size, tov_pretraining, debug_mode, current_split=0)
        
        
    def __fill_list_from_file(self, list, file):
        SMALLEST_INPUT_LENGTH = min(self.INPUT_HEIGHT, self.INPUT_WIDTH)
    
        path_to_charades_train_file = self.base_dir + file
        with open(path_to_charades_train_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            headerline = next(reader)
            id_indx, action_indx = headerline.index('id'), headerline.index('actions')
            for row in reader:
                id = row[id_indx]
                if 'test' in file:
                    self.testIDs.append(id)
                actions = row[action_indx]
                if len(actions) > 0:  # some videos are not annotated with action classes and temporal intervals, skip!
                    path_to_vidfile = self.base_dir + '/Charades_v1_480/{}.mp4'.format(id)
                    for action_triple in actions.split(';'):
                        actionclass, begin_time, end_time = action_triple.split(' ')
                        action_index = self.class_index_dict[actionclass]
                        tmp = ({'filename' : path_to_vidfile,
                                'size' : SMALLEST_INPUT_LENGTH,
                                'interval' : (float(begin_time), float(end_time))},
                               action_index)
                        list.append(tmp)

    def set_training_tuples(self):
        self.__fill_list_from_file(self.training_vidpath_label_tuples, '/charades_meta/Charades_v1_train.csv')
        random.shuffle(self.training_vidpath_label_tuples)
        self.training_vidpath_label_tuples = [self.training_vidpath_label_tuples]
    
    def set_test_tuples(self):
        self.__fill_list_from_file(self.test_vidpath_label_tuples, '/charades_meta/Charades_v1_test.csv')
        self.test_vidpath_label_tuples = [self.test_vidpath_label_tuples]
    
    def get_next_test_video(self):
        current_id = self.testIDs[self.current_test_id]
        vidpath = self.base_dir + '/Charades_v1_480/{}.mp4'.format(current_id)
        video_array = open_video(vidpath, min(self.INPUT_HEIGHT, self.INPUT_WIDTH))
        video_height = video_array.shape[1]
        video_width = video_array.shape[2]
        
        y_offset = int(self.INPUT_HEIGHT / 2)
        x_offset = int(self.INPUT_WIDTH / 2)
        y_start = int(video_height / 2) - y_offset
        y_end = int(video_height / 2) + y_offset
        x_start = int(video_width / 2) - x_offset
        x_end = int(video_width / 2) + x_offset
        train_video_crop = video_array[:,y_start:y_end,x_start:x_end,:]
        
        if self.current_test_id == len(self.testIDs) -1:
            test_ended = True
            self.current_test_id = 0
        else:
            test_ended = False
            self.current_test_id += 1
        
        return train_video_crop, current_id, test_ended

class KineticsPretrainProvider(GenericDataProvider):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.base_dir = './datasets/kinetics'
        self.video_names = os.listdir(self.base_dir + '/kinetics_video/train')
        self.num_videos = len(self.video_names)
        self.model_replica = 3
        self.current_batch = 0
        self.NUM_CLASSES = 2  # correct and incorrect temporal order
        super().__init__(batch_size, False, False, current_split=0)
        
    def set_training_tuples(self):
        pass
    
    def set_test_tuples(self):
        pass
    
    def get_next_training_batch(self, lock):
        batch = np.zeros(
            (self.batch_size, self.model_replica, self.TEMPORAL_DEPTH, self.INPUT_HEIGHT, self.INPUT_WIDTH, self.INPUT_CHANNELS),
            dtype=np.float32)
        labels = np.zeros((self.batch_size, 2))
        
        lock.acquire()
        start = self.current_batch * self.batch_size
        end = start + self.batch_size
        self.current_batch += 1
    
        if end <= self.num_videos:
            lock.release()
            for i in range(self.batch_size):
                vidfile_dict = {
                    'filename' : self.base_dir + '/kinetics_video/train/' + self.video_names[start+i],
                    'size' : 120,
                    'interval' : None,
                    'presampling_depth' : (self.model_replica+1) * self.TEMPORAL_DEPTH
                }
                clip = open_video(**vidfile_dict)
                # cut out center region of size INPUT_WIDTH * INPUT_HEIGHT
                video_height = clip.shape[1]
                video_width = clip.shape[2]
                y_offset = int(self.INPUT_HEIGHT / 2)
                x_offset = int(self.INPUT_WIDTH / 2)
                y_start = int(video_height / 2) - y_offset
                y_end = int(video_height / 2) + y_offset
                x_start = int(video_width / 2) - x_offset
                x_end = int(video_width / 2) + x_offset
                clip = clip[:,y_start:y_end,x_start:x_end,:]
                
                permute = np.random.rand() >= 0.25  # i.e. create negative example (more negatives needed)
                T = self.TEMPORAL_DEPTH
                if permute:
                    # determine whether to include first fragment in middle
                    # clip contains 4 fragments of length T
                    insert_first_fragment = np.random.rand() >= 0.5
                    if insert_first_fragment:
                        clip[2*T:3*T] = clip[:T]
                        batch[i] = np.stack(np.split(clip[T:], 3), 0)
                    else:
                        clip[T:2*T] = clip[3*T:]
                        batch[i] = np.stack(np.split(clip[:3*T], 3), 0)
                else:
                    batch[i] = np.stack(np.split(clip[:3*T], 3), 0)
                # play_clip(np.concatenate(batch[:,i],0).astype(np.uint8))
                labels[i] = one_hot(int(permute), 2)
            epoch_ended = False
        else:
            # pad last returned batch with crops from random videos
            for i in range(self.batch_size):
                if start + i < self.num_videos:
                    indx = start + i
                else:
                    indx = np.random.randint(self.num_videos)
                vidfile_dict = {
                        'filename' : self.base_dir + '/kinetics_video/train/' + self.video_names[indx],
                        'size' : 120,
                        'interval' : None,
                        'presampling_depth' : (self.model_replica+1) * self.TEMPORAL_DEPTH
                    }
                clip = open_video(**vidfile_dict)
                # cut out center region of size INPUT_WIDTH * INPUT_HEIGHT
                video_height = clip.shape[1]
                video_width = clip.shape[2]
                y_offset = int(self.INPUT_HEIGHT / 2)
                x_offset = int(self.INPUT_WIDTH / 2)
                y_start = int(video_height / 2) - y_offset
                y_end = int(video_height / 2) + y_offset
                x_start = int(video_width / 2) - x_offset
                x_end = int(video_width / 2) + x_offset
                clip = clip[:,y_start:y_end,x_start:x_end,:]
                
                permute = np.random.rand() >= 0.25  # i.e. create negative example (more negatives needed)
                T = self.TEMPORAL_DEPTH
                if permute:
                    # determine whether to include first fragment in middle
                    # clip contains 4 fragments of length T
                    insert_first_fragment = np.random.rand() >= 0.5
                    if insert_first_fragment:
                        clip[2*T:3*T] = clip[:T]
                        batch[i] = np.stack(np.split(clip[T:], 3), 0)
                    else:
                        clip[T:2*T] = clip[3*T:]
                        batch[i] = np.stack(np.split(clip[:3*T], 3), 0)
                else:
                    batch[i] = np.stack(np.split(clip[:3*T], 3), 0)
                # play_clip(np.concatenate(batch[i],0).astype(np.uint8))
                labels[i] = one_hot(int(permute), 2)
            
            random.shuffle(self.video_names)
            epoch_ended = True
            self.current_batch = 0
            lock.release()
        
        if self.debug_mode:
            self.delivered_batches.append([batch, labels, epoch_ended])
        return batch, labels, epoch_ended
        pass
    
        
if __name__ == "__main__":
    import time
    import threading
    from videotools import play_clip
    # prov = UCF101Provider()
    prov = KineticsPretrainProvider(40)
    lock = threading.Lock()
    before = time.time()
    batch = prov.get_next_training_batch(lock)
    print('Took', time.time() - before)
    # prov.current_batch = 1240
    # prov.current_test_video = 2647
    # prov.current_test_video = 9750
    # lock = threading.Lock()
    # for i in range(10):
    #     before = time.time()
    #     batch = prov.get_next_test_video_clips(offset=16)
    #     print(np.shape(batch[0]))
    #     print(i, 'Batch ready! Took', time.time() - before)
    
    # prov.current_batch = 200
    # for i in range(300):
    #     before = time.time()
    #     batch, labels, epoch_ended = prov.get_next_training_batch()
    #     print('Took', time.time() - before)
    #     if epoch_ended:
    #         break
    # prov.reset_to_split(1)
    # prov.current_batch = 200
    # for i in range(300):
    #     before = time.time()
    #     batch, labels, epoch_ended = prov.get_next_training_batch()
    #     print('Took', time.time() - before)
    #     if epoch_ended:
    #         break
    
    # print(len(prov.test_vidpath_label_tuples[prov.current_split]))
    # prov.current_test_video = 3780
    # for i in range(6):
    #     clips, onehot_label, test_ended = prov.get_next_test_video_clips()
    #     print(prov.current_test_video, test_ended)
