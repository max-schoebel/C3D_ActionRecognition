from abc import ABC, abstractmethod
import numpy as np
import random
from videotools import open_video, one_hot
import csv


class GenericDataProvider(ABC):
    TEMPORAL_DEPTH = 16
    INPUT_CHANNELS = 3
    NUM_CLASSES = 101
    INPUT_WIDTH = 112
    INPUT_HEIGHT = 112

    def __init__(self, batch_size, debug_mode, current_split):
        """
        videofile_label_tuples... list of self.num_splits lists containing tuples
            of format (path_to_filename, onehot_encoded_label)
        """
        self.batch_size = batch_size
        self.debug_mode = debug_mode

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
        
    def __create_crops_from_video(self, filename, num_crops=1):
        video_array = open_video(filename, self.SAMPLING_WIDTH, self.SAMPLING_HEIGHT)
        num_video_frames = video_array.shape[0]
        
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
            start_frame = np.random.randint(num_video_frames - self.TEMPORAL_DEPTH)
            end_frame = start_frame + self.TEMPORAL_DEPTH
            start_width = np.random.randint(self.SAMPLING_WIDTH - self.INPUT_WIDTH)
            end_width = start_width + self.INPUT_WIDTH
            start_height = np.random.randint(self.SAMPLING_HEIGHT - self.INPUT_HEIGHT)
            end_height = start_height + self.INPUT_HEIGHT
            crops[i, :, :, :, :] = video_array[start_frame:end_frame, start_height:end_height, start_width:end_width, :]
            
            if not crops[i, :, :, :, :].any():
                print("(EE) --- empty crops!!!")
                print(start_frame, end_frame, video_array.shape)
                input()
        return crops
        
    def get_next_training_batch(self, lock):
        batch = np.zeros(
            (self.batch_size, self.TEMPORAL_DEPTH, self.INPUT_HEIGHT, self.INPUT_WIDTH, self.INPUT_CHANNELS),
            dtype=np.float32)
        labels = np.zeros((self.batch_size, self.NUM_CLASSES), dtype=np.float32)

        tuples = self.training_vidpath_label_tuples[self.current_split]
        
        lock.acquire()
        start = self.current_batch * self.batch_size
        end = start + self.batch_size
        self.current_batch += 1
        
        if end <= len(tuples):
            lock.release()
            for i in range(self.batch_size):
                path_to_vidfile, onehot_label = tuples[start + i]
                batch[i, :, :, :, :] = self.__create_crops_from_video(path_to_vidfile)
                labels[i, :] = onehot_label
            epoch_ended = False
        else:
            # pad last returned batch with crops from random videos
            for i in range(self.batch_size):
                if start + i < len(tuples):
                    indx = start + i
                else:
                    indx = np.random.randint(len(tuples))
                path_to_vidfile, onehot_label = tuples[indx]
                batch[i, :, :, :, :] = self.__create_crops_from_video(path_to_vidfile)
                labels[i, :] = onehot_label
            random.shuffle(tuples)
            epoch_ended = True
            self.current_batch = 0
            lock.release()
        if self.debug_mode:
            self.delivered_batches.append([batch, labels, epoch_ended])
        return batch, labels, epoch_ended
    
    def get_next_test_video_clips(self):
        tuples = self.test_vidpath_label_tuples[self.current_split]
        path_to_vidfile, onehot_label = tuples[self.current_test_video]
        video_array = open_video(path_to_vidfile, self.SAMPLING_WIDTH, self.SAMPLING_HEIGHT)
        num_video_frames = video_array.shape[0]
        num_clips = num_video_frames - self.TEMPORAL_DEPTH + 1
        clips = np.zeros((num_clips, self.TEMPORAL_DEPTH, self.INPUT_HEIGHT, self.INPUT_WIDTH, self.INPUT_CHANNELS))
        
        y_offset = int(self.INPUT_HEIGHT / 2)
        x_offset = int(self.INPUT_WIDTH / 2)
        y_start = int(self.SAMPLING_HEIGHT / 2) - y_offset
        y_end = int(self.SAMPLING_HEIGHT / 2) + y_offset
        x_start = int(self.SAMPLING_WIDTH / 2) - x_offset
        x_end = int(self.SAMPLING_WIDTH / 2) + x_offset
        
        for i in range(num_clips):
            clips[i] = video_array[i:i+self.TEMPORAL_DEPTH, y_start:y_end, x_start:x_end, :]
        
        if self.current_test_video == len(tuples) -1:
            # this was the last video
            self.current_test_video = 0
            test_ended = True
        else:
            self.current_test_video += 1
            test_ended = False
        return clips, onehot_label, test_ended


class UCF101Provider(GenericDataProvider):
    SAMPLING_WIDTH = 160
    SAMPLING_HEIGHT = 120
    NUM_CLASSES = 101
    
    def __init__(self, batch_size=40, debug_mode=False, current_split=0):
        self.num_splits = 3
        self.basedir = './datasets/UCF-101'
        super().__init__(batch_size, debug_mode, current_split)
        
    def set_training_tuples(self):
        for i in range(self.num_splits):
            with open(self.basedir + '/ucfTrainTestlist' + '/trainlist0{}.txt'.format(i + 1)) as file:
                tuples = file.read().splitlines()
                tuples = map(lambda s: s.split(' '), tuples)
                # 1 has to be subtracted from class index, because datasets starts enumerating action classes at 1
                tuples = list(map(lambda s: (self.basedir + '/' + s[0], one_hot(int(s[1]) - 1, self.NUM_CLASSES)), tuples))
                random.shuffle(tuples)
            self.training_vidpath_label_tuples.append(tuples)
    
    def set_test_tuples(self):
        with open(self.basedir + '/ucfTrainTestlist' + '/classInd.txt') as file:
            classes = file.read().splitlines()
        classes = map(lambda s: s.split(' '), classes)
        class_index_dict = {}
        for indx, classstring in classes:
            class_index_dict[classstring] = int(indx) - 1
            
        for i in range(self.num_splits):
            with open(self.basedir + '/ucfTrainTestlist' + '/testlist0{}.txt'.format(i + 1)) as file:
                tuples = file.read().splitlines()
            tuples = map(lambda s: s.split('/'), tuples)
            tuples = list(map(lambda s: (self.basedir + '/' + s[0] + '/' + s[1], one_hot(class_index_dict[s[0]], self.NUM_CLASSES)), tuples))
            self.test_vidpath_label_tuples.append(tuples)
    
    
class CharadesProvider(GenericDataProvider):
    """
    Provider of the charades dataset.
    NOTE: vidfile_label_tuples needs to contain tuples of format (path_to_vidfile, onehot_action_label, begin_time, end_time) now
        since the videos in charades dataset contain multiple actions now!!!
    """
    SAMPLING_WIDTH = None
    SAMPLING_HEIGHT = None
    NUM_CLASSES = 157
    
    def __init__(self, batch_size=40, debug_mode=False):
        self.num_splits = 1
        self.base_dir = './datasets/charades'
        with open(self.base_dir + '/charades_meta/Charades_v1_classes.txt') as file:
            classstrings = file.read().splitlines()
        self.class_index_dict = {}
        self.class_text_dict = {}
        for i in range(len(classstrings)):
            split = classstrings[i].split(' ', 1)
            self.class_index_dict[split[0]] = i
            self.class_text_dict[split[0]] = split[1]
        super().__init__(batch_size, debug_mode, current_split=0)
        
        
    def set_training_tuples(self):
        path_to_charades_train_file = self.base_dir + '/charades_meta/Charades_v1_train.csv'
        with open(path_to_charades_train_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            headerline = next(reader)
            id_indx, action_indx = headerline.index('id'), headerline.index('actions')
            for row in reader:
                id = row[id_indx]
                actions = row[action_indx]
                if len(actions) > 0:  # some videos are not annotated with action classes and temporal intervals, skip!
                    path_to_vidfile = self.base_dir + '/charades_v1_480/{}.mp4'.format(id)
                    for action_triple in actions.split(';'):
                        actionclass, begin_time, end_time = action_triple.split(' ')
                        onehot_action_label = one_hot(self.class_index_dict[actionclass], self.NUM_CLASSES)
                        tmp = (path_to_vidfile, onehot_action_label, float(begin_time), float(end_time))
                        self.training_vidpath_label_tuples.append(tmp)
        self.training_vidpath_label_tuples = [self.training_vidpath_label_tuples]  # only one split!
        
    
    def set_test_tuples(self):
        # Not supported! Evaluation on charades is handled externally!
        pass
        
            
if __name__ == "__main__":
    import time
    prov = CharadesProvider()
    
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
