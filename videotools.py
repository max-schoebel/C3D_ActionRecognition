import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
import random
import math


def open_video(filename, size, interval=None, presampling_depth=None):
    video = cv2.VideoCapture(filename)
    video.set(cv2.CAP_PROP_CONVERT_RGB, True)
    
    if type(size) == tuple:
        desired_width = size[0]
        desired_height = size[1]
    elif type(size) == int:
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if video_width == 0 or video_height == 0:
            print(filename)
        resize_factor = size / min(video_width, video_height)
        desired_width = int(np.round(video_width * resize_factor))
        desired_height = int(np.round(video_height * resize_factor))
    else:
        print('ERROR videotools.open_video: Wrong value for argument "size"')
        return

    repeat = True
    while repeat:
        if interval is not None:
            fps = video.get(cv2.CAP_PROP_FPS)
            start_frame = int(interval[0] * fps)
            end_frame = int(interval[1] * fps)
            max_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if end_frame > max_frames:
                end_frame = max_frames
            # print(filename)
            # print(interval)
            # print('fps', fps)
            # print(start_frame, end_frame)
        else:
            start_frame = 0
            end_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if presampling_depth is not None and end_frame > presampling_depth:
            # presample between previously determined interval (start_frame, end_frame),
            # to speed up reading of the video. Only reads what is actually needed.
            start_frame = random.randint(start_frame, end_frame - presampling_depth)
            # start_frame = end_frame - presampling_depth
            end_frame = start_frame + presampling_depth
    
        if end_frame - start_frame < 16:
            start_frame = end_frame -16
        num_frames = end_frame - start_frame
        frame_matrix = np.zeros((num_frames, desired_height, desired_width, 3), dtype=np.uint8)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
        for i in range(num_frames):
            success, frame = video.read()
            if not success:
                # try to redo once if loading was unsuccessful:
                video.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
                success, frame = video.read()
                # if success:
                #     print('RECOVERING WORKED!!!')
                # else:
                #     print('RECOVERING DID NOT WORK!!!', start_frame, end_frame, i)
            if success:
                frame_matrix[i, :, :, :] = cv2.resize(frame, (desired_width, desired_height))
            else:
                break
                
        if not success and presampling_depth is None:
            # not all frames from video could be read, but there are enough
            # cut off zero-frames in frame matrix and return
            frame_matrix = frame_matrix[:i, :, :, :]
            repeat = False
        elif success:
            # all frames could be read, our work here is done.
            repeat = False
        elif not success and presampling_depth is not None:
            # we need to repeat, because not all needed frames could be obtained
            # print("REPEATED!!!!")
            repeat = True
        else:
            # DEBUG case, this should never happen:
            print("IT HAPPENED!!!")
    
    video.release()
    if frame_matrix.shape[0] < presampling_depth and frame_matrix.shape[0] > 0:
        stretch_factor = math.ceil(presampling_depth / frame_matrix.shape[0])
        frame_matrix = np.repeat(frame_matrix, stretch_factor, axis=0)[:presampling_depth]
    elif frame_matrix.shape[0] == 0:
        print('[open_video] could not open video, got 0 Frames', filename)
    return frame_matrix


def permute_clip(clip):
    """
    :param clip: input np.array representing a video clip with shape (1, temp_depth, height, width, channels)
    :param num_fragments: number of slices
    :return: permuted clip
    """
    # sample number of kept beginning and end-frames between 1 and 3
    min_keep = 1
    max_keep = 3
    keep_frames = np.random.randint(min_keep, max_keep+1)
    # print(keep_frames)
    # sample number of fragments to obtain from in between the kept frames
    num_fragments = np.random.randint(2, 6)
    # print(num_fragments)
    splits = np.array_split(clip[:, keep_frames:-keep_frames, :, :], num_fragments, axis=1)
    perm = np.random.permutation(num_fragments)
    while np.array_equal(perm ,list(range(num_fragments))) or np.array_equal(perm, reversed(list(range(num_fragments)))):
        perm = np.random.permutation(num_fragments)
    splits = [splits[i] for i in perm]
    splits = np.concatenate(splits, axis=1)
    # print(np.shape(splits))
    # print(np.shape(clip[:,:keep_frames,:,:]))
    # print(np.shape(clip[:,-keep_frames:,:,:]))
    permuted_clip = np.concatenate((clip[:,:keep_frames,:,:], splits, clip[:,-keep_frames:,:,:]), axis=1)
    return permuted_clip


def enough_motion_present(clip, num_fragments=3):
    splits = np.array_split(clip, num_fragments, axis=1)
    means = [np.mean(s, axis=1) for s in splits]
    height, width = clip.shape[2], clip.shape[3]
    dists = [np.linalg.norm(means[i] - means[i+1])/(width * height) for i in range(len(means) -1)]
    # print(dists)
    if np.sum(dists) > 0.4:
        # print('yes', np.sum(dists))
        return True
    else:
        # print('no', np.sum(dists))
        return False


def one_hot(index, num_classes):
    encoding = np.zeros(num_classes, dtype=np.float32)
    encoding[index] = 1
    return encoding


def play_clip(clip):
    for i in range(clip.shape[0]):
        cv2.imshow('clip', clip[i])
        cv2.waitKey(100)
    cv2.destroyAllWindows()
    
    
def are_equal(dataprovider_batches, extracted_batches, batch_size):
    ip_data, ip_labels, _ = dataprovider_batches
    tf_data1, tf_labels1, _ = extracted_batches[0]
    tf_data2, tf_labels2, _ = extracted_batches[1]
    bhalf = int(batch_size / 2)
    a = np.array_equal(ip_data[0:bhalf], tf_data1)
    b = np.array_equal(ip_data[bhalf:batch_size], tf_data2)
    c = np.array_equal(ip_labels[0:bhalf], tf_labels1)
    d = np.array_equal(ip_labels[bhalf:batch_size], tf_labels2)
    return a and b and c and d


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Implementation taken from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    def __time_test():
        file_list = ['./datasets/UCF-101'+'/'+foldername + '/' + filename for
                     foldername in os.listdir('./datasets/UCF-101') for
                     filename in os.listdir('./datasets/UCF-101/'+foldername)]
        # path = './datasets/charades/Charades_v1_480'
        # file_list = [path + '/' + filename for filename in os.listdir(path)]
        for i in range(500):
            before = time.time()
            arr = open_video(file_list[i], (160, 120))
            print("took {}".format(time.time() - before))
    

if __name__ == '__main__':
    from DataProvider import CharadesProvider
    clip = np.zeros((1,16,20,20,3))
    for i in range(16):
        clip[0,i] = i
    permuted = permute_clip(clip)
    for clip in permuted[0]:
        print(np.mean(clip))
