import os
import cv2
import numpy as np
import time


def open_video(video_file, desired_width, desired_height):
    video = cv2.VideoCapture(video_file)
    video.set(cv2.CAP_PROP_CONVERT_RGB, True)

    max_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_matrix = np.zeros((max_frames, desired_height, desired_width, 3), dtype=np.uint8)
    run = True
    while run:
        frame_index = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        run, frame = video.read()
        
        # try to redo once if loading was unsuccessful:
        if not run and frame_index < max_frames:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            run, frame = video.read()
        if run:
            frame_matrix[frame_index, :, :, :] = cv2.resize(frame, (desired_width, desired_height))
            
    # cut off if some frames could not be read
    if frame_index < max_frames:
        frame_matrix = frame_matrix[:frame_index, :, :, :]
    video.release()
    return frame_matrix


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


if __name__ == '__main__':
    file_list = ['UCF-101'+'/'+foldername + '/' + filename for
                 foldername in os.listdir('UCF-101') for
                 filename in os.listdir('UCF-101/'+foldername)]
    for i in range(100):
        before = time.time()
        arr = open_video(file_list[i], 160, 120)
        print("took {}".format(time.time() - before))
