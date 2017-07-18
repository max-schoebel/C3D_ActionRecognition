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


if __name__ == '__main__':
    file_list = ['UCF-101'+'/'+foldername + '/' + filename for
                 foldername in os.listdir('UCF-101') for
                 filename in os.listdir('UCF-101/'+foldername)]
    for i in range(100):
        before = time.time()
        arr = open_video(file_list[i], 100, 100)
        print("took {}".format(before - time.time()))
