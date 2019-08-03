import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2
import matplotlib.animation as animation
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern

from constants import *

VIDEO_DIR  = raw_movie_190526_path
BAT_NUMS   = ['bat290', 'bat294', 'bat296', 'bat298']
TRIAL_NUMS = ['no1', 'no2', 'no3', 'no4', 'no6']

VIDEO_PATHS = []
for b in BAT_NUMS:
    for t in TRIAL_NUMS:
        for rl in [2, 3]:
            VIDEO_PATHS.append(
                f'{VIDEO_DIR}/{b}/{t}/{t}_001/{t}_001_NX8-S1 Camera(0{rl})/NX8-S1 Camera.avi',
            )

for i, path in enumerate(VIDEO_PATHS):
    print(i, path)


def read_frame(cap):
    _, frame = cap.read()
    return frame


def normalize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / std


def convert_img_8bits(image):
    mx = image.max()
    mn = image.min()
    return (255 * ((image - mn) / (mx - mn))).astype('uint8')


def bat_line_detect(frames):
    out = np.array(frames)
    out = (out[:-1, :, :] - out[1:, :, :])**2
    out = np.mean(out, axis=0)
    out = normalize(out)

    out = cv2.GaussianBlur(out, (25, 25), 2)

    out = convert_img_8bits(out)

    out = np.where(out<6, out.min(), out.max())

#    kernel = np.ones((2,2),np.uint8)
#    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
#    kernel = np.ones((25,25),np.uint8)
#    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)


    return out



def play(path_to_video, start=None, stop=0):
    '''
    play video.

    path_to_video: str
    start: float or int (time(sec) from trigger)
    stop: float or int (time(sec) from trigger)
    '''
    cap = cv2.VideoCapture(path_to_video) 
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_rate = 50

    video_length = frame_count / frame_rate
    if start is None:
        start = -video_length
#    else:
#        start -= 6.0
#    stop -= 6.0
   
    frame_num = 0

    # jump to start point.
#    for i in range(int(frame_count + start * frame_rate)):
#        read_frame(cap)
#        frame_num += 1

    for i in range(100):
        read_frame(cap)
        frame_num += 1

    frame_range = 2
    frames = []
    for i in range(frame_range):
        frames.append(read_frame(cap))
        frame_num += 1

    while(cap.isOpened()):
        frames.append(read_frame(cap))

        if (
            cv2.waitKey(1) & 0xFF == ord('q') or
            frames is None or
#            frame_count + stop * frame_rate <= frame_num
            frame_num >= 300
        ):
            break

#        frame_feature = temporal_intensity(np.array(frames))
#        cv2.imshow('frame', np.append(convert_img_8bits(frame_feature), frames[0], axis=0))

        frame_sum = np.zeros(frames[0].shape)

        display = np.append(
            bat_line_detect(frames),
            convert_img_8bits(normalize(frames[0])),
            axis=0
        )
    #    display = bat_line_detect(frames)

        cv2.imshow('frame', display)

        frames.pop(0)
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    play(VIDEO_PATHS[30])


if __name__ == '__main__':
    main()
