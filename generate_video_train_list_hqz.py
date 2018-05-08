import pickle
import os

FRAMES_NUM_PKL = '/mnt/workspace/pkls/frames_num.pkl'
ACTNET200V13_PKL = '/mnt/workspace/pkls/actNet200-V1-3.pkl'
VIDEO_PATH = '/mnt/workspace/activitynet-frames/resized-activitynet-frames'


def load_groundtruth():
    with open(ACTNET200V13_PKL, 'rb') as f:
        all_groundtruth = pickle.load(f)['database']
    # filter by subset
    remainkeyset = []
    for key in all_groundtruth.keys():
        if all_groundtruth[key]['subset'] == 'training':
            remainkeyset.append(key)
    groundtruth = dict()
    for key in remainkeyset:
        groundtruth[key] = all_groundtruth[key]
    vids = remainkeyset
    return vids, groundtruth


def generdate_frame_info(vids,groundtruth):
    with open(FRAMES_NUM_PKL, 'rb') as f:
        frames_num = pickle.load(f)
    info = []
    count = 0
    for vid in vids:
        annos = groundtruth[vid]['annotations']
        for anno in annos:
            label = anno['label']
            break
        vid2 = 'v_{}.mp4'.format(vid)
        duration = frames_num[vid2]
        info.append(os.path.join(VIDEO_PATH, vid2), duration, label)
        count += 1
        print(info)
        if count > 10:
            break


if __name__ == '__main__':
    generdate_frame_info(load_groundtruth())