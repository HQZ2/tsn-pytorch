import pickle
import os

FRAMES_NUM_PKL = '/mnt/workspace/pkls/frames_num.pkl'
ACTNET200V13_PKL = '/mnt/workspace/pkls/actNet200-V1-3.pkl'
VIDEO_PATH = '/mnt/workspace/activitynet-frames/resized-activitynet-frames'
SAVE_DIR = '/mnt/workspace/activitynet_train_list.txt'


def generdate_frame_info():
    with open(FRAMES_NUM_PKL, 'rb') as f:
        frames_num = pickle.load(f)
    with open(ACTNET200V13_PKL, 'rb') as f:
        all_groundtruth = pickle.load(f)['database']
    info = []

    # filter by subset
    remainkeyset = []
    for key in all_groundtruth.keys():
        if all_groundtruth[key]['subset'] == 'training':
            remainkeyset.append(key)
    groundtruth = dict()
    for key in remainkeyset:
        groundtruth[key] = all_groundtruth[key]
    vids = remainkeyset
    for vid in vids:
        annos = groundtruth[vid]['annotations']
        for anno in annos:
            label = anno['class']
            break
        vid2 = 'v_{}.mp4'.format(vid)
        try:
            num = frames_num[vid2]
        except:
            continue
        info.append((os.path.join(VIDEO_PATH, vid2), num, label))
    return info



if __name__ == '__main__':
    train_info = generdate_frame_info()
    with open(SAVE_DIR, 'w') as fout:
        for item in train_info:
            fout.write('{} {} {}\n'.format(item[0], item[1], item[2]))