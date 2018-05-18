import numpy as np
from sklearn.metrics import confusion_matrix


test1 = np.load("/mnt/workspace/activitynet_info/kinetics400_clip_RGB_test.npz")
#test2 = np.load("/mnt/workspace/activitynet_info/kinetics400_RGB_test.npz")
test3 = np.load("/mnt/workspace/activitynet_info/kinetics600_clip_RGB_test.npz")
#video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]
video_pred = []
labels = []
for vid in range(4900):
    scores1 = test1["scores"][vid][0]
    #scores2 = test2["scores"][vid][0]
    scores3 = test3["scores"][vid][0]
    video_pred1 = np.mean(scores1, axis=0)
    #video_pred2 = np.mean(scores2, axis=0)
    video_pred3 = np.mean(scores3, axis=0)
    video_pred.append(np.argmax(video_pred1 + video_pred3))
    labels.append(test3["scores"][vid][1])
    print(vid)

print(labels)
print(video_pred)
cf = confusion_matrix(labels, video_pred).astype(float)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)
print(cf)
print(cls_hit)
print(cls_cnt)
cls_acc = cls_hit / cls_cnt
print(cls_acc)
print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))