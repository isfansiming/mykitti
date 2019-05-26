'''autohd:isfansiming
convert labels(000001.txt ~ 007480.txt(only val) to frustum-pointnets' rgb_detection_val format)
like this
dataset/KITTI/object/training/image_2/000001.png 2 0.998467 389 181 424 202
dataset/KITTI/object/training/image_2/000001.png 2 0.0448065 512 176 528 187
...
dataset/KITTI/object/training/image_2/007480.png 3 0.979358 497 174 525 230
dataset/KITTI/object/training/image_2/007480.png 3 0.0244306 480 176 526 209

'''
import numpy as np
import os

datapath = 'dataset/KITTI/object/training/image_2/'
indexpath = 'dataset/KITTI/object/ImageSets'
result_filename_train = 'rgb_detection_train.txt'
result_filename_val = 'rgb_detection_val.txt'
labelpath='/home/fsm/视频/kitti/mmdetection2/work_dirs/carpedcyc/faster_rcnn_r50_fpn_1x_voc0712_carpedcyc190513/results/data'
trainidxs=[]
validxs=[]
typename = ['Car', 'Pedestrian', 'Cyclist']
type2num = {typename[0]:2, typename[1]:1, typename[2]:3}

print('load indexs.')
index_train_file = os.path.join(indexpath,'train.txt')
for line in open(index_train_file,'r'):
    trainidxs.append(int(line))
index_val_file = os.path.join(indexpath,'val.txt')
for line in open(index_val_file,'r'):
    validxs.append(int(line))

print('convert val.')

outputs = []
for i,id in enumerate(validxs):
    fin=os.path.join(labelpath,'%06d.txt'%(int(id)))
    for line in open(fin,'r'):
        t = line.rstrip().split(" ")
        imgpath=(datapath+'%06d.png'%(id))
        output_str = imgpath + " "
        type=type2num[t[0]]
        output_str += "%d"%(type)+" "
        prob=float(t[15])
        output_str += "%f"%(prob)+" "
        box2d=np.array([int(float(t[i])) for i in range(4,8)])
        output_str += "%d %d %d %d"%(box2d[0],box2d[1],box2d[2],box2d[3])
        outputs.append(output_str)
        
    print('read %06d.txt Done.'%(validxs[i]))
print('Write to rgb_detection_val.txt')
fout = open(result_filename_val, 'w')
for line in outputs:
    fout.write(line + '\n')
fout.close
