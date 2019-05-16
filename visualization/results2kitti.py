def results2kitti():
    '''
    Convert mmdetection's results to kitti label's format.
    needs path of images,indexs,results.
    '''
    # 测试多张图片
    file_dir = 'data/KITTI/object/training/image_2'
    idx_file = 'data/KITTI/object/image_sets/val.txt'
    result_dir = 'work_dirs/carpedcyc/faster_rcnn_r50_fpn_1x_voc0712_carpedcyc190513/results'

    import os
    from mmdet.core import get_classes
    import numpy as np
    from mmcv.visualization import color_val
    from mmcv.image import imread, imwrite
    import cv2

    print('load indexs.')
    idxs = []
    for line in open(idx_file,'r'):
        idxs.append(int(line))

    print('load imgs.')
    imgs = []
    for idx in idxs:
        imgs.append(mmcv.imread(os.path.join(file_dir, '%06d.png' % (idx))))

    print('inference.')
    #convert result[:4][:5] to a txt file
    results = {}
    for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
        idx = idxs[i]
        typename = ['Car','Pedestrian','Cyclist']
        for type_id in range(3):
            for j in result[type_id]:
                box2d = result[0][j][:4]
                prob = result[0][j][4]
                output_str = typename[type_id] + " -1 -1 -10 "
                output_str += "%f %f %f %f " % (box2d[0], box2d[1], box2d[2], box2d[3])
                output_str += "-1 -1 -1 -1000 -1000 -1000 -10 %f" % (prob)###
                results[i].append(output_str)
        if not os.path.exists(result_dir): os.mkdir(result_dir)
        output_dir = os.path.join(result_dir, 'data')
        if not os.path.exists(output_dir): os.mkdir(output_dir)

    print('write results.')
    #write txt
    for i in results:
        pred_filename = os.path.join(output_dir, '%06d.txt' % (idxs[i]))
        fout = open(pred_filename, 'w')
        for line in results[i]:
            fout.write(line + '\n')
        fout.close()