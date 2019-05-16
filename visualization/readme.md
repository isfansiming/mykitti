
# 2019.5.16

## write the detection result to txt files and eval
PS:result[:3][n][5]
3 means ('Car','Pedestrian','Cyclist')
n means each class has n object
5 means xmin,ymin,xmax,ymax,probability

run

    python tools/result2kitti.py
then

    ./kitti_eval/evaluate_object_3d_offline data/KITTI/object/training/label_2 work_dirs/carpedcyc/faster_rcnn_r50_fpn_1x_voc0712_carpedcyc190513/results

------

    Thank you for participating in our evaluation!
    Loading detections...
    number of files for evaluation: 3769
      done.
    save work_dirs/carpedcyc/faster_rcnn_r50_fpn_1x_voc0712_carpedcyc190513/results/plot/car_detection.txt
    car_detection AP: 90.691986 90.443130 89.466705
    PDFCROP 1.38, 2012/11/02 - Copyright (c) 2002-2012 by Heiko Oberdiek.
    ==> 1 page written on `car_detection.pdf'.
    save work_dirs/carpedcyc/faster_rcnn_r50_fpn_1x_voc0712_carpedcyc190513/results/plot/pedestrian_detection.txt
    pedestrian_detection AP: 88.646851 80.660240 79.710365
    PDFCROP 1.38, 2012/11/02 - Copyright (c) 2002-2012 by Heiko Oberdiek.
    ==> 1 page written on `pedestrian_detection.pdf'.
    save work_dirs/carpedcyc/faster_rcnn_r50_fpn_1x_voc0712_carpedcyc190513/results/plot/cyclist_detection.txt
    cyclist_detection AP: 90.909096 90.909096 90.776375
    PDFCROP 1.38, 2012/11/02 - Copyright (c) 2002-2012 by Heiko Oberdiek.
    ==> 1 page written on `cyclist_detection.pdf'.
    Finished 2D bounding box eval.
    Finished Birdeye eval.
    Finished 3D bounding box eval.
    Your evaluation results are available at:
    work_dirs/carpedcyc/faster_rcnn_r50_fpn_1x_voc0712_carpedcyc190513/results

