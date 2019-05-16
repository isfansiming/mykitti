import os
import numpy as np
import cv2

def show_image_det_from_lists(img,bboxes,types,occlusions=None,truncations=None,type_names=None,ignored_type_names=None,saveto=-1):
    #mmdetection result(list)->visualize and save image
    #result:3769*3*5,where 3769=#images, 3=#classes, 5 means [xmin,ymin,xmax,ymax,probability]
    pass
def save_txt_det_from_lists():
    #mmdetection result->txt(kitti)
    pass
def show_image_det_from_class(img,objects,type_names=None,ignored_type_names=None,occlusion=-1,truncation=-1,saveto=-1):
    '''
    (needs dataset class and object class from frustum-pointnets.)
    show image with labels from objects,bbox,occlusion,truncation
    :param img(ndarray):
    :param objects(list of class): type,xmin,ymin,xmax,ymax,occlusion,truncation
    :param type_names(tuple): like('Car','Pedestrian','Cyclist')
    :param ignored_type_names(tuple): like ('DontCare')
    :param occlusion(int): >0:show occlusion(0,1,2,3)
    :param truncation(int): >0:show truncation(0~1)
    :param saveto(str):str:save else: show
    :return: None
    '''
    assert objects.type_names != None
    for obj in objects:
        if obj.type in type_names:
            try:
                cv2.rectangle(img, (int(obj.xmin),int(obj.ymin)),
                    (int(obj.xmax),int(obj.ymax)), (0,255,0), 1)
                cv2.putText(img, obj.type[:3], (int(obj.xmin), int(obj.ymin - 15)), cv2.FONT_HERSHEY_DUPLEX, \
                        0.5, (0, 255, 0), 1)
            except AttributeError:
                print("No object.xmin/ymin/xmax/ymax/type(str)")
            if occlusion>0:
                try:
                    cv2.putText(img, 'o'+str(obj.occlusion), (int(obj.xmin), int(obj.ymin + 15)), cv2.FONT_HERSHEY_DUPLEX, \
                                0.5, (0, 255, 0), 1)
                except AttributeError:
                    print("No object.occlusion but you set occlusion!=-1")
            if truncation>0:
                try:
                    cv2.putText(img, 't'+str(obj.truncation), (int(obj.xmin), int(obj.ymin)), cv2.FONT_HERSHEY_DUPLEX, \
                                0.5, (0, 255, 0), 1)
                except AttributeError:
                    print("No object.occlusion but you set truncation!=-1")

        elif obj.type in ignored_type_names:
            cv2.rectangle(img, (int(obj.xmin),int(obj.ymin)),
                (int(obj.xmax),int(obj.ymax)), (0,100,100), 1)
            cv2.putText(img, 'ignored', (int(obj.xmin), int(obj.ymin - 15)), cv2.FONT_HERSHEY_DUPLEX, \
                        0.5, (0, 100, 100), 1)
    if isinstance(saveto,str):
        dir = os.abspath(os.dirname(saveto))
        if not os.path.exists(dir):
            os.makedirs(dir)
        img.save(saveto)
    else:
        cv2.imshow(img)
        cv2.waitKey(0)

def demo_vis_kitti_images():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    dataset = objects(os.path.join(ROOT_DIR, 'dataset/KITTI/object'),)
    for data_idx in range(0,7480):
        objects = dataset.get_label_objects(data_idx)
        objects[0].print_object()
        img = dataset.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        show_image_det_from_class(img=img, objects=objects, \
            typenames=('Car','Pedestrian','Cyclist'),ignored_type_names=('DontCare'), \
            occlusion=1,truncation=1,saveto=-1)

if __name__ == '__main__':
    demo_vis_kitti_images()
