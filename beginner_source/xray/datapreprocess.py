from collections import  defaultdict
from PIL import Image
import numpy as np
class TextBoxReuslt(object):
    def __init__(self, text_file):

        self.result_filename = defaultdict(list)

        with open(text_file,'r') as f:
            _all = f.readlines()
            for _line in _all:
                _line_result = _line.split(',')
                #[filename, x1,y1,x2,y1,x2,y2,x1,y2]

                self.result_filename[_line_result[0]].append(list(map(lambda x:  int(x), _line_result[1:])))

            # print(len(_all),self.result_filename)
    def __calculate_crop_bbox(self,text_boxes, im_height, im_width):
        if len(text_boxes)==0:
            return None
        _boxes = np.array(text_boxes)
        # filter the text boxes that are at top
        #                       extract min y value             greater than half of the image height
        _boxes_at_bottom_half = np.min(_boxes[:,1::2],axis=1)>(im_height/2)
        _boxes_remains = _boxes[_boxes_at_bottom_half]

        if len(_boxes_remains) == 0:
            return None
        else:
            # found min y in all remaining box and set that as crop anchor.
            min_y = np.min(_boxes_remains[:,1::2])
            return [0, 0, im_width, min_y]


    def crop_image(self, image:Image, image_name:str):

        _img_text_boxes = self.result_filename.get(image_name,[])
        _width, _height = image.size
        _crop_bbox = self.__calculate_crop_bbox(_img_text_boxes,_height,_width)
        if _crop_bbox is None:
            return  image
        _croped_image = image.crop(_crop_bbox)
        # close oraginal image
        image.close()
        return _croped_image


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    t =TextBoxReuslt('/data2/qilei_chen/DATA/xray/result_box_coordinates.txt')
    for i in range(len(t.result_filename)):
        file_name = list(t.result_filename.keys())[i]
        test_image = Image.open(f'/data2/qilei_chen/DATA/xray/xray_images/{file_name}')
        croped = t.crop_image(test_image,file_name)
        croped.save(f'/data2/qilei_chen/DATA/xray/xray_images1/{file_name}')
        #plt.imshow(croped)
        #plt.show()
        #croped.show()