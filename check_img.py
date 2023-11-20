python
import cv2
import numpy as np
import os

class ImageProcessor:
    def __init__(self, path, train_file):
        self.path = path
        self.train_file = train_file
        self.num = 0

    def process_images(self):
        result = os.listdir(self.path)
        if not os.path.exists(self.train_file):
            os.mkdir(self.train_file)
        for i in result:
            try:
                image = cv2.imread(self.path + '/' + i)
                cv2.imwrite(self.train_file + '/' + 'Compressed' + i, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                self.num += 1
            except:
                pass
        print('数据有效性验证完毕,有效图片数量为 %d' % self.num)
        if self.num == 0:
            print('您的图片命名有中文，建议统一为1（1）.jpg/png')
