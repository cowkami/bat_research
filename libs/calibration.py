import numpy as np
import cv2
import matplotlib.pyplot as plt


from constants import *


calib_imgs_path = raw_movie_190526_path / 'calib2' / 'calib2_001'
calib_img2_path = calib_imgs_path / 'calib2_001_NX8-S1 Camera(02)/NX8-S1 Camera-000001.bmp'
calib_img3_path = calib_imgs_path / 'calib2_001_NX8-S1 Camera(03)/NX8-S1 Camera-000001.bmp'


class Calibration(objcet):

    def __init__(self, img_name,img_path):
        self.img_name = img_name
        self.img = cv2.imread(str(img_path), 0)
        self.points = []


    def __onclick__(self, click):
        self.points = (click.xdata, click.ydata)
        return self.points


    def get_coors(self):
        ax = plt.gca()
        fig = plt.gcf()
        implot = ax.imshow(selfimg)
        cid = fig.cavas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        while len(self.points) <= 8:



        return self.points


cali = Calibration('calib2', calib_img2_path)
points = cali.get_coors()
print(points)

