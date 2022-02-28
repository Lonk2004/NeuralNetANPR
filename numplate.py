import matplotlib.pyplot as plt 
import cv2
import numpy as np

class Plate:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.numplate = " "
 

    def blur_images(self):
        self.img2 = cv2.bilateralFilter(self.img2, 11, 90, 90)

    def outline_edges(self):
        self.img2 = cv2.Canny(self.img2, 30,200)
        
    
    def reduce_edges(self):
        self.cnts, new = cv2.findContours(self.img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = sorted(self.cnts, key=cv2.contourArea, reverse=True)[:40]
        self.img2 = cv2.drawContours(self.img2, self.cnts, -1, (255,0,255), 2)

    def outline_numberplate(self):
        for c in self.cnts: 
            self.perimeter = cv2.arcLength(c, True)
            self.edges_count = cv2.approxPolyDP(c, 0.02 * self.perimeter, True)
            if len(self.edges_count) == 4: 
                x,y,w,h = cv2.boundingRect(c)
                self.numplate = self.img1[y:y+h, x:x+w]
        
    def extract(self):
        self.blur_images()
        self.outline_edges()
        self.reduce_edges()
        self.outline_numberplate()


    def getnumplate(self):
        return self.numplate



for i in range(0, 11):
    path = r'D:\Downloads\archive\crimages\cars' + str(i) +'.png'
    picture = cv2.imread(path, 0)
    picturecolor = cv2.imread(path, 1)
    display = Plate(picturecolor, picture)
    newpath = r'D:\Downloads\archive\imagesextrct\cars' + str(i) +'.png'
    display.extract()
    numplate = display.getnumplate()
    if numplate != " ":
        cv2.imwrite(newpath, numplate)
    print (i)