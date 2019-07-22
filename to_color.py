import cv2
import numpy as np

color_map=([128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32],[0,0,0])
color_map2=([128,64,128],[232,35,244],[70,70,70],[153,153,190],[153,153,153],[0,220,200],[35,142,107],[152,251,152],[180,130,70],[60,20,220],[142,0,0],[70,0,0],[32,11,119])
color_map3=([128,64,128],[244,35,232],[70,70,70],[190,153,153],[153,153,153],[200,220,0],[107,142,35],[152,251,152],[70,130,180],[220,20,60],[0,0,142],[0,0,70],[119,11,32])
def cloor(image):
    res=np.zeros((1024,1024,3))
    for i in range(1024):
        for j in range(1024):
            res[i][j]=np.array(color_map[image[i][j]])
    return res

def cloor2(image):
    res=np.zeros((768,768,3))
    for i in range(768):
        for j in range(768):
            res[i][j]=np.array(color_map2[int(image[i][j])])
    return res
