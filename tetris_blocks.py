import cv2
import sys
import numpy as np

image = cv2.imread('/home/enes/Documents/github_repos/Computer_Vision_Codes/Enes Satıcı - 23 Ara 2025, 22_02.jpg')
image_rgb = cv2.cvtColor(image,cv2.COLOR_BAYER_BG2BGR)

lower_red =np.array([150,0,0])
upper_red = np.array([255,80,80])

