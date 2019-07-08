#Kinect eye tracking implementation following article:
#3D Eye-Tracking Method Using HD Face Model of Kinect v2 by Byoung Cheul Kim and Eui Chul Lee

import cv2
import numpy as np
import dlib

import math

# codice aggiuntivo fornito dalla issue #67 su: https://github.com/Kinect/PyKinect2/issues/67
# https://github.com/limgm/PyKinect2
import lib.utils_PyKinectV2 as utils

from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

#dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facepoints = [36,39,42,45] #face landmarks che ci interessano

#kinect 
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                         PyKinectV2.FrameSourceTypes_Depth |
                                         PyKinectV2.FrameSourceTypes_Infrared)

depth_width, depth_height = kinect.depth_frame_desc.Width, kinect.depth_frame_desc.Height # Default: 512, 424
color_width, color_height = kinect.color_frame_desc.Width, kinect.color_frame_desc.Height # Default: 1920, 1080


# Loop principale
while(True):
    # riceve i frame dal kinect
    if kinect.has_new_color_frame() and \
       kinect.has_new_depth_frame():
           
           color_frame = kinect.get_last_color_frame()
           depth_frame = kinect.get_last_depth_frame()
          
           # da utils:
           #########################################
           ### Reshape from 1D frame to 2D image ###
           #########################################
           color_img        = color_frame.reshape(((color_height, color_width, 4))).astype(np.uint8)
           depth_img        = depth_frame.reshape(((depth_height, depth_width))).astype(np.uint16) 
           
           ###############################################
           ### Useful functions in utils_PyKinectV2.py ###
           ###############################################
           
           # "allinea" il flusso RGB con quello depth 
           # Stesso risultato del Coordinate Mapper: 
           align_color_img = utils.get_align_color_image(kinect, color_img)
           
           ######################################
           ### Display 2D images using OpenCV ###
           ######################################
           color_img_resize = cv2.resize(color_img, (0,0), fx=0.5, fy=0.5) # Resize (1080, 1920, 4) into half (540, 960, 4)
           depth_colormap   = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=255/1500), cv2.COLORMAP_OCEAN) # Scale to display from 0 mm to 1500 mm
           
           
           # frame vuoti su cui "disegnare" il tracciamento
           blank_frame = np.zeros((414,512,3), np.uint8)
           blank_frame2 = np.zeros((1080,1920,3), np.uint8)
           
           
           gray = cv2.cvtColor(align_color_img, cv2.COLOR_BGR2GRAY)
           # dlib
           faces = detector(gray)
           
           # per ogni viso...
           for face in faces:
               x1 = face.left()
               y1 = face.top()
               x2 = face.right()
               y2 = face.bottom()
               
               landmarks = predictor(gray, face)
               
              
               ##### i landmark piu interno ed esterno dell'occhio sinistro #########
               cv2.circle(align_color_img, (landmarks.part(36).x, landmarks.part(36).y), 1, (0,255,0), -1)
               cv2.circle(align_color_img, (landmarks.part(39).x, landmarks.part(39).y), 1, (0,255,0), -1)
               ##### i landmark piu interno ed esterno dell'occhio destro ########
               cv2.circle(align_color_img, (landmarks.part(42).x, landmarks.part(42).y), 1, (0,255,0), -1)
               cv2.circle(align_color_img, (landmarks.part(45).x, landmarks.part(45).y), 1, (0,255,0), -1)
              
               
               # landmark dell'occhio sinistro
               left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                           (landmarks.part(37).x, landmarks.part(37).y),
                                           (landmarks.part(38).x, landmarks.part(38).y),
                                           (landmarks.part(39).x, landmarks.part(39).y),
                                           (landmarks.part(40).x, landmarks.part(40).y),
                                           (landmarks.part(41).x, landmarks.part(41).y)],
                                           np.int32)
               # landmark dell'occhio destro
               right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y), 
                                            (landmarks.part(43).x, landmarks.part(43).y),
                                            (landmarks.part(44).x, landmarks.part(44).y),
                                            (landmarks.part(45).x, landmarks.part(45).y),
                                            (landmarks.part(46).x, landmarks.part(46).y),
                                            (landmarks.part(47).x, landmarks.part(47).y)],
                                           np.int32)
               
               # frame isolato occhio sx 
               left_min_x = np.min(left_eye_region[:, 0]) - 5
               left_max_x = np.max(left_eye_region[:, 0]) + 5
               left_min_y = np.min(left_eye_region[:, 1]) - 5
               left_max_y = np.max(left_eye_region[:, 1]) + 5
               # frame isolato occhio dx
               right_min_x = np.min(right_eye_region[:, 0]) - 5
               right_max_x = np.max(right_eye_region[:, 0]) + 5
               right_min_y = np.min(right_eye_region[:, 1]) - 5
               right_max_y = np.max(right_eye_region[:, 1]) + 5
               
               # frame occhio destro
               right_eye = align_color_img[right_min_y: right_max_y, right_min_x: right_max_x]
               right_gray_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
               
               
               # frame occhio sinistro
               left_eye = align_color_img[left_min_y: left_max_y, left_min_x: left_max_x]
               left_gray_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
               
               
               # calcolo centro dell'occhio (punto medio)
               # SX
               left_eye_height, left_eye_width,_e = left_eye.shape
               x_left_eye_center = int(left_eye_width / 2)
               y_left_eye_center = int(left_eye_height / 2)
               # DX
               right_eye_height, right_eye_width,_e = right_eye.shape
               x_right_eye_center = int(right_eye_width / 2)
               y_right_eye_center = int(right_eye_height / 2)
               
               # disegna il centro dell'occhio sul frame 
               #cv2.circle(left_eye, (x_left_eye_center, y_left_eye_center) , 1 , (255,0,0) , -1)
               #cv2.circle(right_eye, (x_right_eye_center, y_right_eye_center) , 1 , (255,0,0) , -1)
               
               
               ##################################
               ##### DETECTION CENTRO DELLA PUPILLA ######
               ##################################
               kernel = np.ones((3,3), np.uint8)
               # occhio destro
               equ_right = cv2.equalizeHist(right_gray_eye)
               thres_right = cv2.inRange(equ_right,0,18) # ...dipende molto dalle condizioni di luce nell'ambiente di test
               dilation_right = cv2.dilate(thres_right,kernel,iterations = 1)
               erosion_right = cv2.erode(dilation_right,kernel,iterations = 2)
               
               # Ricerca di shape (per trovare l'iride)
               image_right, contours_right, hierarchy_right = cv2.findContours(erosion_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
               
               #debug = cv2.resize(image_right, (0,0), fx = 4, fy = 4 )
               #cv2.imshow("erosion" , debug)
               
               if(len(contours_right) != 0):
                   M = cv2.moments(contours_right[0])
                   if M['m00']!=0:
                           # Centro della pupilla
                           cx_right = int(M['m10']/M['m00'])
                           cy_right = int(M['m01']/M['m00'])
                           
                           # disegno sul frame originale
                           cv2.circle(align_color_img, ((right_min_x + cx_right),(right_min_y+cy_right)), 1, (0,0,255), -1)
                           # vettore tra centro dell'occhio e centro pupilla
                           cv2.line(align_color_img, (right_min_x + x_right_eye_center, right_min_y + y_right_eye_center), (right_min_x + cx_right, right_min_y+cy_right), (0, 255, 0) , 1)  
                           
                           ## RIPORTO LE COORDINATE SUL FRAME PRINCIPALE
                           # pupilla
                           cx_right = cx_right + right_min_x
                           cy_right = cy_right + right_min_y
                           #print("cx_right: ", cx_right , " cy_right: " , cy_right , " depth: " , z_iris_right)
                           
                           # centro dell'occhio
                           x_right_eye_center = x_right_eye_center + right_min_x
                           y_right_eye_center = y_right_eye_center + right_min_y
                           #print("x_eye_center_right: ", x_right_eye_center , " y_eye_center_right: " , y_right_eye_center, " depth" , z_center_right)
                 
                    
                           ####################################################################
                           ####### X e Y nelle coordinate della camera ( in mm) ##############
                           ###################################################################
                           # parametri intrinseci trovati qui: 
                           # github: https://github.com/shanilfernando/VRInteraction/blob/master/calibration/camera_param.yaml
                           
                           # principal point of the camera (depth)
                           p_x = 256.684
                           p_y = 207.085
                           # fov
                           fov_x = 366.193
                           fov_y = 366.193
    
                           
                           # centro della pupilla in coordinate mondo
                           # profondità pupilla
                           z_iris_right = depth_frame[(cy_right) * 512 +  (cx_right)]
                           #print("depth iris center " , z_iris_right)
                           
                           ## Formule trovate anche qui: https://stackoverflow.com/questions/12007775/to-calculate-world-coordinates-from-screen-coordinates-with-opencv
                           x_right_iris_world = ((cx_right - p_x) * z_iris_right) / fov_x
                           y_right_iris_world = ((cy_right - p_y) * z_iris_right) / fov_y
                           #print("IRIS CENTER world: " , x_right_iris_world , " , ", y_right_iris_world , " mm and depth: " , z_iris_right , " mm" )
                            
                           
                           # Centro di rotazione dell'occhio in coordinate mondo
                           # profondità
                           z_center_right = depth_frame[(y_right_eye_center) * 512 +  (x_right_eye_center)]
                           z_center_right = z_center_right + 13.5 #eyeball center è 13.5 mm piu indietro rispetto al punto medio
                           # X e Y
                           x_right_eye_center_world = ((x_right_eye_center - p_x) * z_center_right) / fov_x
                           y_right_eye_center_world = ((y_right_eye_center - p_y) * z_center_right) / fov_y
                           #print("EYE center world: x: " , x_right_eye_center_world, " , y: " ,y_right_eye_center_world , " depth: " , z_center_right)
                           
                            
                           ##### CALCOLO GAZE TRACKING ###############
                           # Offset kinect rispetto allo 0,0 dello schermo
                           Dx = 170 # mm
                           Dy = 80 # mm
                           
                           # Equazioni paper
                           # (xr,yr,zr) - (xi,yi,zi)
                           x_vector = x_right_eye_center_world - x_right_iris_world
                           y_vector = y_right_eye_center_world - y_right_iris_world
                           z_vector = z_center_right - z_iris_right
                           #print("gaze vector: " , x_vector, y_vector, z_vector)
                           
                           # numeratori equazioni
                           x_num = (-1*z_iris_right) * (x_vector)
                           y_num = (-1*z_iris_right) * (y_vector)
                           den = z_center_right - z_iris_right
                           
                           # Equazione principali:
                           x_vector_gaze = x_right_iris_world + (x_num / z_vector)
                           y_vector_gaze = y_right_iris_world + (y_num / z_vector)
                           #print("xc, yc (mm): " , x_vector_gaze, y_vector_gaze)
                           # queste sono ancora nel camera space coordinates rispetto al principal point del kinect (centro)
                           
                           
                           # Calibrazione geometrica (ultimo passaggio)
                           x_d = x_vector_gaze
                           y_d = y_vector_gaze + Dy
                           
                           # Visualizzazione su display
                           ### https://www.scratchapixel.com/lessons/3d-basic-rendering/computing-pixel-coordinates-of-3d-point/mathematics-computing-2d-coordinates-of-3d-points ###
                           
                           # Abbiamo il punto di intersezione con il canvas della camera
                           # Bisogna convertire da Screen-space a Raster-Space (da punto in coordinate mondo a pixel)
                           
                           # la dimensione del canvas (in mm? in px? --> sul sito non viene specificato)
                           # assumo che sia la width e la height in mm entro i quali il kinect ha la visuale
                           kwidth_mm = 481     # in mm (481 = valore trovato sperimentalmente)
                           kheight_mm = 322.5   # in mm (322.5 = trovato sperimentalmente)
                           ScreenWidth = 1920
                           ScreenHeight = 1080
                           
                           # se è visibile al Kinect..
                           if(abs(x_d) <= kwidth_mm/2 or abs(y_d) <= kheight_mm/2):
                               # visibile
                               # Spazio NDC (Normalized Device Coordinate)
                               Pnormalized_x = ((x_d + (kwidth_mm/2)) / kwidth_mm)
                               Pnormalized_y = ((y_d + (kheight_mm/2)) / kheight_mm)
                               #print("Normalized: " , Pnormalized_x, Pnormalized_y)
                               
                               # questo  è il pixel in cui sta guardando
                               Praster_x = math.floor((Pnormalized_x * ScreenWidth))
                               Praster_y = math.floor((Pnormalized_y * ScreenHeight))
                               print("Raster: " , Praster_x, " ", Praster_y)
                               
                               
                               # disegna sul blank frame il gaze point
                               cv2.circle(blank_frame2, (Praster_x, Praster_y), 8, (0,0,255), -1)
                           
           
           #**************************
           # mostra i frame in OpenCv
           #**************************
           
           
           ##### reference points ################################
           cv2.circle(blank_frame2, (320, 270), 4, (0,255,0) , -1)
           cv2.circle(blank_frame2, (320, 810), 4, (0,255,0) , -1)
           cv2.circle(blank_frame2, (960, 540), 4, (0,255,0) , -1)
           cv2.circle(blank_frame2, (1600, 270), 4, (0,255,0) , -1)
           cv2.circle(blank_frame2, (1600, 810), 4, (0,255,0) , -1)
           ########################################################
           
           # frame su schermo intero
           cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
           cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
           cv2.imshow("window", blank_frame2)
           
           # scalo il frame principale per vedere meglio
           align_color_img = cv2.resize(align_color_img, (0,0), fx = 2, fy = 2)
           cv2.imshow("Color and Depth aligned" , align_color_img)
           
           #right_eye = cv2.resize(right_eye, (0,0), fx = 4, fy = 4)
           #cv2.imshow("occhio destro" , right_eye)
           
           #cv2.imshow("Color", color_img_resize)
           #cv2.imshow("Depth" , depth_colormap)
           
           
           # PREMERE ESC PER USCIRE DAL LOOP
           key = cv2.waitKey(30)
           if key==27:
               break
kinect.close()
cv2.destroyAllWindows()
           