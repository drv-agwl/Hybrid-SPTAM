#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import os
import os
import glob
import random
import numpy as np

# ## Corner convention : in order TL, TR, BR, BL (clockwise)
# ## BBox convention : (xtl,ytl),(xbr,ybr)
#
# # Note : Currently corners are store wrt Frame, for tracker change wrt to bbox
#

# In[9]:


img_ptr = 0
FBF_history = {}
UID_history = {}
def bbox2whxy(bbox):
    w = bbox[2] - bbox[0] + 6
    h = bbox[3] - bbox[1] + 6
    y = bbox[1]
    x = bbox[0]
    return (w,h,x,y)

def bbox2xys(bbox):  # clockwise : TL...
    w,h,x,y = bbox2whxy(bbox)
    return [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]

def load_next(): # load image one by one

    global img_ptr
    img_ptr += 1
    name = frames_lst[img_ptr][:-3] + 'png'
    #print(name)
    img = cv2.imread('../Dataset-tl/B/' + name)
    tree = ET.parse(path + 'xmls/' +frames_lst[img_ptr])
    root = tree.getroot()
    xamal = []
    for member in root.findall('object'):
        value = {
                'name' : member[0].text,
                'bbox' : ( int(member[4][0].text), int(member[4][1].text), int(member[4][2].text), int(member[4][3].text))
                }
        xamal.append(value)

    return [img,xamal]

def historical_interpolation():
    pass



def find_missing_corners(TL,TR,BR,BL,name,bbox,debug = False): # Heuristics based corner imputation
    miss = 0
    # corners are -1 if not detected, there wont be a case where some are -1 and others are detected
    if(TL == -1):
        miss += 1
    if(TR == -1):
        miss += 1
    if(BL == -1):
        miss += 1
    if(BR == -1):
        miss += 1

    else:
        # More heuristics can be added here on the go
        if(name == 'door_knob' or name == 'window'):   #knobs need more exammination, skipped for now
                                                       #windows are predicted good, heuristic is only ruining it
            return (TL,TR,BR,BL)

        # check if corner lie in some threshold distance from edge
        w,h,_,_ = bbox2whxy(bbox)
        hth = 0.2      # horizontal threshold
        vth = 0.2      # vertical threshold
        if(TL[0]*1.0 > (hth)*w or TL[1]*1.0 > (vth)*h):
            #if(debug):
            #    print('TL-out',TL,(hth)*w,(vth)*h)
            TL = -1
        if(TR[0]*1.0 < (1-hth)*w or TR[1]*1.0 > (vth)*h):
            #if(debug):
            #    print('TR-out',TR,(1-hth)*w,(vth)*h)
            TR = -1
        if(BR[0]*1.0 < (1-hth)*w or BR[1]*1.0 < (1-vth)*h):
            #if(debug):
            #    print('BR-out',BR,(1-hth)*w,(1-vth)*h)
            BR = -1
        if(BL[0]*1.0 > (hth)*w or BL[1]*1.0 < (1-vth)*h):
            #if(debug):
            #    print('BL-out',BL,(hth)*w,(1-vth)*h)
            BL = -1

        #calculate avg left margin
        avgLM = 0
        tmp = 0
        if(TL != -1):
            avgLM += TL[0]
            tmp += 1
        if(BL != -1):
            avgLM += BL[0]
            tmp += 1
        if(tmp == 0):
            avgLM = (hth)*w
        else :
            avgLM /= tmp
        #calculate avg right margin
        avgRM = 0
        tmp = 0
        if(TR != -1):
            avgRM += TR[0]
            tmp += 1
        if(BR != -1):
            avgRM += BR[0]
            tmp += 1
        if(tmp == 0):
            avgRM = (1-hth)*w
        else :
            avgRM /= tmp
        #calculate avg top margin
        avgTM = 0
        tmp = 0
        if(TL != -1):
            avgTM += TL[1]
            tmp += 1
        if(TR != -1):
            avgTM += TR[1]
            tmp += 1
        if(tmp == 0):
            avgTM = (vth)*h
        else :
            avgTM /= tmp
        #calculate avg bottom margin
        avgBM = 0
        tmp = 0
        if(BL != -1):
            avgBM += BL[1]
            tmp += 1
        if(BR != -1):
            avgBM += BR[1]
            tmp += 1
        if(tmp == 0):
            avgBM = (1-vth)*h
        else :
            avgBM /= tmp

        #impute missing vals acc to margins
        if(TL == -1):
            TL = (avgLM, avgTM)
        if(TR == -1):
            TR = (avgRM, avgTM)
        if(BR == -1):
            BR = (avgRM, avgBM)
        if(BL == -1):
            BL = (avgLM, avgBM)

        return (TL,TR,BR,BL)

def get_corn(img, name, bbox ,plot = False): # Returns corners to store in the log

    TL,TR,BR,BL = findCorn(img)
    w,h,x,y = bbox2whxy(bbox)

    #code to plot image as processed
    if(plot):
        #print(w,h)

        final_corners = []
        if(TL != -1):
            final_corners.append([TL[0],TL[1],'TL'])
        if(TR != -1):
            final_corners.append([TR[0],TR[1], 'TR'])
        if(BR != -1):
            final_corners.append([BR[0],BR[1], 'BR'])
        if(BL != -1):
            final_corners.append([BL[0],BL[1], 'BL'])

        #plt.scatter(x=[i[0] for i in final_corners], y=[i[1] for i in final_corners], c='r', s=100)
        # for x,y,s in final_corners:
            #plt.text(x, y, s, bbox=dict(facecolor='red', alpha=0.5))


    #find missing corners
    '''try:
        TL,TR,BR,BL = find_missing_corners(TL,TR,BR,BL,name,bbox)
    except :
        #print('NoneType by FMC')
        TL,TR,BR,BL = bbox2xys(bbox)


    if(plot):

        final_corners = []
        if(TL != -1):
            final_corners.append([TL[0],TL[1],'TL'])
        if(TR != -1):
            final_corners.append([TR[0],TR[1], 'TR'])
        if(BR != -1):
            final_corners.append([BR[0],BR[1], 'BR'])
        if(BL != -1):
            final_corners.append([BL[0],BL[1], 'BL'])

#         fig,ax = #plt.subplots(1)
#         ax.imshow(img)
#         rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
#         ax.add_patch(rect)

        #plt.scatter(x=[i[0] for i in final_corners], y=[i[1] for i in final_corners], c='r', s=100)
        # for x,y,s in final_corners:
        #     #plt.text(x, y, s, bbox=dict(facecolor='blue', alpha=0.5))

        # try:
        #     #plt.imshow(img)
        #     #plt.show()
        # except:
        #     pass

    #validate corners

        #To be done if required
        '''
    #the corners were wrt to their bbox, converting them to wrt frame
    w,h,x,y = bbox2whxy(bbox)
    w = w - 3
    h = h - 3
    if(TL == -1):
        TL = (x , y )
    else :
        TL = (x + TL[0], y + TL[1])

    if(TR == -1):
        TR = (x + w , y)
    else:
        TR = (x + TR[0], y + TR[1])

    if(BR == -1):
        BR = (x + w, y + h)
    else:
        BR = (x + BR[0], y + BR[1])

    if(BL == -1):
        BL = (x , y + h)
    else:
        BL = (x + BL[0], y + BL[1])




    return (TL,TR,BR,BL)

# Get unique ID from World coordinates
def get_uid(obj):
    return random.randint(1,100)   # placeholder till WC works

def findCorn(frame):    # Corner Detector

    try :
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([30,150,50])
        upper_red = np.array([255,255,180])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame,frame, mask= mask)
    except:
        return [-1,-1,-1,-1]

    edges = cv2.Canny(frame,50,200)

    x = []
    y = []

    for i in range(len(edges)):
        for j in range(len(edges[0])):
            if(edges[i][j] == 255):
                x.append((i,j))

    if(len(x) == 0) :
        return [-1,-1,-1,-1]


    for i in x:
        frame[i[0]][i[1]] = 255

    x = np.asarray(x)
    dist_TL = x[np.argmin(np.sum((x - [0,0])**2,axis = 1))]
    dist_TR = x[np.argmin(np.sum((x - [0,frame.shape[1]])**2,axis = 1))]
    dist_BL = x[np.argmin(np.sum((x - [frame.shape[0],0])**2,axis = 1))]
    dist_BR = x[np.argmin(np.sum((x - [frame.shape[0],frame.shape[1]])**2,axis = 1))]
    # we need w,h
    return [dist_TL.tolist()[::-1], dist_TR.tolist()[::-1], dist_BR.tolist()[::-1], dist_BL.tolist()[::-1]]

def process(img, obj, plot = True):     # Load detected objects one by one and log them
    global FBF_history
    global UID_history


    uid = get_uid(obj)

#     if img_ptr not in FBF_history:
#         FBF_history[img_ptr] = {}

#     FBF_history[img_ptr][uid] = [ (obj['name'],(get_corn( img ,obj['name'], obj['bbox'], plot=plot ) )) ]

    if uid not in UID_history:
        UID_history[uid] = {}

    UID_history[uid][img_ptr] = [ (obj['name'],(get_corn( img ,obj['name'], obj['bbox'], plot = plot ) )) ]

    return UID_history[uid][img_ptr][0][1]



# In[10]:


# # main loop
# frames_lst = os.listdir('../Dataset-tl/A/xmls')
# img_ptr = -1
# FBF_history = {}   # Frame by Frame History
# UID_history = {}   # History of Unique ID, basically of every object in world
# import pprint
# for i in range(2):
#     img,objs = load_next()
#     coords = []
#     for j,obj in enumerate(objs):
#         w = obj['bbox'][2] - obj['bbox'][0]
#         h = obj['bbox'][3] - obj['bbox'][1]
#         y = obj['bbox'][1]
#         x = obj['bbox'][0]
#         crop_img = img[y-3:y+h+3, x-3:x+w+3]
#         #cv2.imwrite('./objs/' + str(img_ptr) +  str(j) + '.png', crop_img )
#         c = process(crop_img,obj, False)
#         coords.append(c[0])
#         coords.append(c[1])
#         coords.append(c[2])
#         coords.append(c[3])

# #     #plt.figure(figsize = (100,120))
# #     #plt.scatter(x=[i[0] for i in coords], y=[i[1] for i in coords], c='r', s=1000)
# #     #plt.imshow(img)
# #     #plt.show()


# pp = pprint.PrettyPrinter(indent = 4)
# #print(pp.pprint(FBF_history))
# #print(pp.pprint(UID_history))


# In[23]:


def find_all_corners_from_bboxs(bbox,left=True):
    global img_ptr
    corners = []

    # if left:
    #     img = cv2.imread('/home/mehul/Downloads/seq22/00/image_0/'+imgName)
    # else:
    #     img = cv2.imread('/home/mehul/Downloads/seq22/00/image_1/'+imgName)

    o1=1
    for obj in bbox:
        # print(obj[0])
        c = []
        # w = int(obj[3] - obj[1])
        # h = int(obj[4] - obj[2])
        # y = int(obj[2])
        # x = int(obj[1])
        x1,y1 = obj[1],obj[2]
        x2,y2 = obj[3],obj[2]
        x3,y3 = obj[1],obj[4]
        x4,y4 = obj[3],obj[4]

        # objd = {'name' : obj[0], 'bbox' : (obj[1],obj[2],obj[3],obj[4])}
        c.append(obj[0])
        c.append([x1,y1])
        c.append([x2,y2])
        c.append([x3,y3])
        c.append([x4,y4])
        
        # crop_img = img[y-3:y+h+3, x-3:x+w+3]
        # c.extend(list(get_corn( crop_img ,objd['name'], objd['bbox'])))
        corners.append(c)
        '''if left:
            img2=img.copy()
            img2=cv2.rectangle(img2, (x,y), (x+w,y+h), (255,0,0), 3)
            img2=cv2.circle(img2, (int(c[1][0]),int(c[1][1])), 10, (255,0,0), 3)
            img2=cv2.circle(img2, (int(c[2][0]),int(c[2][1])), 10, (255,0,0), 3)
            img2=cv2.circle(img2, (int(c[3][0]),int(c[3][1])), 10, (255,0,0), 3)
            img2=cv2.circle(img2, (int(c[4][0]),int(c[4][1])), 10, (255,0,0), 3)
            cv2.imwrite('/home/destro/kala semant/sptam/training/boxes/'+imgName[:-4]+"_"+str(o1)+".png",img2)
        o1=o1+1'''
    # print("Tracker Output: ", corners)
    return corners





