
# coding: utf-8

# In[1]:

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mping
from skimage.transform import resize
from PIL import Image, ImageOps, ImageEnhance
import PIL
import random
import glob
import os, sys

get_ipython().magic(u'matplotlib inline')


# In[83]:

img = mping.imread('000012.jpg')
# img_ = np.array([img, img, img]).transpose(1,2,0)
print img.shape
# img = np.array([img, img, img])
plt.imshow(img)


# In[70]:

def rotation(image, prefix='example', save_path='./', state=False, num = 10, rotate=[90, 180, 270], suffix='.jpg'):
    img = Image.fromarray(image)
    background = np.mean(image[2,:])
    
    if state is not False:
        rotate = random.sample(range(1, 361),  num)
    for angle in rotate:
        img_ = img.rotate(angle)
        img_ = np.array(img_)
        img_[img_==0] = background
        img_ = Image.fromarray(img_)
        img_.save(os.path.join(save_path, prefix + '_rot_' + str(angle) + suffix))
        
    return


# In[71]:

def flip(image, prefix='example', save_path='./', choice=[True, True, True], suffix='.jpg'):
    img = Image.fromarray(image)
    filp = ['l2r', 't2b', 'comb']
    
    if choice[0]:
        img.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(save_path, prefix + '_flip_' + filp[0] + suffix))
    if choice[1]:
        img.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(save_path, prefix + '_flip_' + filp[1] + suffix))
    if choice[2]:
        img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(save_path, prefix + '_flip_' + filp[2] + suffix))
    
    return


# In[88]:

def translation(image, prefix='example', num=10, save_path='./', suffix='.jpg'):
    W, H = image.shape[:2]
    W_max = 0.2 * W
    H_max = 0.2 * H
    
    img = Image.fromarray(image)
    background = np.mean(image[2,:])
    a,b,c,d,e,f = 1,0,0,0,1,0 # no trans params
    for i in xrange(num):
        c = random.randint(10, int(W_max))
        f = random.randint(10, int(H_max))
        img_ = img.transform(img.size, Image.AFFINE, (a,b,c,d,e,f))
        
        img_ = np.array(img_)
        img_[img_==0] = background
        img_ = Image.fromarray(img_)
        img_.save(os.path.join(save_path, prefix + '_trans_' + str(i) + suffix))
    return


# In[73]:

def colorjitter(image, prefix='example', num=10, save_path='./', suffix='.jpg'):
    
    H, W, C = image.shape # only for color images
    threshold = np.max(image)
    
    for i in xrange(num):
        noise = np.random.randint(0,threshold,(H, W)) # design jitter/noise here
        jitter = np.zeros_like(image)
        jitter[:,:,random.randint(0,C-1)] = noise
        img = Image.fromarray(jitter+image)
        img.save(os.path.join(save_path, prefix + '_coljit_' + str(i) + suffix))
    return


# In[74]:

def colorjittergray(image, prefix='example', num=10, save_path='./', suffix='.jpg'):
    
    threshold = np.max(image)
    image = np.stack((image,)*3).transpose(1,2,0) # gray to color
    H, W, C = image.shape
    
    for i in xrange(num):
        noise = np.random.randint(0,threshold,(H, W)) # design jitter/noise here
        jitter = np.zeros_like(image)
        jitter[:,:,random.randint(0,C-1)] = noise
        jitter_mean = np.mean(jitter+image, axis=2)
        
        img = Image.fromarray(jitter_mean)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(os.path.join(save_path, prefix + '_coljit_' + str(i) + suffix))
    return


# In[75]:

def colorjitter3(image, prefix='example', num=6, save_path='./', suffix='.jpg'):
    
    H, W, C = image.shape # only for color images
    threshold = np.min([H,W])
    
    for i in xrange(num):
        R = image[:,:,0]
        G = image[:,:,1]
        B = image[:,:,2]
        RGBshifted = np.dstack((np.roll(R, random.randint(0, int(threshold*0.1)), axis=0), 
                                np.roll(G, random.randint(0, int(threshold*0.1)), axis=1), 
                                np.roll(B, -random.randint(0, int(threshold*0.1)), axis=0)))
        img = Image.fromarray(RGBshifted)
        img.save(os.path.join(save_path, prefix + '_coljit3_' + str(i) + suffix))
    return


# In[76]:

def colorjitter3gray(image, prefix='example', num=6, save_path='./', suffix='.jpg'):
    
    threshold = np.min(image.shape)
    image = np.stack((image,)*3).transpose(1,2,0) # gray to color
    H, W, C = image.shape
    for i in xrange(num):
        R = image[:,:,0]
        G = image[:,:,1]
        B = image[:,:,2]
        RGBshifted = np.dstack((np.roll(R, random.randint(0, int(threshold*0.1)), axis=0), 
                                np.roll(G, random.randint(0, int(threshold*0.1)), axis=1), 
                                np.roll(B, -random.randint(0, int(threshold*0.1)), axis=0)))
        img = Image.fromarray(np.mean(RGBshifted, axis=2))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(os.path.join(save_path, prefix + '_coljit3_' + str(i) + suffix))
    return


# In[77]:

rotation(img, state=True, num=3)


# In[78]:

flip(img)


# In[89]:

translation(img)


# In[94]:

colorjitter(img)


# In[92]:

colorjittergray(img)


# In[95]:

colorjitter3(img)


# In[66]:

colorjitter3gray(img)


# In[29]:

img_fold = '/home/mercury/Dropbox/Work/testdata/*/*'
img_type = '.jpg'


# In[31]:

img_path = glob.glob(os.path.join(img_fold + img_type))
print img_path


# In[51]:

for path in img_path:
    temp = path.split('/')[:-1]
    save_path = '/'.join(temp)
    prefix = path.split('/')[-1].split('.')[-2]
#     print save_path,prefix
    
    img = mping.imread(path)
    img_ = np.array([img, img, img]).transpose(1,2,0)
    
    ## jitter & save
    rotation(img, state=True, num=6, prefix=prefix, save_path=save_path, suffix=img_type)


# In[ ]:



