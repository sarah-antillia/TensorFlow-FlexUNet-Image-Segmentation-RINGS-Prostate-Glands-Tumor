# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2024/12/23 ImageMaskDatasetGenerator.py

import os
import sys
import io
import shutil
import glob
import nibabel as nib
import numpy as np
from PIL import Image, ImageOps
import traceback
import math
from scipy.ndimage.interpolation import map_coordinates

from scipy.ndimage.filters import gaussian_filter
import cv2

import numpy as np
import cv2


class ImageMaskDatasetGenerator:

  def __init__(self, 
               images_dir  = "", 
               glands_masks_dir  = "", 
               tumor_masks_dir = "",
               output_dir = "", 
               angle      = 0, 
               resize     = 512,
               augmentation=True):
    
    self.images_dir = images_dir
    self.glands_masks_dir  = glands_masks_dir
    self.tumor_masks_dir  = tumor_masks_dir

    
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    self.output_images_dir = os.path.join(output_dir, "images")
    self.output_masks_dir  = os.path.join(output_dir, "masks")

    if not os.path.exists(self.output_images_dir):
      os.makedirs(self.output_images_dir)

    if not os.path.exists(self.output_masks_dir):
      os.makedirs(self.output_masks_dir)
    self.angle   = 0


    self.RESIZE    = (resize, resize)
    self.seed = 137
    self.W = resize
    self.H = resize
     
    self.augmentation = augmentation
    if self.augmentation:
      self.hflip    = False
      self.vflip    = False
      self.rotation = False
      #self.ANGLES   = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
      self.ANGLES   = [90, 180, 270]

      self.deformation=True
      self.alpha    = 1300
      #self.sigmoids = [8, 10,]
      self.sigmoids = [8,]
          
      self.distortion=True
      self.gaussina_filer_rsigma = 40
      self.gaussina_filer_sigma  = 0.5
      #self.distortions           = [0.02, 0.03,]
      self.distortions           = [0.02, ]
      self.rsigma = "sigma"  + str(self.gaussina_filer_rsigma)
      self.sigma  = "rsigma" + str(self.gaussina_filer_sigma)
      
      self.resize = False
      self.resize_ratios = [0.7, 0.8, 0.9]

      self.barrel_distortion = True
      self.radius     = 0.3
      self.amounts    = [0.3]
      self.centers    = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)]

      self.pincushion_distortion= True
      self.pincradius  = 0.3
      self.pincamounts = [-0.3]
      self.pinccenters = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)]

  #    color = self.bgr_color_map[pathology]
  #    mask  = self.colorize_mask(mask, color=color, gray=255)
  def colorize_mask(self, mask, color=(255, 255, 255), gray=0):
    h, w = mask.shape[:2]
    rgb_mask = np.zeros((w, h, 3), np.uint8)
    #condition = (mask[...] == gray) 
    condition = (mask[...] >= gray-10) & (mask[...] <= gray+10)   
    rgb_mask[condition] = [color]  
    return rgb_mask   

  def generate(self):
    glands_masks_files = glob.glob(self.glands_masks_dir + "/*.png")
    tumor_masks_files  = glob.glob(self.tumor_masks_dir + "/*.png")
    image_files        = glob.glob(self.images_dir + "/*.png")
    glands_masks_files = sorted(glands_masks_files)
    tumor_masks_files  = sorted(tumor_masks_files)
    image_file         = sorted(image_files)

    l1 = len(tumor_masks_files)
    l2 = len(image_files)
    print("--- l1: {} l2: {}".format(l1, l2))
    if l1 != l2:
      raise Exception("Unmatched number of seg_files and image_files ")
    for i in range(l1):
      print("--index {}".format(i))
      rc = self.generate_mask_files(tumor_masks_files[i], glands_masks_files[i], i, self.output_masks_dir) 
      if rc:
        self.generate_files(image_files[i],  i, self.output_images_dir, mask=False) 

  def resize_to_square(self, image, mask=True):
    h, w = image.shape[:2]
    RESIZE = h
    if w > h:
      RESIZE = w
    # 1. Create a black background
    if mask:
      background = np.zeros((RESIZE, RESIZE, 3),  np.uint8) 
    else:
      pixel = image[10][10]
      background = np.ones((RESIZE, RESIZE, 3),  np.uint8) * pixel 
    x = int((RESIZE - w)/2)
    y = int((RESIZE - h)/2)
    # 2. Paste the image to the background 
    background[y:y+h, x:x+w] = image
    # 3. Resize the background to (512x512)
    resized = cv2.resize(background, (self.W, self.H))

    return resized

  def normalize(self, image):
    min = np.min(image)/255.0
    max = np.max(image)/255.0
    scale = (max - min)
    if scale == 0:
      scale +=  1
    image = (image -min) / scale
    image = image.astype('uint8') 
    return image

  def generate_mask_files(self, tumor_file, glands_mask_file, index, output_dir):
     # GBR
     # Green mask

     tumor_mask  = cv2.imread(tumor_file)
     tumor_mask  = cv2.cvtColor(tumor_mask, cv2.COLOR_BGR2GRAY)
     if not tumor_mask.any() >0:
        glands_mask  = cv2.imread(glands_mask_file)
        glands_mask  = cv2.cvtColor(glands_mask, cv2.COLOR_BGR2GRAY)
        if not glands_mask.any() >0:
          print("   Skipped an empty mask")
          return False
        else:
          glands_mask  = cv2.resize(glands_mask, (self.RESIZE))
          glands_mask  = self.colorize_mask(glands_mask,  color=(0, 255, 0),  gray= 255)
          basename = os.path.basename(glands_mask_file)
          #basename = str(index)  + ".jpg" 
          filepath = os.path.join(output_dir, basename)
          cv2.imwrite(filepath, glands_mask)
          if self.augmentation:
           self.augment(glands_mask, basename, output_dir, border=(0, 0, 0), mask=True)
          return True

     tumor_mask  = cv2.resize(tumor_mask, (self.RESIZE))

     tumor_mask  = self.colorize_mask(tumor_mask,  color=(0, 0, 255),  gray= 255)

     #image = self.resize_to_square(image, mask=mask)
     basename = os.path.basename(tumor_file)
     #basename = str(index)  + ".jpg" 
     filepath = os.path.join(output_dir, basename)
     cv2.imwrite(filepath, tumor_mask)
     if self.augmentation:
       self.augment(tumor_mask, basename, output_dir, border=(0, 0, 0), mask=True)
     return True

  def generate_files(self, image_file, index, output_dir, mask=False):
     image = cv2.imread(image_file)
     image = cv2.resize(image, (self.RESIZE))
     #image = self.resize_to_square(image, mask=mask)
     basename = os.path.basename(image_file)
     #basename = str(index)  + ".jpg" 
     filepath = os.path.join(output_dir, basename)
     cv2.imwrite(filepath, image)
     if self.augmentation:
       self.augment(image, basename, output_dir, border=(0, 0, 0), mask=mask)


  def augment(self, image, basename, output_dir, border=(0, 0, 0), mask=False):
    border = image[2][2].tolist()
  
    print("---- border {}".format(border))
    if self.hflip:
      flipped = self.horizontal_flip(image)
      output_filepath = os.path.join(output_dir, "hflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.vflip:
      flipped = self.vertical_flip(image)
      output_filepath = os.path.join(output_dir, "vflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.rotation:
      self.rotate(image, basename, output_dir, border)

    if self.deformation:
      self.deform(image, basename, output_dir)

    if self.distortion:
      self.distort(image, basename, output_dir)

    if self.resize:
      self.shrink(image, basename, output_dir, mask)

    if self.barrel_distortion:
      self.barrel_distort(image, basename, output_dir)


  def horizontal_flip(self, image): 
    print("shape image {}".format(image.shape))
    if len(image.shape)==3:
      return  image[:, ::-1, :]
    else:
      return  image[:, ::-1, ]

  def vertical_flip(self, image):
    if len(image.shape) == 3:
      return image[::-1, :, :]
    else:
      return image[::-1, :, ]

  def rotate(self, image, basename, output_dir, border):
    for angle in self.ANGLES:      
      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

      rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(self.W, self.H), borderValue=border)
      output_filepath = os.path.join(output_dir, "rotated_" + str(angle) + "_" + basename)
      cv2.imwrite(output_filepath, rotated_image)
      print("--- Saved {}".format(output_filepath))

  def deform(self, image, basename, output_dir): 
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    random_state = np.random.RandomState(self.seed)

    shape = image.shape
    print("--- shape {}".format(shape))

    for sigmoid in self.sigmoids:
      dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      #dz = np.zeros_like(dx)

      x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
      indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

      deformed_image = map_coordinates(image, indices, order=1, mode='nearest')  
      deformed_image = deformed_image.reshape(image.shape)

      image_filename = "deformed" + "_alpha_" + str(self.alpha) + "_sigmoid_" +str(sigmoid) + "_" + basename
      image_filepath  = os.path.join(output_dir, image_filename)
      cv2.imwrite(image_filepath, deformed_image)

  # This method is based on the code of the following stackoverflow.com webstie:
  # https://stackoverflow.com/questions/41703210/inverting-a-real-valued-index-grid/78031420#78031420
  def distort(self, image, basename, output_dir):
    shape = (image.shape[1], image.shape[0])
    (w, h) = shape
    xsize = w
    if h>w:
      xsize = h
    # Resize original img to a square image
    resized = cv2.resize(image, (xsize, xsize))
    shape   = (xsize, xsize)
 
    t = np.random.normal(size = shape)
    for size in self.distortions:
      filename = "distorted_" + str(size) + "_" + self.sigma + "_" + self.rsigma + "_" + basename
      output_file = os.path.join(output_dir, filename)    
      dx = gaussian_filter(t, self.gaussina_filer_rsigma, order =(0,1))
      dy = gaussian_filter(t, self.gaussina_filer_rsigma, order =(1,0))
      sizex = int(xsize*size)
      sizey = int(xsize*size)
      dx *= sizex/dx.max()
      dy *= sizey/dy.max()

      image = gaussian_filter(image, self.gaussina_filer_sigma)

      yy, xx = np.indices(shape)
      xmap = (xx-dx).astype(np.float32)
      ymap = (yy-dy).astype(np.float32)

      distorted = cv2.remap(resized, xmap, ymap, cv2.INTER_LINEAR)
      distorted = cv2.resize(distorted, (w, h))
      cv2.imwrite(output_file, distorted)
      print("=== Saved distorted image file{}".format(output_file))

  def shrink(self, image, basename, output_dir, mask):
    print("----shrink shape {}".format(image.shape))
    h, w    = image.shape[0:2]
    pixel   = image[2][2]
    for resize_ratio in self.resize_ratios:
      rh = int(h * resize_ratio)
      rw = int(w * resize_ratio)
      resized = cv2.resize(image, (rw, rh))
      h1, w1  = resized.shape[:2]
      y = int((h - h1)/2)
      x = int((w - w1)/2)
      # black background
      background = np.zeros((w, h, 3), np.uint8)
      if mask == False:
        # white background
        background = np.ones((h, w, 3), np.uint8) * pixel
      # paste resized to background
      print("---shrink mask {} rsized.shape {}".format(mask, resized.shape))
      background[x:x+w1, y:y+h1] = resized
      filename = "shrinked_" + str(resize_ratio) + "_" + basename
      output_file = os.path.join(output_dir, filename)    

      cv2.imwrite(output_file, background)
      print("=== Saved shrinked image file{}".format(output_file))

  # This method is based on the code in the following stackoverflow.com website:
  # https://stackoverflow.com/questions/59776772/python-opencv-how-to-apply-radial-barrel-distortion
  def barrel_distort(self, image, basename, output_dir):    
    (h,  w,  _) = image.shape

    # set up the x and y maps as float32
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    scale_x = 1
    scale_y = 1
    index   = 1000
    for amount in self.amounts:
      for center in self.centers:
        index += 1
        (ox, oy) = center
        center_x = w * ox
        center_y = h * oy
        radius = w * self.radius
           
        # negative values produce pincushion
 
        # create map with the barrel pincushion distortion formula
        for y in range(h):
          delta_y = scale_y * (y - center_y)
          for x in range(w):
            # determine if pixel is within an ellipse
            delta_x = scale_x * (x - center_x)
            distance = delta_x * delta_x + delta_y * delta_y
            if distance >= (radius * radius):
              map_x[y, x] = x
              map_y[y, x] = y
            else:
              factor = 1.0
              if distance > 0.0:
                v = math.sqrt(distance)
                factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
              map_x[y, x] = factor * delta_x / scale_x + center_x
              map_y[y, x] = factor * delta_y / scale_y + center_y
            
        # do the remap
        image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        filename = "barrdistorted_"+str(index) + "_" + str(self.radius) + "_" + str(amount) + "_" + basename
        output_filepath = os.path.join(output_dir, filename)
        cv2.imwrite(output_filepath, image)
  
  # This method is based on the code in the following stackoverflow.com website:
  # https://stackoverflow.com/questions/59776772/python-opencv-how-to-apply-radial-barrel-distortion
  def pincushion_distort(self, image, basename, output_dir):    
    (h,  w,  _) = image.shape

    # set up the x and y maps as float32
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    scale_x = 1
    scale_y = 1
    index   = 1000
    for amount in self.pincamounts:
      for center in self.pinccenters:
        index += 1
        (ox, oy) = center
        center_x = w * ox
        center_y = h * oy
        radius = w * self.pincradius
           
        # negative values produce pincushion

        # create map with the barrel pincushion distortion formula
        for y in range(h):
          delta_y = scale_y * (y - center_y)
          for x in range(w):
            # determine if pixel is within an ellipse
            delta_x = scale_x * (x - center_x)
            distance = delta_x * delta_x + delta_y * delta_y
            if distance >= (radius * radius):
              map_x[y, x] = x
              map_y[y, x] = y
            else:
              factor = 1.0
              if distance > 0.0:
                v = math.sqrt(distance)
                factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
              map_x[y, x] = factor * delta_x / scale_x + center_x
              map_y[y, x] = factor * delta_y / scale_y + center_y
            
        # do the remap
        image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        filename = "pincdistorted_"+str(index) + "_" + str(self.pincradius) + "_" + str(amount) + "_" + basename
        output_filepath = os.path.join(output_dir, filename)
        cv2.imwrite(output_filepath, image)

if __name__ == "__main__":
  try:
    output_dir  = "./Prostate-Glands-and-Tumor-master/"

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    train_images_dir  = "./RINGS/TRAIN/IMAGES/"
    glands_masks_dir  = "./RINGS/TRAIN/MANUAL GLANDS/"
    tumor_masks_dir   = "./RINGS/TRAIN/MANUAL TUMOR/"

    augmentation= True
    generator = ImageMaskDatasetGenerator(images_dir       = train_images_dir, 
                                          glands_masks_dir = glands_masks_dir, 
                                          tumor_masks_dir  = tumor_masks_dir,
                                          output_dir  = output_dir, 
                                          angle = 0,
                                          augmentation=augmentation)
    
    generator.generate()

  except:
    traceback.print_exc()

 
