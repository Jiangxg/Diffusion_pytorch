from PIL import Image
import numpy as np
import os
import imageio


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def normalize_to_zero_to_one(img):
    return (img  + 1.) * 0.5


def save_image(filename, img, output_dir):
    '''
    Args:
        path: path of saved images
        img: tensor of a image, with a size of CxHxW, range from 0 to 1
    '''
    img = img.permute(1, 2, 0)
    img_array = np.array(img)
    assert len(img_array.shape) == 2 or (len(img_array.shape) == 3
                                         and img_array.shape[2] in [3, 4])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0., 1.) * 255).astype(np.uint8)
    Image.fromarray(img_array).save(os.path.join(output_dir, filename))
    

def video_gen(dir, img_list):
    '''
    Generate a .gif file for the images in img_list.
    
    Args:
        dir: the directory of the images
        img_list: list of the name of images
    '''
    img_array_lst = []
    for img_name in img_list:
        img_filepath = os.path.join(dir, img_name)
        img_array = np.asarray(Image.open(img_filepath))
        img_array_lst.append(img_array)
    filename = 'video.gif'
    imageio.mimwrite(os.path.join(dir, filename), img_array_lst, fps=5, format='GIF')