from utils import *
from PIL import Image
import os
import numpy as np
import scipy.misc as misc

def rem_cl_eg(source_path, c_path, e_path):
    """
    :param source_path: The unpreprocessed datasets are from vedio
    :param c_path: croped and resized datasets
    :param e_path: smoothed datasets
    :return:
    """
    #Easy to remove clear edge, just reduce the image and then Enlarge the image
    filenames = os.listdir(source_path)
    for filename in filenames:
        img = np.array(Image.open(source_path+filename))
        img = resize_and_crop(img, 256)
        size2small = misc.imresize(img, [128, 128])
        size2original = misc.imresize(size2small, [256, 256])
        # Image.fromarray(size2original).show()
        Image.fromarray(size2original).save(e_path + filename)
        Image.fromarray(img).save(c_path + filename)

if __name__ == "__main__":
    rem_cl_eg("E://DeepLearn_Experiment//CartoonSet//source//", "E://DeepLearn_Experiment//CartoonSet//c//", "E://DeepLearn_Experiment//CartoonSet//e//")
