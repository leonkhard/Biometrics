import numpy as np
import os, os.path
from skimage import io
from Classyfiers import *

def classify(clf, img, classes):
    #найти изображдение среди всех классов с данным класификатором, которое бы было ближе всего к исходному
    
    distances = np.array(list(map(lambda img2: clf(img, img2), classes)))
    #distances = distances / np.sum(distances)
    
    return distances.argmin()
