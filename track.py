import imageio as im
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numba
import numba.cuda as cuda
import numpy as np
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
import os

im.plugins.ffmpeg.download()

@numba.njit
def img_to_grayscale(im):
    """
    Convert image to grayscale using Rec. 709 conversion spec
    """
    r,g,b = im[:,:,0],im[:,:,1],im[:,:,2]

    return 0.2126*r + 0.7512*g + 0.0722*b

@numba.autojit
def vid_to_grayscale(vid,new_vid):
    """Convert video to grayscale"""
    t,x,y = vid.shape[0],vid.shape[1],vid.shape[2]
    for i in numba.prange(t):
        new_vid[i] = img_to_grayscale(vid[i])
    return new_vid

def video_to_frames(video,directory):
    """
    Helper function for dumping video as frames
    """
    os.makedirs(directory,exist_ok=True)
    for i in range(len(video)):
        im.imwrite(os.path.join(directory,'frame{}.png'.format(i)),data[i])

    


if __name__ == "__main__":
    
    frame_list = os.listdir('frames')
    img = im.imread(os.path.join('frames',frame_list[0]))
    imm = img
    img = np.array(img)[:,:,0]
    
    div_window = 10
    window_size = 20

    #Get A matrix entires
    imgy = gaussian_filter1d(img,1,axis=0,order=1,truncate=div_window)
    imgx = gaussian_filter1d(img,1,axis=1,order=1,truncate=div_window)
    imgxy = gaussian_filter1d(imgx,1,axis=0,order=1,truncate=div_window)

    a = gaussian_filter(imgx,1,truncate=window_size)
    b = gaussian_filter(imgxy,1,truncate=window_size)
    c = gaussian_filter(imgy,1,truncate=window_size)
    print("Getting Eigs")
    eigs = np.ndarray(shape=img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            A = np.array([[a[i,j]**2, b[i,j]],[b[i,j],c[i,j]**2]])
            eigs[i,j] = np.min(np.linalg.eigvals(A))
    print("Done")

    vals = eigs.flatten()
    vals.sort()
    plt.hist(vals)
    t = vals[-200:]
    mask = np.where(eigs>20000,1,0)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j]==1:
                imm[i,j] = np.array([255,0,0])


    plt.show()
    plt.show()
    plt.imshow(imm)
    plt.show()
    
