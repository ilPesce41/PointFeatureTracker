import imageio as im
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numba
import numba.cuda as cuda
import numpy as np
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
import scipy.ndimage
import os
import math

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

@numba.autojit
def get_eigs(img):
    div_window = 10
    window_size = 20

    #Get A matrix entires
    imgy = gaussian_filter1d(img,1,axis=1,order=1,truncate=div_window)
    imgx = gaussian_filter1d(img,1,axis=0,order=1,truncate=div_window)
    imgxy = gaussian_filter1d(imgx,1,axis=1,order=1,truncate=div_window)

    a = gaussian_filter(imgx,1,truncate=window_size)
    b = gaussian_filter(imgxy,1,truncate=window_size)
    c = gaussian_filter(imgy,1,truncate=window_size)
    eigs = np.ndarray(shape=img.shape)
    for i in numba.prange(img.shape[0]):
        for j in numba.prange(img.shape[1]):
            A = np.array([[a[i,j]**2, b[i,j]],[b[i,j],c[i,j]**2]])
            # eigs[i,j] = np.min(np.linalg.eigvals(A))
            eigs[i,j] = np.linalg.det(A) - 0.06*np.trace(A)**2
    return eigs

def get_displacement(I,J):
    I = I/255
    J = J/255
    #Gradient window
    grad_window = 5
    c = int(grad_window/2)
    #Get gradient at center of window
    gx = gaussian_filter1d(J,1,axis=0,order=1,truncate=grad_window)
    gy = gaussian_filter1d(J,1,axis=1,order=1,truncate=grad_window)
    # gx = scipy.ndimage.sobel(J,axis=1)
    # gy = scipy.ndimage.sobel(J,axis=0)

    e1 = np.sum((J-I)*gx)
    e2 = np.sum((J-I)*gy)
    e = -np.array([e1,e2]).T
    gx=np.sum(gx)
    gy=np.sum(gy)
    Z = np.array([[gx**2,gx*gy],[gx*gy,gy**2]])

    d = np.linalg.pinv(Z)@e
    return d

def get_feature_points(img):    
    
    eigs = get_eigs(img)
    vals = eigs.flatten()
    vals.sort()
    ratio = .005
    t = vals[-50:]
    mask = np.where(eigs>t[0],1,0)
    plist = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == 1:
                plist.append((i,j))
    return plist

@numba.autojit
def track_points(imgs,_imgs,pnts):
    img = imgs[0]
    for i in range(1,len(imgs)):
        print(i)
        npnts = []
        # print(pnts)
        win_size = 5
        c = int(win_size/2)
        for pnt in pnts:
            # print(pnt)
            x,y = int(pnt[0]),int(pnt[1])
            if i ==1:
                _imgs[0][x,y] = np.array([255,0,0])
            if x<c or x>img.shape[0]-(c+1):
                continue
            if y<c or y>img.shape[1]-(c+1):
                continue
            dt = 100
            window_1 = imgs[i-1][x-c:x+c,y-c:y+c]
            cnt = 0
            min_err = 1e6
            min_xy = (x,y)
            while dt>0.01 and cnt<200:
                cnt +=1
                xt,yt =int(x),int(y)
                window_2 = imgs[i][xt-c:xt+c,yt-c:yt+c]
                err = np.sum(np.abs(window_1-window_2))
                if err<min_err:
                    min_err = err
                min_xy = (xt,yt)
                d = get_displacement(window_1,window_2)
                # d0 = math.ceil(np.abs(d[0]))*d[0]/np.abs(d[0]+0.0001)
                # d1 = math.ceil(np.abs(d[1]))*d[1]/np.abs(d[1]+0.0001)
                d0 = d[0]
                d1 = d[1]
                x,y = x+d0,y+d1
                dt = np.abs(d[0]) + np.abs(d[1])
                if x<c or x>img.shape[0]-(c+1):
                    break
                if y<c or y>img.shape[1]-(c+1):
                    break
                # print(pnt,dt,(x,y))
            print(dt)
            xi,yi = int(x),int(y)
            npnts.append((xi,yi))
            for pnt in npnts:
                x,y = map(int,pnt)
                if x<0 or x>_imgs[i].shape[0]-1:
                    continue
                if y<0 or y>_imgs[i].shape[1]-1:
                    continue
                _imgs[i][x,y] = np.array([255,0,0])
            pnts = npnts
    return pnts, _imgs


def points_from_frames(frame_dir,outdir = None):
    join = os.path.join
    frame_flist = os.listdir(frame_dir)
    
    temp = []
    for f in frame_flist:
        if f.lower().endswith('.png'):
            temp.append(f)
    frame_flist = temp
    frame_flist = sorted(frame_flist,key=lambda x: int(x.split('.')[0]))[:10]
    print(frame_flist)

    frames = [im.imread(join(frame_dir,x)) for x in frame_flist]
    if len(frames[0].shape)>2:
        gframes = [x[:,:,0] for x in frames]
    else:
        gframes = frames
        _frames = []
        for frame in frames:
            temp = np.ndarray(shape=(*frame.shape,3),dtype=int)
            temp[:,:,0] = frame
            temp[:,:,1] = frame
            temp[:,:,2] = frame
            _frames.append(temp)
        frames = _frames

    pnts = get_feature_points(gframes[0])
    pnts,frames = track_points(gframes,frames,pnts)
    
    if outdir is None:
        outdir = join(frame_dir,"output")
        os.makedirs(outdir,exist_ok=True)
    for i,frame in enumerate(frames):
        im.imwrite(join(outdir,frame_flist[i]),frame)
    

def points_from_video(video):
    pass

if __name__ == "__main__":

    points_from_frames('moon_frames')