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


class KalmanPoint:

    def __init__(self,x,y,A):

        self.R = A
        self.x = x
        self.y = y
        self.A = np.ndarray(
            [[1,0,1,0],
            [0,1,0,1],
            [0,0,1,0],
            [0,0,0,1]]
        )
        self.Q = np.eye(4)@np.array([25,25,45,45]).T
        self.P = Q
        self.history = []
        self.history.append(x,y)
    
    def update(self,x,y):
        P = self.P
        St = np.array([self.x,self.y]).T
        H = np.ndarray([[1,0,0,0],[0,1,0,0])
        Em = self.A@St
        Ptm = A@P@A.T + Q
        K = Ptm@H.T@np.linalg.pinv(H@Ptm@H.T+self.R)
        St = Stm + K@(np.array(x,y).T-H@Stm)
        self.P = (np.eye(4)-K@H)@Ptm
        self.x = St[0]
        self.y = St[1]
        self.history.append((self.x,self,y))

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
def get_corners(img):
    """
    Finds all of the good feature points in the image
    """

    #Controls the size of the the gaussian kernels
    div_window = 10
    window_size = 20

    #Get A matrix entires
    imgy = gaussian_filter1d(img,1,axis=1,order=1,truncate=div_window)
    imgx = gaussian_filter1d(img,1,axis=0,order=1,truncate=div_window)
    imgxy = gaussian_filter1d(imgx,1,axis=1,order=1,truncate=div_window)

    a = gaussian_filter(imgx,1,truncate=window_size)
    b = gaussian_filter(imgxy,1,truncate=window_size)
    c = gaussian_filter(imgy,1,truncate=window_size)

    #Iterates through each pixel and gets an interest value
    eigs = np.ndarray(shape=img.shape)
    for i in numba.prange(img.shape[0]):
        for j in numba.prange(img.shape[1]):
            A = np.array([[a[i,j]**2, b[i,j]],[b[i,j],c[i,j]**2]])
            # eigs[i,j] = np.min(np.linalg.eigvals(A))
            eigs[i,j] = np.linalg.det(A) - 0.06*np.trace(A)**2
    return eigs

def get_displacement(I,J):
    """
    Determines delta for newton-raphson optimization
    """
    I = I/255
    J = J/255
    #Gradient window
    grad_window = 5
    c = int(grad_window/2)

    #Get gradient at center of window
    gx = gaussian_filter1d(J,1,axis=0,order=1,truncate=grad_window)
    gy = gaussian_filter1d(J,1,axis=1,order=1,truncate=grad_window)

    e1 = np.sum((J-I)*gx)
    e2 = np.sum((J-I)*gy)
    e = -np.array([e1,e2]).T

    gx=np.sum(gx)
    gy=np.sum(gy)

    #Find delta assuming just translation
    Z = np.array([[gx**2,gx*gy],[gx*gy,gy**2]])

    d = np.linalg.pinv(Z)@e
    return d

def get_feature_points(img):    
    """
    Finds set of feature points in an image
    """
    
    #Get all points w/ value indicating uniquness
    eigs = get_corners(img)

    #Just keep the 50 with highest uniqueness
    vals = eigs.flatten()
    vals.sort()
    t = vals[-50:]
    
    #Organize points into a list
    mask = np.where(eigs>t[0],1,0)
    plist = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == 1:
                plist.append((i,j))
    return plist

@numba.autojit
def track_points(imgs,_imgs,pnts):
    """
    Track points from first grayscale image and add red dots over those
    points in _imgs a colored image array
    """
    
    #Starting image
    img = imgs[0]
    
    for i in range(1,len(imgs)):
        
        #Array for new points
        npnts = []
        #window size for searching for point delta
        win_size = 5
        c = int(win_size/2)
        
        for pnt in pnts:
            #Pixel value
            x,y = int(pnt[0]),int(pnt[1])
            
            #Add markers in first image
            if i ==1:
                _imgs[0][x,y] = np.array([255,0,0])
            #Check if pixel value is invalid
            if x<c or x>img.shape[0]-(c+1):
                continue
            if y<c or y>img.shape[1]-(c+1):
                continue
            
            #Determine new point
            dt = 100
            #Point window in first image
            window_1 = imgs[i-1][x-c:x+c,y-c:y+c]
            cnt = 0
            #Continue iteration until stop conditions are met
            while dt>0.01 and cnt<200:
                #Increment iteration count
                cnt +=1
                #Pixel value
                xt,yt =int(x),int(y)
                #Get window in next frame
                window_2 = imgs[i][xt-c:xt+c,yt-c:yt+c]
                #Determine translation value
                d = get_displacement(window_1,window_2)
                d0 = d[0]
                d1 = d[1]
                x,y = x+d0,y+d1
                #Measure movement
                dt = np.abs(d[0]) + np.abs(d[1])
                
                #Pixel has moved off screen
                if x<c or x>img.shape[0]-(c+1):
                    break
                if y<c or y>img.shape[1]-(c+1):
                    break
            #Set new pixel values for point
            xi,yi = int(x),int(y)
            npnts.append((xi,yi))
            
            #Add markers for points in color image
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
    """
    Track points from frames in a directory
    """
    
    #Get list of frame files
    join = os.path.join
    frame_flist = os.listdir(frame_dir)
    
    #Filter for just png's and put them in numerical order
    temp = []
    for f in frame_flist:
        if f.lower().endswith('.png')or f.lower().endswith('.jpg'):
            temp.append(f)
    frame_flist = temp
    frame_flist = sorted(frame_flist,key=lambda x: int(x.split('.')[0]))[:10]
    print(frame_flist)
    #Import the image data and split into color and grayscale images
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

    #Get points to track in first image
    pnts = get_feature_points(gframes[0])
    #Track points in all other frames
    pnts,frames = track_points(gframes,frames,pnts)
    
    #Output new frames
    if outdir is None:
        outdir = join(frame_dir,"output")
        os.makedirs(outdir,exist_ok=True)
    for i,frame in enumerate(frames):
        im.imwrite(join(outdir,frame_flist[i]),frame)
    
def points_from_video(video):
    pass

if __name__ == "__main__":

    fdir = input("Frame Directory:")
    points_from_frames(fdir)
    input("Complete")