import pyximport; pyximport.install()
import video_util
width=512
height=660
num_frames=16
import numpy as np
import test
def get_image_for_frame(i):

   # r=np.zeros((height,width))
   # r[:,i]=1.0
   # r[:, -i] = 0.5
   # return r
    files=test.list_files()#list all .aps files in sequential order
    r=files[0]#0 stands for subject 0, all 16 images for this subject are included
    header=test.read_header(r)#extracting the header dictionary
    r=test.read_data(r)#reading the binary .aps file & extracting image data
    r=r.transpose()#so we can index each of the 16 images
    r=r[i]#images 1-16
    print(r.max())
    print(r.min())
    print(r.mean())
    r=r.astype(np.double)
    print(r.shape)
    r=r[::-1,]
    #r=(test.get_single_image(files[0],0))[i].flatten()
    #r=r.astype(np.double)
    return r/r.max()

with video_util.VideoHelper("output.mp4", fps=3, frameSize=(width, height)) as vh:
    for i in range(num_frames):
        with vh.frame() as buffer:
            frameData = get_image_for_frame(i)
            buffer.blit_flat_float_mono_as_alpha[:, :] = frameData.flatten()
