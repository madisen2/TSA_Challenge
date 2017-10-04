import pyximport; pyximport.install()
import video_util
width=660
height=512
num_frames=16
import numpy as np
import test
def get_image_for_frame(i):

   # r=np.zeros((height,width))
   # r[:,i]=1.0
   # r[:, -i] = 0.5
   # return r
    files=test.list_files()
    r=files[0]
    header=test.read_header(r)
    data_scale_factor=header['data_scale_factor']
    data_scale_factor=data_scale_factor[0]
    r=test.read_data(r)
    r=r.transpose()
    r=r[i]
    r=r.astype(np.double)
    r=r.flatten()*data_scale_factor
    #r=(test.get_single_image(files[0],0))[i].flatten()
    #r=r.astype(np.double)
    return r

with video_util.VideoHelper("output.mp4", fps=3, frameSize=(width, height)) as vh:
    for i in range(num_frames):
        with vh.frame() as buffer:
            frameData = get_image_for_frame(i)
            buffer.blit_flat_float[:, :] = frameData.reshape((width*height,))
