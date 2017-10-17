#madisen phillips
#teuscher labs
#kaggle 2017 tsa challenge
"""this file loads image data from the tsa challenge on Kaggle.com and segments it"""

from __future__ import print_function
from __future__ import division
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import click
import pyximport; pyximport.install()
import video_util
from PIL import Image
import zones

# constants
COLORMAP = 'pink'


APS_FILE_NAME = '/stash/tlab/datasets/2017KaggleTsa/stage1/aps/00360f79fd6e02781457eda48f85da90.aps'
BODY_ZONES = '/stash/tlab/datasets/2017KaggleTsa/stage1/body_zones.png'
THREAT_LABELS = '/stash/tlab/datasets/2017KaggleTsa/stage1/stage1_labels.csv'


#class Model:
 #   def fit(self, inputs, outputs):
  #      self.params=lstsq
   # def predict(self,inputs):
        #return applying self.params to inputs
#____________________________________________________________________________#




def list_files():
    """Returns a list of the file names from the path provided."""

    path="/stash/tlab/datasets/2017KaggleTsa/stage1/aps"
    file_names=os.listdir(path)
    file_names=[
            os.path.join(path,p)
            for p in file_names]

    return file_names



def to_rgb(file_name):

    data=read_data(file_name)
    data=data.astype('uint8')

    return data


def animation_files(files):

    file_list=[]
    for i in files:
        file_list.append([to_rgb(i)])


    return file_list




#----------------------------------------------------------------------------------
# read_header(infile):  takes an aps file and creates a dict of the data
#
# infile:               an aps file
#
# returns:              all of the fields in the header
#----------------------------------------------------------------------------------

def read_header(infile):
    # declare dictionary
    h = dict()

    with open(infile, 'r+b') as fid:

        h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
        h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
        h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
        h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
        h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
        h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
        h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
        h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
        h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
        h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)

    return h




#----------------------------------------------------------------------------------
# read_data(infile):  reads and rescales any of the four image types
#
# infile:             an .aps, .aps3d, .a3d, or ahi file
#
# returns:            the stack of images
#
# note:               word_type == 7 is an np.float32, word_type == 4 is np.uint16
#----------------------------------------------------------------------------------

def read_data(infile):

    # read in header and get dimensions
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])

    extension = os.path.splitext(infile)[1]

    with open(infile, 'rb') as fid:

        # skip the header
        fid.seek(512)

        # handle .aps and .a3aps files
        if extension == '.aps' or extension == '.a3daps':

            if(h['word_type']==7):
                data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)

            elif(h['word_type']==4):
                data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

            # scale and reshape the data
            data = data * h['data_scale_factor']
            data = data.reshape(nx, ny, nt, order='F').copy()

        # handle .a3d files
        elif extension == '.a3d':

            if(h['word_type']==7):
                data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)

            elif(h['word_type']==4):
                data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

            # scale and reshape the data
            data = data * h['data_scale_factor']
            data = data.reshape(nx, nt, ny, order='F').copy()

        # handle .ahi files
        elif extension == '.ahi':
            data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)
            data = data.reshape(2, ny, nx, nt, order='F').copy()
            real = data[0,:,:,:].copy()
            imag = data[1,:,:,:].copy()

        if extension != '.ahi':
            return data
        else:
            return real, imag



#-----------------------------------------------------------------------------
# get_hit_rate_stats(infile):  gets the threat probabilities in a useful form
#
# infile:                      labels csv file
#
# returns:                     a dataframe of the summary hit probabilities
#
#-----------------------------------------------------------------------------

def get_hit_rate_stats(infile):
    # pull the labels for a given patient
    df = pd.read_csv(infile)

    # Separate the zone and patient id into a df
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    df = df[['Subject', 'Zone', 'Probability']]

    # make a df of the sums and counts by zone and calculate hit rate per zone, then sort high to low
    df_summary = df.groupby('Zone')['Probability'].agg(['sum','count'])
    df_summary['Zone'] = df_summary.index
    df_summary['pct'] = df_summary['sum'] / df_summary['count']
    df_summary.sort_values('pct', axis=0, ascending= False, inplace=True)

    return df_summary


#-----------------------------------------------------------------------------
# chart_hit_rate_stats(df_summary): charts threat probabilities in desc order by zone
#
# df_summary:   a dataframe like that returned from get_hit_rate_stats(...)
#
#-----------------------------------------------------------------------------

def chart_hit_rate_stats(df_summary):
    fig, ax = plt.subplots(figsize=(15,5))
    sns.barplot(ax=ax, x=df_summary['Zone'], y=df_summary['pct']*100)



#----------------------------------------------------------------------------
# print_hit_rate_stats(df_summary): lists threat probabilities by zone
#
# df_summary:               a dataframe like that returned from get_hit_rate_stats(...)
#
#-----------------------------------------------------------------------------

def print_hit_rate_stats(df_summary):
# print the table of values readbly
    print ('{:6s}   {:>4s}   {:6s}'.format('Zone', 'Hits', 'Pct %'))
    print ('------   ----- ----------')
    for zone in df_summary.iterrows():
        print ('{:6s}   {:>4d}   {:>6.3f}%'.format(zone[0],
            np.int16(zone[1]['sum']), zone[1]['pct']*100))
    print ('------   ----- ----------')
    print ('{:6s}   {:>4d}   {:6.3f}%'.format('Total ',
        np.int16(df_summary['sum'].sum(axis=0)),
    ( df_summary['sum'].sum(axis=0) / df_summary['count'].sum(axis=0))*100))


#-----------------------------------------------------------------------------
# def get_subject_labels(infile, subject_id): lists threat probabilities by zone
#
# infile:            labels csv file
#
# subject_id:        the individual you want the threat zone labels for
#
# returns:           a df with the list of zones and contraband (0 or 1)
#
#-----------------------------------------------------------------------------

def get_subject_labels(infile, subject_id):

    # read labels into a dataframe
    df = pd.read_csv(infile)

    # Separate the zone and subject id into a df
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    df = df[['Subject', 'Zone', 'Probability']]
    threat_list = df.loc[df['Subject'] == subject_id]

    return threat_list


#-----------------------------------------------------------------------------
# plot_image_set(infile):  takes an aps file and shows all 16 90 degree shots
#
# infile:                  an aps file
#-----------------------------------------------------------------------------

def plot_image_set(infile):

    # read in the aps file, it comes in as shape(512, 620, 16)
    img = read_data(infile)

    # transpose so that the slice is the first dimension shape(16, 620, 512)
    img = img.transpose()

    # show the graphs
    fig, axarr = plt.subplots(nrows=4, ncols=4, figsize=(10,10))

    i = 0
    for row in range(4):
        for col in range(4):
            resized_img = cv2.resize(img[i], (0,0), fx=0.1, fy=0.1)
            axarr[row, col].imshow(np.flipud(resized_img), cmap=COLORMAP)
            i += 1

    print('Done!')


#-----------------------------------------------------------------------------
# get_single_image(infile, nth_image):returns the nth image from the image stack
#
# infile:                              an aps file
#
# returns:                             an image
#-----------------------------------------------------------------------------

def get_single_image(infile, nth_image):

    # read in the aps file, it comes in as shape(512, 620, 16)
    img = read_data(infile)

    # transpose so that the slice is the first dimension shape(16, 620, 512)
    img = img.transpose()

    return np.flipud(img[nth_image])


#-----------------------------------------------------------------------------
# convert_to_grayscale(img):           converts a ATI scan to grayscale
#
# infile:                              an aps file
#
# returns:                             an image
#-----------------------------------------------------------------------------

def convert_to_grayscale(img):
    # scale pixel values to grayscale
    base_range = np.amax(img) - np.amin(img)
    rescaled_range = 255 - 0
    img_rescaled = (((img - np.amin(img)) * rescaled_range) / base_range)

    return np.uint8(img_rescaled)

#-------------------------------------------------------------------------------
# spread_spectrum(img):        applies a histogram equalization transformation
#
# img:                         a single scan
#
# returns:                     a transformed scan
#-------------------------------------------------------------------------------

def spread_spectrum(img):
    img = stats.threshold(img, threshmin=12, newval=0)

    # see http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img= clahe.apply(img)

    return img

#-----------------------------------------------------------------------------
# roi(img, vertices):              uses vertices to mask the image
#
# img:                             the image to be masked
#
# vertices:                        a set of vertices that define the region of interest
#
# returns:                         a masked image
#------------------------------------------------------------------------------

def roi(img, vertices):

    # blank mask
    mask = np.zeros_like(img)

    # fill the mask
    cv2.fillPoly(mask, [vertices], 255)

    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)


    return masked


#-----------------------------------------------------------------------------
# crop(img, crop_list):                uses vertices to mask the image
#
# img:                                 the image to be cropped
#
# crop_list:                           a crop_list entry with [x , y, width, height]
#
# returns:                             a cropped image
#-----------------------------------------------------------------------------

def crop(img, crop_list):

    x_coord = crop_list[0]
    y_coord = crop_list[1]
    width = crop_list[2]
    height = crop_list[3]
    cropped_img = img[x_coord:x_coord+width, y_coord:y_coord+height]

    return cropped_img


#------------------------------------------------------------------------------
# normalize(image): Take segmented tsa image and normalize pixel values to be
#                   between 0 and 1
#
# parameters:      image - a tsa scan
#
# returns:         a normalized image
#
#------------------------------------------------------------------------------

def normalize(image):
    MIN_BOUND = 0.0
    MAX_BOUND = 255.0

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

#------------------------------------------------------------------------------
# zero_center(image): Shift normalized image data and move the range so it is 0 c
#                     entered at the PIXEL_MEAN
#
# parameters:         image
#
# returns:            a zero centered image
#
#-----------------------------------------------------------------------------


def zero_center(image):

    PIXEL_MEAN = 0.014327

    image = image - PIXEL_MEAN
    return image



@click.command()
#@click.option('--normalize')
@click.option('--subject',prompt='Subject #: ')
@click.option('--zone',prompt='Zone #: ')


def main(subject,zone):
    """Used for file framework"""

    files=list_files()
    subject=files[subject]
    subject=read_data(subject).transpose()
    cropped_images=[]

    #for x in subject:
     #   cropped_images.append([])
      #  cropped_images.append(crop(x,zones.zone_crop_list[zone]))

    return 0

if __name__ == '__main__':
    main()


