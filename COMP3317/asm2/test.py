import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

from pathlib import Path

def rgb2gray(img_color) :

    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image

    # TODO: using the Y channel of the YIQ model to perform the conversion
    
    
    img_gray = np.dot(img_color[..., :3], [0.299, 0.587, 0.114])
    # def temp(channels):
    #     return np.dot(channels, [0.299, 0.587, 0.114])
    # img_gray = np.apply_along_axis(temp, -1, img_color)

    return img_gray
# a = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]])
# print(rgb2gray(a))

def smooth1D(img, sigma, m) :
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result

    n = int(sigma * (2*np.log(1000))**0.5)
    # arange(-n, n+1) due to the 0 mean value of the gaussian distribution we set
    kernal = np.exp(-(np.arange(-n, n+1)**2) / (2 * sigma**2))
    
    if m:
        # # normalize method 1: 
        norm_mat = kernal / kernal.sum()
        
        img_smoothed = convolve1d(img, norm_mat, axis=-1, output=np.float64, mode='constant', cval=0, origin=0)
    else:
        # normalize method 2:
        img_smoothed = convolve1d(img, kernal, axis=-1, output=np.float64, mode='constant', cval=0, origin=0)
        
        norm_mat = np.ones(img.shape[1])
        norm_mat = convolve1d(norm_mat, kernal, axis=-1, output=np.float64, mode='constant', cval=0, origin=0)
        norm_mat = np.tile(norm_mat, (img.shape[0], 1))
        
        img_smoothed /= norm_mat
    
    return img_smoothed

def smooth2D(img:np.ndarray, sigma) :
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result

    # TODO: smooth the image along the vertical direction
    img_smoothed = smooth1D(img, sigma, 0)
    # TODO: smooth the image along the horizontal direction
    img_smoothed = smooth1D(img_smoothed.T, sigma, 0).T
    
    return img_smoothed


def load_image(inputfile) :
    # input:
    #    inputfile - path of the image file
    # return:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image

    try :
        img_color = plt.imread(inputfile)
        return img_color
    except :
        print('Cannot open \'{}\'.'.format(inputfile))
        exit()
        
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 2')
    parser.add_argument('-i', '--image', type = str, default = 'grid1.jpg',
                        help = 'filename of input image')
    parser.add_argument('-s', '--sigma', type = float, default = 1.0,
                        help = 'sigma value for Gaussain filter (default = 1.0)')
    parser.add_argument('-t', '--threshold', type = float, default = 1e6,
                        help = 'threshold value for corner detection (default = 1e6)')
    parser.add_argument('-o', '--output', type = str,
                        help = 'filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('COMP3317 Assignment 2')
    print('input file : {}'.format(args.image))
    print('sigma      : {:.2f}'.format(args.sigma))
    print('threshold  : {:.2e}'.format(args.threshold))
    print('output file: {}'.format(args.output))
    print('------------------------------')

    # load the image
    img_color = load_image(Path(__file__).parent / args.image)
    print('\'{}\' loaded...'.format(args.image))

    # uncomment the following 2 lines to show the color image
    # plt.imshow(np.uint8(img_color))
    # plt.show()

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = smooth2D(rgb2gray(img_color), 1.0)
    # uncomment the following 2 lines to show the grayscale image
    plt.imshow(np.float32(img_gray), cmap = 'gray')
    plt.show()

main()