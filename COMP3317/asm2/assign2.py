################################################################################
# COMP3317 Computer Vision
# Assignment 2 - Conrner detection
################################################################################
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from pathlib import Path

################################################################################
#  perform RGB to grayscale conversion
################################################################################
def rgb2gray(img_color) :
    img_gray = np.dot(img_color[...,:3], [0.299, 0.587, 0.114])
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image

    # TODO: using the Y channel of the YIQ model to perform the conversion

    img_gray = np.dot(img_color[..., :3], [0.299, 0.587, 0.114])

    return img_gray

################################################################################
#  perform 1D smoothing using a 1D horizontal Gaussian filter
################################################################################
def smooth1D(img:np.ndarray, sigma) -> np.ndarray: 
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result

    n = int(sigma * (2*np.log(1000))**0.5)
    
    # arange(-n, n+1) due to the 0 mean value of the gaussian distribution we set
    kernal = np.exp(-(np.arange(-n, n+1)**2) / (2 * sigma**2))
    
    # # normalize method 1: 
    # norm_mat = kernal / kernal.sum()
    # img_smoothed = convolve1d(img, kernal, axis=-1, mode='constant', Cval=0, origin=0)
    
    # normalize method 2: (partial filter)
    img_smoothed = convolve1d(img, kernal, axis=-1, mode='constant', cval=0, origin=0)
    
    norm_mat = np.ones(img.shape[1])
    norm_mat = convolve1d(norm_mat, kernal, axis=-1, mode='constant', cval=0, origin=0)
    norm_mat = np.tile(norm_mat, (img.shape[0], 1))
    
    img_smoothed /= norm_mat
    
    return img_smoothed

################################################################################
#  perform 2D smoothing using 1D convolutions
################################################################################
def smooth2D(img:np.ndarray, sigma) :
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result

    # TODO: smooth the image along the vertical direction
    img_smoothed = smooth1D(img, sigma)
    # TODO: smooth the image along the horizontal direction
    img_smoothed = smooth1D(img_smoothed.T, sigma).T
    
    return img_smoothed

################################################################################
#   perform Harris corner detection
################################################################################
def harris(img, sigma, threshold) :
    # input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner

    # TODO: compute Ix & Iy
    
    Iy, Ix = np.gradient(img)
    

    # TODO: compute Ix2, Iy2 and IxIy
    
    Iy2, Ix2 = np.square(Iy), np.square(Ix)
    IxIy = np.multiply(Ix, Iy)
    

    # TODO: smooth the squared derivatives
    
    Ix2, Iy2, IxIy = smooth2D(Ix2, sigma), smooth2D(Iy2, sigma), smooth2D(IxIy, sigma)


    # TODO: compute cornesness functoin R
    
    R = (Ix2*Iy2 - np.square(IxIy)) - 0.04 * np.square(Ix2 + Iy2) # k = 0.04


    # TODO: mark local maxima as corner candidates;
    #       perform quadratic approximation to local corners upto sub-pixel accuracy

    final_coor_of_corners = [[], []]
    valid_R_coor = np.where(R > 0)  # some R value is extremely small and float will round them to 0, so we eliminate them
    for i, j in zip(*valid_R_coor):
        if i == 0 or i == R.shape[0] - 1 or j == 0 or j == R.shape[1] - 1: continue
        # 3x3 window of the point
        window = R[i-1:i+2, j-1:j+2]
        # check if the center pixel is the maximum
        if window[1, 1] == np.max(window):
            final_coor_of_corners[0].append(i)
            final_coor_of_corners[1].append(j)
    
    def subPixelAcc(x_neighbors, y_neighbors):
        def subPixelAcc1D(one_d_neighbors):
            # coefficients of input's quatric function
            a = (one_d_neighbors[2] + one_d_neighbors[0] - 2*one_d_neighbors[1]) / 2
            b = (one_d_neighbors[2] - one_d_neighbors[0]) / 2
            
            sub_coor = -b / (2*a)
            
            return a, b, sub_coor
        
        a, c, x = subPixelAcc1D(x_neighbors)
        b, f, y = subPixelAcc1D(y_neighbors)
        
        r = a*x**2 + b*y**2 + c*x + f*y + x_neighbors[1]
        
        return x, y, r
    
    z = 0
    sub_corners = np.zeros((len(final_coor_of_corners[0]), 3))
    for i, j in zip(*final_coor_of_corners):
        if i == 0 or i == R.shape[0] - 1 or j == 0 or j == R.shape[1] - 1: continue
        # sub-pixel accuracy
        x, y, r = subPixelAcc(R[i, j-1:j+2], R[i-1:i+2, j])
        sub_corners[z] = np.array([j - y, i + x, r])  # reverse x and y because matplot's axis is different from numpy
        z += 1

    # TODO: perform thresholding and discard weak corners
    
    corners = sub_corners[sub_corners[:, 2] > threshold]
    

    return sorted(corners, key = lambda corner : corner[2], reverse = True)

################################################################################
#  show corner detection result
################################################################################
def show_corners(img_color, corners) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]

    plt.ion()
    fig = plt.figure('Harris corner detection')
    plt.imshow(img_color)
    plt.plot(x, y,'r+',markersize = 5)
    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  load image from a file
################################################################################
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
        sys.exit(1)

################################################################################
#  save corners to a file
################################################################################
def save_corners(outputfile, corners) :
    # input:
    #    outputfile - path of the output file
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    try :
        file = open(outputfile, 'w')
        file.write('{}\n'.format(len(corners)))
        for corner in corners :
            file.write('{:.6e} {:.6e} {:.6e}\n'.format(corner[0], corner[1], corner[2]))
        file.close()
    except :
        print('Error occurs in writing output to \'{}\'.'.format(outputfile))
        sys.exit(1)

################################################################################
#  load corners from a file
################################################################################
def load_corners(inputfile) :
    # input:
    #    inputfile - path of the file containing corner detection output
    # return:
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    try :
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        print('loading {} corners'.format(nc))
        corners = np.zeros([nc, 3], dtype = np.float64)
        for i in range(nc) :
            line = file.readline()
            x, y, r = line.split()
            corners[i] = [np.float64(x), np.float64(y), np.float64(r)]
        file.close()
        return corners
    except :
        print('Error occurs in loading corners from \'{}\'.'.format(inputfile))
        sys.exit(1)

################################################################################
#  main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 2')
    parser.add_argument('-i', '--image', type = str, default = str(Path(__file__).parent / 'grid1.jpg'),
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
    img_color = load_image(args.image)
    print('\'{}\' loaded...'.format(args.image))

    # uncomment the following 2 lines to show the color image
    # plt.imshow(np.uint8(img_color))
    # plt.show()

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    # plt.imshow(np.float32(img_gray), cmap = 'gray')
    # plt.show()

    # perform corner detection
    print('perform Harris corner detection...')
    corners = harris(img_gray, args.sigma, args.threshold)

    # plot the corners
    print('{} corners detected...'.format(len(corners)))
    show_corners(img_color, corners)

    # save corners to a file
    if args.output :
        save_corners(args.output, corners)
        print('corners saved to \'{}\'...'.format(args.output))

if __name__ == '__main__' :
    main()
