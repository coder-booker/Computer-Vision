################################################################################
# COMP3317 Computer Vision
# Assignment 4 - Camera calibration
################################################################################
import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from numpy.linalg import lstsq, qr, inv
from pathlib import Path

################################################################################
#  estimate planar projective transformations for the 2 calibration planes
################################################################################
def calibrate2D(ref3D, ref2D) :
    #  input:
    #    ref3D - a 8 x 3 numpy ndarray holding the 3D coodinates of the extreme
    #            corners on the 2 calibration planes
    #    ref2D - a 8 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of the corners in ref3D
    # return:
    #    Hxz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the X-Z plane
    #    Hyz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the Y-Z plane

    # TODO : form the matrix equation Ap = b for the X-Z plane
    A_xz = np.zeros((8, 8)) # assume last column is 1, and move to RHS as the b for lstsq;
    b_xz = np.zeros((8,))
    
    # XZ plane, therefore y = 0
    for i in range(4):
        pt_in_3D = ref3D[i] # (x, y, z)
        useful_coor_3D = pt_in_3D[np.where(pt_in_3D != 0)] # (x, z)
        homo_coor_3D = np.concatenate((useful_coor_3D, [1])) # (x, z, 1)
        
        pt_in_2D = ref2D[i] # (u, v)
        
        A_xz[2*i, 0:3] = homo_coor_3D
        A_xz[2*i, 6:] = -pt_in_2D[0] * useful_coor_3D
        A_xz[2*i+1, 3:6] = homo_coor_3D
        A_xz[2*i+1, 6:] = -pt_in_2D[1] * useful_coor_3D
        b_xz[2*i], b_xz[2*i+1] = pt_in_2D[0], pt_in_2D[1]
    
    # TODO : solve for the planar projective transformation using linear least squares
    temp = lstsq(A_xz, b_xz, rcond=None)[0].flatten()
    p_hat_xz = np.concatenate((temp, [1])).reshape((3, 3))

    # TODO : form the matrix equation Ap = b for the Y-Z plane
    A_yz = np.zeros((8, 8))
    b_yz = b_xz = np.zeros((8, 1))
    
    # YZ plane, therefore x = 0
    for i in range(4):
        pt_in_3D = ref3D[i+4] # (x, y, z)
        useful_coor_3D = pt_in_3D[np.where(pt_in_3D != 0)] # (y, z)
        homo_coor_3D = np.concatenate((useful_coor_3D, [1])) # (y, z, 1)
        
        pt_in_2D = ref2D[i+4] # (u, v)
        
        A_yz[2*i, 0:3] = homo_coor_3D
        A_yz[2*i, 6:] = -pt_in_2D[0] * useful_coor_3D
        A_yz[2*i+1, 3:6] = homo_coor_3D
        A_yz[2*i+1, 6:] = -pt_in_2D[1] * useful_coor_3D
        b_yz[2*i], b_yz[2*i+1] = pt_in_2D[0], pt_in_2D[1]

    # TODO : solve for the planar projective transformation using linear least squares
    temp = lstsq(A_yz, b_yz, rcond=None)[0].flatten()
    p_hat_yz = np.concatenate((temp, [1])).reshape((3, 3))

    return p_hat_xz, p_hat_yz

#  generate correspondences for all the corners on the 2 calibration planes
################################################################################
def gen_correspondences(Hxz, Hyz, corners) :
    # input:
    #    Hxz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the X-Z plane
    #    Hyz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the Y-Z plane
    #    corners - a n0 x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n0 being the number of corners)
    # return:
    #    ref3D - a 160 x 3 numpy ndarray holding the 3D coodinates of all the corners
    #            on the 2 calibration planes
    #    ref2D - a 160 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of all the corners in ref3D

    # TODO : define 3D coordinates of all the corners on the 2 calibration planes
    xz_corners_useful_3D_homo = np.zeros((80, 3))
    yz_corners_useful_3D_homo = np.zeros((80, 3))
    xz_corners_3D_cart = np.zeros((80, 3))
    yz_corners_3D_cart = np.zeros((80, 3))
    for i in range(8):
        for j in range(10):
            xz_corners_useful_3D_homo[i*10 + j] = np.array([9.5 - j, 7.5 - i, 1])
            yz_corners_useful_3D_homo[i*10 + j] = np.array([0.5 + j, 7.5 - i, 1])
            xz_corners_3D_cart[i*10 + j] = np.array([9.5 - j, 0, 7.5 - i])
            yz_corners_3D_cart[i*10 + j] = np.array([0, 0.5 + j, 7.5 - i])

    # TODO : project corners on the calibration plane 1 onto the image
    def project_xz_planar_corners(a_corner_3D):
        return np.dot(Hxz, a_corner_3D)
    xz_corners_2D_homo = np.apply_along_axis(project_xz_planar_corners, axis=1, arr=xz_corners_useful_3D_homo)
    xz_corners_2D = xz_corners_2D_homo[:, :2] / (xz_corners_2D_homo[:, 2]).reshape((-1, 1))
    
    
    # TODO : project corners on the calibration plane 2 onto the image
    def project_yz_planar_corners(a_corner_3D):
        return np.dot(Hyz, a_corner_3D)
    yz_corners_2D_homo = np.apply_along_axis(project_yz_planar_corners, axis=1, arr=yz_corners_useful_3D_homo)
    yz_corners_2D = yz_corners_2D_homo[:, :2] / (yz_corners_2D_homo[:, 2]).reshape((-1, 1))


    # TODO : locate the nearest detected corners
    ref3D = np.concatenate((xz_corners_3D_cart, yz_corners_3D_cart))
    ref2D = find_nearest_corner(np.concatenate((xz_corners_2D, yz_corners_2D)), corners)

    # print(ref3D)
    # print(ref2D)
    return ref3D, ref2D


#  estimate the camera projection matrix
################################################################################
def calibrate3D(ref3D, ref2D):
    # input:
    #    ref3D - a 160 x 3 numpy ndarray holding the 3D coodinates of all the corners
    #            on the 2 calibration planes
    #    ref2D - a 160 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of all the corners in ref3D
    # output:
    #    P - a 3 x 4 numpy ndarray holding the camera projection matrix

    # TODO : form the matrix equation Ap = b for the camera
    # my_checking(img_color, ref2D[:, 0], ref2D[:, 1])
    # print(ref3D)
    
    m, n = ref3D.shape
    A = np.zeros((m*2, 11)) # assume last column is 1, and move to RHS as the b for lstsq;
    b = np.zeros((m*2,))
    
    for i in range(m):
        pt_in_3D = ref3D[i] # (x, y, z)
        homo_coor_3D = np.concatenate((pt_in_3D, [1])) # (x, y, z, 1)
        
        pt_in_2D = ref2D[i] # (u, v)
        
        A[2*i, 0:4] = homo_coor_3D
        A[2*i, 8:] = -pt_in_2D[0] * pt_in_3D
        A[2*i+1, 4:8] = homo_coor_3D
        A[2*i+1, 8:] = -pt_in_2D[1] * pt_in_3D
        b[2*i], b[2*i+1] = pt_in_2D[0], pt_in_2D[1]
        
    # TODO : solve for the projection matrix using linear least squares
    temp = lstsq(A, b, rcond=None)[0].flatten()
    p_hat = np.concatenate((temp, [1])).reshape((3, 4))

    return p_hat

#  decompose the camera calibration matrix in K[R T]
################################################################################
def decompose_P(P) :
    # input:
    #    P - a 3 x 4 numpy ndarray holding the camera projection matrix
    # output:
    #    K - a 3 x 3 numpy ndarry holding the K matrix
    #    RT - a 3 x 4 numpy ndarray holding the rigid body transformation

    # TODO: extract the 3 x 3 submatrix from the first 3 columns of P
    # TODO : perform QR decomposition on the inverse of [P0 P1 P2]
    q, r = qr(inv(P[:, :3]))

    # TODO : obtain K as the inverse of R
    K = inv(r)
        
    # TODO : obtain R as the tranpose of Q
    R = q.T
    
    # TODO : normalize K
    alpha = K[2, 2]
    K /= alpha
    # preprocess K and R into suitable domain
    if K[0, 0] < 0:
        K[0, 0] = -K[0, 0]
        R[0, :] = -R[0, :]
    if K[1, 1] < 0:
        K[:, :2] = -K[:, :2]
        R[1, :] = -R[1, :]

    # TODO : obtain T from P3
    T = (1 / alpha) * (inv(K) @ P[:, 3])

    RT = np.concatenate((R, T.reshape((3, 1))), axis=1)
    
    return K, RT

def my_checking(img_color, a, b):
    print("My checking")
    print(a.shape)
    print(b.shape)
    plt.ion()
    fig = plt.figure("My checking")
    plt.imshow(img_color)
    plt.plot(a, b, 'r.', markersize = 3)
    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  check the planar projective transformations for the 2 calibration planes
################################################################################
def check_H(img_color, Hxz, Hyz):
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    Hxz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the X-Z plane
    #    Hyz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the Y-Z plane

    # plot the image
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)

    # define 3D coordinates of all the corners on the 2 calibration planes
    X_ = np.arange(10) + 0.5 # Y == X
    Z_ = np.arange(8) + 0.5
    X_ = np.tile(X_, 8)
    Z_ = np.repeat(Z_, 10)
    X = np.vstack((X_, Z_, np.ones(80)))

    # project corners on the calibration plane 1 onto the image
    w = Hxz @ X
    u = w[0, :] / w[2, :]
    v = w[1, :] / w[2, :]
    plt.plot(u, v, 'r.', markersize = 3)

    # project corners on the calibration plane 2 onto the image
    w = Hyz @ X
    u = w[0, :] / w[2, :]
    v = w[1, :] / w[2, :]
    plt.plot(u, v, 'r.', markersize = 3)

    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  check the 2D correspondences for the 2 calibration planes
################################################################################
def check_correspondences(img_color, ref2D) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    ref2D - a 160 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of all the corners on the 2 calibration planes

    # plot the image and the correspondences
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)
    plt.plot(ref2D[:, 0], ref2D[:, 1], 'bx', markersize = 5)
    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  check the estimated camera projection matrix
################################################################################
def check_P(img_color, ref3D, P) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    ref3D - a 160 x 3 numpy ndarray holding the 3D coodinates of all the corners
    #            on the 2 calibration planes
    #    P - a 3 x 4 numpy ndarray holding the camera projection matrix

    # plot the image
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)

    # project the reference 3D points onto the image
    w = P @ np.append(ref3D, np.ones([len(ref3D), 1]), axis = 1).T
    u = w[0, :] / w[2, :]
    v = w[1, :] / w[2, :]
    plt.plot(u, v, 'r.', markersize = 3)
    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  pick seed corners for camera calibration
################################################################################
def pick_corners(img_color, corners) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)
    # return:
    #    ref3D - a 8 x 3 numpy ndarray holding the 3D coodinates of the extreme
    #            corners on the 2 calibration planes
    #    ref2D - a 8 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of the corners in ref3D

    # plot the image and corners
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)
    plt.plot(corners[:,0], corners[:,1],'r+', markersize = 5)
    plt.show()

    # define 3D coordinates of the extreme corners on the 2 calibration planes
    ref3D = np.array([(9.5, 0, 7.5), (0.5, 0, 7.5), (9.5, 0, 0.5), (0.5, 0, 0.5),
                      (0, 0.5, 7.5), (0, 9.5, 7.5), (0, 0.5, 0.5), (0, 9.5, 0.5)],
                      dtype = np.float64)
    # ref2D = np.array([(145.0788, 129.7413), (318.3315, 102.5841), (149.0518, 298.0115), (315.0561, 254.6429), 
    #                   (336.2306, 103.5627), (494.1933, 145.1426), (331.6419, 254.9929), (481.9233, 314.6322)], 
    #                  dtype = np.float64)
    ref2D = np.zeros([8, 2], dtype = np.float64)
    for i in range(8) :
        selected = False
        while not selected :
            # ask user to pick the corner on the image
            print('please click on the image point for ({}, {}, {})...'.format(
                  ref3D[i, 0], ref3D[i, 1], ref3D[i, 2]))
            plt.figure(fig.number)
            pt = plt.ginput(n = 1, timeout = - 1)
            # locate the nearest detected corner
            pt = find_nearest_corner(np.array(pt), corners)
            if pt[0, 0] > 0 :
                plt.figure(fig.number)
                plt.plot(pt[:, 0], pt[:, 1], 'bx', markersize = 5)
                ref2D[i, :] = pt[0]
                selected = True
            else :
                print('cannot locate detected corner in the vicinity...')
            # print(ref2D[i])
        
    plt.close(fig)

    return ref3D, ref2D

################################################################################
#  find nearest corner
################################################################################
def find_nearest_corner(pts, corners) :
    # input:
    #    pts - a n0 x 2 numpy ndarray holding the coordinates of the points
    #          (n0 being the number of points)
    #    corners - a n1 x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n1 being the number of corners)
    # return:
    #    selected - a n0 x 2 numpy ndarray holding the coordinates of the nearest_corner
    #               corner

    x = corners[:, 0]
    y = corners[:, 1]
    x_ = pts[:, 0]
    y_ = pts[:, 1]

    # compute distances between the corners and the pts
    dist = np.sqrt(np.square(x.reshape(-1,1).repeat(len(x_), axis = 1) - x_)
                 + np.square(y.reshape(-1,1).repeat(len(y_), axis = 1) - y_))
    # find the index of the corner with the min distance for each pt
    min_idx = np.argmin(dist, axis = 0)
    # find the min distance for each pt
    min_dist = dist[min_idx, np.arange(len(x_))]
    # extract the nearest corner for each pt
    selected = corners[min_idx, 0:2]
    # identify corners with a min distance > 10 and replace them with [-1, -1]
    idx = np.where(min_dist > 10)
    selected[idx, :] = [-1 , -1]
    return selected

################################################################################
#  save K[R T] to a file
################################################################################
def save_KRT(outputfile, K, RT) :
    # input:
    #    outputfile - path of the output file
    #    K - a 3 x 3 numpy ndarry holding the K matrix
    #    RT - a 3 x 4 numpy ndarray holding the rigid body transformation

    try :
        file = open(outputfile, 'w')
        for i in range(3) :
            file.write('{:.6e} {:.6e} {:.6e}\n'.format(K[i,0], K[i, 1], K[i, 2]))
        for i in range(3) :
            file.write('{:.6e} {:.6e} {:.6e} {:.6e}\n'.format(RT[i, 0], RT[i, 1],
                       RT[i, 2], RT[i, 3]))
        file.close()
    except :
        print('Error occurs in writing output to \'{}\'.'.format(outputfile))
        sys.exit(1)

################################################################################
#  load K[R T] from a file
################################################################################
def load_KRT(inputfile) :
    # input:
    #    inputfile - path of the file containing K[R T]
    # return:
    #    K - a 3 x 3 numpy ndarry holding the K matrix
    #    RT - a 3 x 4 numpy ndarray holding the rigid body transformation

    try :
        file = open(inputfile, 'r')
        K = np.zeros([3, 3], dtype = np.float64)
        RT = np.zeros([3, 4], dtype = np.float64)
        for i in range(3) :
            line = file.readline()
            e0, e1, e2 = line.split()
            K[i] = [np.float64(e0), np.float64(e1), np.float64(e2)]
        for i in range(3) :
            line = file.readline()
            e0, e1, e2, e3 = line.split()
            RT[i] = [np.float64(e0), np.float64(e1), np.float64(e2), np.float64(e3)]
        file.close()
    except :
        print('Error occurs in loading K[R T] from \'{}\'.'.format(inputfile))
        sys.exit(1)

    return K, RT

################################################################################
#  load image from a file
################################################################################
def load_image(inputfile) :
    # input:
    #    inputfile - path of the image file
    # return:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)

    try :
        img_color = plt.imread(inputfile)
        return img_color
    except :
        print('Cannot open \'{}\'.'.format(inputfile))
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
        # print('loading {} corners'.format(nc))
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
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 4')
    parser.add_argument('-i', '--image', type = str, default = str(Path(__file__).parent/'grid1.jpg'),
                        help = 'filename of input image')
    parser.add_argument('-c', '--corners', type = str, default = str(Path(__file__).parent/'grid1.crn'),
                        help = 'filename of corner detection output')
    parser.add_argument('-o', '--output', type = str,
                        help = 'filename for outputting camera calibration result')
    args = parser.parse_args()

    print('-------------------------------------------')
    print('COMP3317 Assignment 4 - Camera calibration')
    print('input image : {}'.format(args.image))
    print('corner list : {}'.format(args.corners))
    print('output file : {}'.format(args.output))
    print('-------------------------------------------')

    # load the image
    img_color = load_image(args.image)
    print('\'{}\' loaded...'.format(args.image))

    # load the corner detection result
    corners = load_corners(args.corners)
    print('{} corners loaded from \'{}\'...'.format(len(corners), args.corners))

    # pick the seed corners for camera calibration
    print('pick seed corners for camera calibration...')
    ref3D, ref2D = pick_corners(img_color, corners)

    # estimate planar projective transformations for the 2 calibration planes
    print('estimate planar projective transformations for the 2 calibration planes...')
    H1, H2 = calibrate2D(ref3D, ref2D)
    check_H(img_color, H1, H2)

    # generate correspondences for all the corners on the 2 calibration planes
    print('generate correspondences for all the corners on the 2 calibration planes...')
    ref3D, ref2D = gen_correspondences(H1, H2, corners)
    check_correspondences(img_color, ref2D)

    # estimate the camera projection matrix
    print('estimate the camera projection matrix...')
    P = calibrate3D(ref3D, ref2D)
    print('P = ')
    print(P)
    check_P(img_color, ref3D, P)

    # decompose the camera projection matrix into K[R T]
    print('decompose the camera projection matrix...')
    K, RT = decompose_P(P)
    print('K =')
    print(K)
    print('[R T] =')
    print(RT)
    check_P(img_color, ref3D, K @ RT)

    # save K[R T] to a file
    if args.output :
        save_KRT(args.output, K, RT)
        print('K[R T] saved to \'{}\'...'.format(args.output))

if __name__ == '__main__':
    main()
