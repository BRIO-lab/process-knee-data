import numpy as np
from numpy.linalg import norm

def rotsc(axis, sine, cosine, centerORsize=3):

    # The sine-cosine version of rot(). 
    # Calculates the 3x3 rotation matrix or the 4x4 homogeneous transformation
    # matrix. The type of the returned matrix is controlled by by the 'centerOrsize'
    # parameter. 

    # axis:         Direction of the axis of rotation. 
    #               1 = x, 2 = y, 3 = z. Or the axis of rotation represented by a 
    #               3-vector (does not need to be normalized). 
    #               See parameter 'centerORsize' for more info.

    # sine, cosine: Rotation angle's sine and cosine.

    # CenterORsize: Optional. Center of rotation (i.e., a point on the axis of
    #               rotation), or the size of the returned matrix. 
    #               - If a center of rotation is provided, the returned matrix 
    #               will be a 4x4 transformation amtrix.
    #               - If omitted, or the specified value is a scalar, center of 
    #               rotation is assumed to be at [0,0,0]. The scalar is 
    #               interpreted as the desired size of the returned matrix, 
    #               and can be the value of 3,4,9, (i.e., 3*3), or 16. 
    #               If omitted, the returned matrix is a 3x3 rotation matrix.

    # Author: Shang Mu, 2008-2009.
    # Revision: v2. 2009-06-11
    # Python v.     2021-08-06

    c = cosine
    s = sine
    v = 1-c 

    if type(axis) == int:
        m = np.array([[0],[0],[0]])
        m[axis-1] = 1
    else:
        m = np.array(axis)
        m = m/norm(m,2)

    # Rotation part of the matrix

    if type(centerORsize)==int and (centerORsize==3 or centerORsize == 9):
        R = np.array([
            [m[0,0]*m[0,0]*v+c,         m[0,0]*m[1,0]*v-m[2,0]*s,       m[0,0]*m[2,0]*v+m[1,0]*s],
            [m[0,0]*m[1,0]*v+m[2,0]*s,  m[1,0]*m[1,0]*v+c,              m[1,0]*m[2,0]*v-m[0,0]*s],
            [m[0,0]*m[2,0]*v-m[1,0]*s,  m[1,0]*m[2,0]*v+m[0,0]*s,       m[2,0]*m[2,0]*v+c],         
        ])
    else:
        if type(centerORsize)==int and centerORsize != 4 and centerORsize != 16:
            raise Exception("rot()/rotsc(): Invalid centerORsize parameter.")
        R = np.array([
            [m[0,0]*m[0,0]*v+c,         m[0,0]*m[1,0]*v-m[2,0]*s,       m[0,0]*m[2,0]*v+m[1,0]*s,    0],
            [m[0,0]*m[1,0]*v+m[2,0]*s,  m[1,0]*m[1,0]*v+c,              m[1,0]*m[2,0]*v-m[0,0]*s,    0],
            [m[0,0]*m[2,0]*v-m[1,0]*s,  m[1,0]*m[2,0]*v+m[0,0]*s,       m[2,0]*m[2,0]*v+c,           0],
            [0,                         0,                              0,                           1]         
        ])

    # Add the translation
    if type(centerORsize) !=  int:
        t = np.array(centerORsize)
        t.shape = (3,1)
        R[0:3,3:4] = t
        tmp = np.identity(4)
        tmp[0:3,3:4] = -t
        R = np.matmul(R,tmp)
    
    return R


def rotate(axis, angle, centerORsize=3):
    # Calculate the 3x3 rotation matrix or the 4x4 homogenous
    # transformation matrix. The type of the returned matrix is
    # controlled by 'centerORsize' parameter.

    #     axis:       The direction of the acis of rotation.
    #                 1=x, 2=y, 3=z. Or the axis of rotation represented
    #                 by a 3-vector (does not need to be normalized to length
    #                 of 1). See parameter 'centerORsize' for more info.

    # angle:          in degrees.

    # centerORsize:   Optional. Center of rotation (i.e., a point on the axis of
    #                 rotation), or the size of the returned matrix. 
    #                 # If a center of rotation is provided, the returned matrix will
    #                 be a 4x4 transformation matrix.
    #                 # If omitted, or a scalar, center of rotation is assumed to be at
    #                 [0,0,0]. The scalar is interpreted as the desired size of the returned
    #                 matrix, and can be the value of 3, 4, 9 (i.e., 3x3) or 16. 
    #                 If omitted, the returned matrix is a 3x3 rotation matrix.

    # Examples:
    #         30deg rotation about the y axis:
    #             R = rot(2,30,3); R = rot([0;1;0],30,3)
    #             T = rot(2,30,4*4);
    #         30deg rotation about axis x=z=1
    #             T = rot(2, 30, [1;0;1]);

    # Author: Shand Mu, 2008-2009
    # Revision: v2. 2009-06-11
    # Python v1.     2021-08-06

    sine = np.sin(np.deg2rad(angle))
    cosine = np.cos(np.deg2rad(angle))
    R = rotsc(axis, sine, cosine, centerORsize)

    return R

def x(angle, centerORsize=3):

    # Calculate a 3x3 rotation matrix or 4x4 homogeneous transformation
    # matrix of a rotation about x axis (or a parallel axis).

    # Author: Shang Mu, 2008-2009
    # Revision: v1. 2009-06-11
    # Python v1. 2021-08-06

    R = rotate(1, angle, centerORsize)
    return R

def y(angle, centerORsize=3):

    # Calculate a 3x3 rotation matrix or 4x4 homogeneous transformation
    # matrix of a rotation about y axis (or a parallel axis).

    # Author: Shang Mu, 2008-2009
    # Revision: v1. 2009-06-11
    # Python v1. 2021-08-06

    R = rotate(2, angle, centerORsize)
    return R

def z(angle, centerORsize=3):

    # Calculate a 3x3 rotation matrix or 4x4 homogeneous transformation
    # matrix of a rotation about x axis (or a parallel axis).

    # Author: Shang Mu, 2008-2009
    # Revision: v1. 2009-06-11
    # Python v1. 2021-08-06

    R = rotate(3, angle, centerORsize)
    return R