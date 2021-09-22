import numpy as np
from numpy.linalg import inv
import scipy.signal as sci                                                  
import rot
import os

def txt2jts(file):
    # Converts a .txt file to .jts format. 
    # file:  file name without extension
    
    filetxt = file + '.txt'
    filejts = file + '.jts'
    with open((filetxt), 'r') as file:
        filedata = file.read()
    
    filedata = filedata.replace(',\n', '\n')
    filedata = filedata.replace('\t', '')

    with open((filetxt), 'w') as file:
        file.write(filedata)

    txt = np.loadtxt(filetxt, delimiter = ',', skiprows = 2)
    hdr = 'JT_EULER_312\n          x_tran,          y_tran,       z_tran,                   z_rot,           x_rot,           y_rot\n'
    np.savetxt(filejts, txt, header = hdr, comments = '', delimiter = ',')
    

def read(jtsfile, smooth='false'):

    # [matrices, N] = read(jtsfile, smooth)
    # Read data from jts file into an array to be used for kinematic and pose calculations.


    # N (output):         Number of poses, equal to the length of matrices.
    # MatrixList (output):  4x4 transformation matrices in a 1x1xN array, where
    #                     matrices[k] is the4x4 matrix for the kth pose. 

    # jtsfile:            Jts file location, with extension name. Can also be in the form
    #                     of a .txt file.
    # smooth:             (Optional) If smooth=true, performs a 4-element boxcar filter.

    # Revision: v2. 2010-07-19, Shang Mu
    #           v3. 2016-05-20, Scott Banks
    #                 - added tweak for y-rotations near +180
    #                 - removed 'raw' option - which I have never used.
    #                 - made smoothing an optional argument.
    # Python: 2021-08-06

        if os.path.splitext(jtsfile)[1] == '.txt':
            txt2jts(os.path.splitext(jtsfile)[0])
            jtsfile = os.path.splitext(jtsfile)[0]+'.jts'

        with open(jtsfile) as file:
            first_line = file.readline().strip()

        if first_line == 'JT_EULER_312' or first_line == 'JTA_EULER_KINEMATICS':
            format = 312
        else:
            raise Exception('read_jts: Not a valid jts file or currently not supported format')
        
        jts = np.loadtxt(jtsfile,skiprows=2,delimiter=',')

        N = jts[:,0].size

    # Deal with trig problems when images are taken from wrong side
    # and y-rotations are close to +/- 180 degrees.

        for k in range(0,N):
            if jts[k,5] > 160.0:
                jts[k,5] = jts[k,5] - 360.0

    # Smooth data
        if smooth == 'true':
            jts = np.concatenate((np.zeros((3,6)),jts))
            jts[0,:] = jts[3,:]
            jts[1,:] = jts[3,:]
            jts[2,:] = jts[3,:]
            a = 1
            b = [.25, .25, .25, .25]

            jts[:,0] = sci.lfilter(b,a,jts[:,0])
            jts[:,1] = sci.lfilter(b,a,jts[:,1])
            jts[:,2] = sci.lfilter(b,a,jts[:,2])
            jts[:,3] = sci.lfilter(b,a,jts[:,3])
            jts[:,4] = sci.lfilter(b,a,jts[:,4])
            jts[:,5] = sci.lfilter(b,a,jts[:,5])

            jts = jts[3:N+3,:]

        matrices = np.zeros((4,4,N))
        for k in range(0,N):
            matrices[0:3,0:3,k] = rot.z(jts[k,3]) @ rot.x(jts[k,4]) @ rot.y(jts[k,5])
            matrices[0:3, 3, k] = jts[k,0:3].T
            matrices[3,3,k] = 1

        matrixList = [[0]] * N
        for k in range(N):
            matrixList[k] = matrices[:,:,k]

        return matrixList, N

def relativePose(fixedBody, movingBody, smooth = 'false'):

    import jtsFunctions as jts
    # [matrices, N] = relativePose(fixedBody, movingBody)
    
    # Poses of the body with respect to the "fixed" body.

    # N:                      Number of poses, equal to the length of matrices.
    # matrices:               4x4 transformation matrices in a 1x1xN matric, so that matrices[k] is
    #                         the 4x4 matrix for the kth pose. 
    # fixedBody, movingBody:  jts files with extensions.
    # Smooth:                 If smooth = true, perform 4x4 boxcar filter.



    if smooth == 'true':
        [moving, n1] = jts.read(movingBody, 'true')
        [fixed, n2] = jts.read(fixedBody, 'true')
    elif smooth == 'false':
        [moving, n1] = jts.read(movingBody)
        [fixed, n2] = jts.read(fixedBody)

    if n1 != n2:
        raise Exception('Input .jts files do not have the same number of frames')
    
    N = n1
    matrices = [[0]] * N

    for k in range(N):
       matrices[k] = inv(fixed[k])  @ moving[k]

    # #Reshape the matrix to be 4x4xN
    # matrices = np.rot90(matrices, k = 3, axes = (0,1))
    # matrices = np.rot90(matrices, k=1, axes = (1,2))
    # matrices = np.flip(matrices, (1,2))

    return np.array(matrices), N  

def getRotations(sequence, matrix, rangerot2=0):
    # Calculate the corresponding Euler angles from a 3x3 rotation matrix 
    # using the specified sequence.

    # Note that this program follows the convention in the field of robotics: 
    # We use column vectors (not row vectors) to denote vectors/points in 3D 
    # Euclidean space:
    #     V_global = R * V_local
    # where R is a 3x3 rotation matrix, and V_* are 3x1 column vectors. 
    # In this convention, for example, an R_z (rotation about z axis) is 
    # [c -s 0; s c 0; 0 0 1] and NOT [c s 0; -s c 0; 0 0 1], (c = cosine and s = sine)

    # We also acknowledge that Euler angle rotations are always about the axes of 
    # the local coordinate system, never the global coordinate system. (If you would 
    # rather recognize the existence of Euler angles with global coordinate system based
    # rotations and further confuse people in the world, simply reverse the order: an x(3 degrees)-y(4deg)-z(5deg)
    # global-base-rotation is simply a z(5)-y(4)-x(3) local based rotation.) Following the equation
    # and convention above, an x-y-z (1-2-3) Euler angle sequence would mean:
    #     V_global = R * V_local
    #              = R_x * R_y * R_x * V_local
    
    # Sequence:       The rotation sequence used for output (e.g., 312, 213)
    
    # Matrix:         The 3x3 rotation matrix (the so-called Direction Cosine
    #                 Matrix, or DCM). 4x4 homogeneous transformation matrix is acceptable, 
    #                 so long as the first 3 rows and columns are a rotation matrix.

    # rangerot2:      Optional. The range of the second rotation in output. It should be a 
    #                 value of 0 (default) or 1. For symmetric Euler sequences, if rangerot2 == 0,
    #                 the 2nd otation is in the range [0, pi]. If rangerot2 == 1, the range is [-pi,0].
    #                 If the second rotation is 0 or pi, singularity occurs.
                    
    #                 For asymmetric Euler sequences:
    #                 if rangerot== 0, the 2nd rotation is in the range [-pi/2, pi/2]; if rangerot==1, 
    #                 the range is [pi/2, pi*3/2]. If the second rotation is +/- pi/2, singularity occurs.

    # Author:         Shang Mu, 2005-2010
    # Revision:       v8, 2010-05-23. 2010-07-08
    # Revision:       Amiya Gupta, 2021-08-09
    # Python:         2021-08-09

    def __c312(M):
        s2 = M[2,1]                 # x rot
        if (rangerot2 == 0):
            rot2 = np.arcsin(s2)
        else:
            rot2 = np.pi-np.arcsin(s2)
        if s2 == any((0,1)):      # singularity
            rot1 = 0
            rot3 = np.arctan2(M[0,2], M[0,0])
        else:
            if rangerot2 == 0:
                rot1 = np.arctan2(-M[0, 1], M[1,1])     # z rot
                rot3 = np.arctan2(-M[2, 0], M[2,2])       # y rot
            else:
                rot1 = np.arctan2(M[0,1], -M[1,1])      # z rot
                rot3 = np.arctan2(M[2,0], -M[2,2])      # y rot
        return rot1, rot2, rot3

    def __c313(M):
        c2 = M[2,2]     #x rot
        if rangerot2 == 0:
            rot2 = np.arccos(c2)
        else:
            rot2 = -np.arccos(c2)
        if c2 == any((-1,1)):     # singularity
            rot1 = 0
            rot3 = np.arctan2(M[2,0],M[2,1])
        else:
            if rangerot2 == 0:
                rot1 = np.arctan2(M[0,2], -M[1,2])      # z rot
                rot3 = np.arctan2(M[2,0],M[2,1])        # y rot
            else:
                rot1 = np.arctan2(-M[0,2],M[1,2])       #z rot
                rot3 = np.arctan2(-M[2,0], -M[2,1])     #y rot
        return rot1, rot2, rot3

    __rot120 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    if rangerot2 != 1 and rangerot2 != 0:
        raise Exception('Invalid value for parameter rangerot2')

    M = matrix[0:3, 0:3]
    sequence = "c" + str(sequence)

    # Asymmetric Sequences
    if sequence == "c312":
        [rot1, rot2, rot3] = __c312(M)
    elif sequence == "c123":
        [rot1, rot2, rot3] = __c312(__rot120.T @ M @ __rot120)
    elif sequence == "c213":
        [rot1, rot2, rot3] = np.negative(np.flip(__c312(M.T)))
    elif sequence == "c231":
        M = __rot120.T @ M @ __rot120
        [rot1, rot2, rot3] = __c312(__rot120.T @ M @ __rot120)
    elif sequence == "c321":
        [rot1, rot2, rot3] = np.negative(np.flip(__c312(__rot120.T @ M @ __rot120)))
    elif sequence == "c132":
        M = __rot120.T @ M @ __rot120 #231
        [rot1, rot2, rot3] = np.negative(np.flip(__c312(__rot120.T @ M @ __rot120)))

    # Symmetric Sequences
    elif sequence == "c313":
        [rot1, rot2, rot3] = __c313(M)
    elif sequence == "c212":
        [rot1, rot2, rot3] = __c313(rot.x(-90).T @ M @ rot.x(-90))
    elif sequence == "c232":
        [rot1, rot2, rot3] = __c313(__rot120 @ M @ __rot120.T)
    elif sequence == "c131":
        M =__rot120 @ M @ __rot120.T
        [rot1, rot2, rot3] = __c313(rot.x(-90).T @ M @ rot.x(-90))
    elif sequence == "c121":
        M = __rot120 @ M @ __rot120.T
        M = rot.x(-90).T @ M @ rot.x(-90)
        [rot1, rot2, rot3] = __c313(rot.x(-90).T @ M @ rot.x(-90))

    else:
        raise Exception('getRotations(): Sequence not yet supported')
        rotations = []
    
    return np.rad2deg(rot1), np.rad2deg(rot2), np.rad2deg(rot3)
    # Sidenote:

    # These methods are used to compute the sequences from 312 and 313 to simplify the program and increase the efficiency.

    # There are many ways to simplify this program:

    # Preliminary (for methods 1 and 2):
    # The 12 different Euler angle sequences can be divided into 4 groups:
    # (123, 231, 312), (321, 213, 132) , (121, 232, 313), (323, 212, 131).
    # (The first two groups can be further combined to a single one using
    # method 3 below. Similarly the last two groups can be united by method 4.) 

    # Simplification method 1:
    # Basic idea: if I have two different coordinate systems on the same rigid
    # body, an x-rotation seen from one coordinate system can be a y-rotation
    # in the other coordinate system.
    # The tool is a rotation matrix:
    #   rot120 = [
    #      0     0     1
    #      1     0     0
    #      0     1     0
    #   ];  (a shift of axis indices, or a 120 degree rotation about [1 1 1]).
    # For any sequence, the three Euler angles can be easily calculated using
    # the same program for any other sequence in the same group (as defined in
    # the "preliminary" section). For example (assuming M is 3x3):
    # getRotations(123, M) == getRotations(312, rot120.'*M*rot120)
    #                      == getRotations(231, rot120*M*rot120.');
    # getRotations(232, M) == getRotations(121, rot120.'*M*rot120)
    #                      == getRotations(313, rot120*M*rot120.').

    # Simplification method 2:
    # There are rules we can follow. For example, s2 or c2 (sine or cosine of
    # the 2nd rotation) is always at the (1st rot)(3rd rot) element in the
    # rotation matrix. The four groups we saw above are the only variations we
    # need to take care of.

    # Simplification method 3:
    # An A-B-C order Euler angle rotations of a body EE with respect to some
    # fixed body FF, can be seen as a C-B-A order rotations of the body FF with 
    # respect to the body EE (but with the negative angles). For example
    # (assuming M is 3x3):
    # getRotations(123, M) == -getRotations(321, M.')(end:-1:1);
    # getRotations(312, M) == -getRotations(213, M.')(end:-1:1).

    # Simplification method 4:
    # Similar to method 1, a relabel of axes utilizing 90 degree rotations is
    # extremely useful for the symmetrical sequences. A 90 degree rotation
    # about either of the two axes in a symmetrical sequence would result in a
    # new valid sequence. For example:
    # getRotations(212, M) == getRotations(232, roty(90).'*M*roty(90))
    #                      == getRotations(313, rotx(-90).'*M*rotx(-90)).
    # This essentially unites all the symmetric sequences.
    # This method could also be used on the asymmetric sequences, but care
    # must be taken to negate the sign of individual angles.