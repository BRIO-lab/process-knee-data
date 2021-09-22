import numpy as np
from numpy.linalg import inv
import jtsFunctions as jts
import stl          
import vedo as vedo
import subprocess
import cv2
import scipy.io as sci
import os


def kneeData(stlFem, stlTib, jtsFem, jtsTib, side, videoFileName, dataFileName, homeDir, zeroZ = 0):
#     Program renders to video the kinematics of femoral and tibial imlant models
#     whose kinematics are found in related JTS files. The output variable is a 
#     list containing calculated tibiofemoral kinematics, lowest condylar point trajectories,
#     center of condyle projections on the tibial transverse surface, the calculated average
#     center of rotation over activity in tibial reference frame, and the AP and ML extents
#     of the distal femur and proximal tibia.

#     Assumes the x-axis of implant models is the AP direction, y-axis is SI, and Z is medial.O_NOFOLLOW

#     Input arguments:
#                        # stlFem:          Binary format STL model of femoral component
#                        # stlTib:          Binary format STL model of tibial component
#                        # jtsFem:          JointTrack output file for femur (.jts)
#                        # jtsTib:          JointTrack output file for tibia (.jts)
#                        # side:            Either 'left' or 'right'
#                        # videoFileName:   Name of output file for video
#                        # dataFileName:    Name of output for data (.csv file)
#                        # zeroZ:           Boolean to zero z translations (default 0)
            
#     Output arguments:
#          # Out:     CSV file containing all output data. The data fields include:
#                         # video:        Boolean indicating whether video creation was successful.
#                         # kin           Joint kinematics (x,y,z) translations of 
#                                         femur relative to tibia, and (flex, abduc, ext)
#                                         rotation of tibia relative to femur.
#                         # med           (x,y,z) array of medial lowest points in tibial 
#                                         reference frame.
#                         # lat           (x,y,z) array of lateral lowest points in tibial 
#                                         reference frame. 
#                         # CoR           Average center of rotation over entire activity               
#                                         in tibial reference frame.
#                         # dims          AP and ML extents of distal femur.
#                         #tdims          AP and ML extents of proximal tibia.

#     Example Call: 

#                 JR_out = process.kneeData('JRightFem.stl','JRightTib.stl','JRightFem.jts', ...
#                         'JRightTib.jts','right','JRight_test', 'JRight_data')
    
#     By Scott A. Banks, 2016
#     Python v: Amiya Gupta, 2021

    def __Sphere(r,s=30):
        # Create sphere object with vertices (x,y,z) and faces.
        # r:    radius
        # s:    resolution, as interval used to calculate spherical coordinates (default 45). 
        #       Must be divisible by 3 when squared to ensure triangles can be created. 

        if np.mod(s*s,3) != 0:
            raise Exception('Invalid resolution. Ensure resolution squared is divisible by 3')
        pi = np.pi
        cos = np.cos
        sin = np.sin
        N = int(s*s)
        s = complex(s)
        phi, theta = np.mgrid[0.0:pi:s, 0.0:2.0*pi:s]
        x = r*sin(phi)*cos(theta)
        y = r*sin(phi)*sin(theta)
        z = r*cos(phi)

        x = x.reshape(N,1)
        y = y.reshape(N,1)
        z = z.reshape(N,1)
        

        faces = np.array(range(N))
        faces = faces.reshape(int(N/3),3)

        return x,y,z,faces

    def __intersect(P0,P1):
        # P0 and P1 are NxD arrays defining N lines.
        # D is the dimension of the space. This function 
        # returns the least squares intersection of the N
        # lines from the system.
        
        # generate all line direction vectors 
        P1 = P1.T
        P0 = P0.T
        n = (P1-P0)/np.linalg.norm(P1-P0,axis=1)[:,np.newaxis] # normalized

        projs = np.eye(n.shape[1]) - n[:,:,np.newaxis]*n[:,np.newaxis] 

        R = projs.sum(axis=0)
        q = (projs @ P0[:,:,np.newaxis]).sum(axis=0)

        p = np.linalg.lstsq(R,q,rcond=None)[0]

        return p



# Load STL Files
    fVerts = stl.mesh.Mesh.from_file(stlFem)
    N = fVerts.points.shape[0]
    fVerts = fVerts.points.reshape(N*3, 3)
    fVerts = np.vstack((fVerts.T, [1] * N*3))
    ffaces = np.array(range(N*3))
    ffaces = ffaces.reshape(N,3)


    tVerts = stl.mesh.Mesh.from_file(stlTib)
    N = tVerts.points.shape[0]
    tVerts = tVerts.points.reshape(N*3,3)
    tVerts = np.vstack((tVerts.T, [1]*N*3))
    tfaces = np.array(range(N*3))
    tfaces = tfaces.reshape(N,3)

# Get Proximal tibial dimensions to scale CoR
    ttop = np.amax(tVerts[1,:]) # Find top of the tibial plateau
    tind = [i for i, v in enumerate(tVerts[1,:]) if v > ttop-20] # Tray is probably top 20mm or so
    tdims = [np.amin(tVerts[0,tind]), np.amax(tVerts[0,tind]), np.amin(tVerts[2, tind]), np.amax(tVerts[2, tind])]

# Get distal femoral dimensions
    fbottom = np.amin(fVerts[1,:]) 
    fbind = [i for i, v in enumerate(fVerts[1,:]) if v > fbottom-20]
    fdims = [np.amin(fVerts[0,fbind]), np.amax(fVerts[0,fbind]), np.amin(fVerts[2, fbind]), np.amax(fVerts[2, fbind])]

# Compute Knee Kinematics
    # Assume that if z-trans are to be zeroed, then data also should be smoothed.
    smooth = 'false'
    if zeroZ == 1:
        smooth = 'true'

    # Get transformation matrices for relative poses
    [poses, N] = jts.relativePose(jtsFem, jtsTib, smooth)

    #Convert transformation matrices into translations and Euler angle rotations
    result = np.zeros((N,6))
    for k in range(N):
        result[k,0:3] = ((np.negative(poses[k][0:3,0:3].T)) @ poses[k][0:3,3].T)       # tibia based translations 
        result[k,3:6] = jts.getRotations(312, poses[k][:,:])                           # femur based rotations

    # Modify signs to keep left/right consistency per clinical terms
    if side == 'left':
        result[:,3] = np.negative(result[:,3])  # flexion (+)
        result[:,5] = np.negative(result[:,5])  #tibial internal (+)/External (-) rotation
    else:
        result[:,0] = np.negative(result[:,0])       # a(+)/p(-) position
    
# Lowest femur points relative to tibial transverse plane
    # Determine the medial and lateral indices of the femoral component.
    medialIndices = [i for i, v in enumerate(fVerts[2,:]) if v > 0]
    lateralIndices =[i for i, v in enumerate(fVerts[1,:]) if v < 0]
    medialLowest = np.zeros((N,3))
    lateralLowest = np.zeros((N,3))

# Set up rendering window
    clr = [0.4,0.4,1]
    plt = vedo.Plotter(N=4, bg = clr, size = (640,640), interactive = False, sharecam = False)

    #Set up video file
    videoFileMP4 = homeDir + videoFileName + ".mp4"
    video = vedo.Video(videoFileMP4, fps = 7, backend = 'opencv')
    
# Make dumbbell figure and test points for CoR Calculations
    width = np.amax(fVerts[2,:])-np.amin(fVerts[2,:])
    scale = width/12

    # Create a sphere
    [x, y, z, bfaces] = __Sphere(1,30)

    z1 = scale*z + width/3.5
    z2 = scale*z - width/3.5
    y = scale*y
    x = scale*x

    bverts1 = np.hstack((x,y,z1,np.ones((len(x),1))))
    bverts2 = np.hstack((x,y,z2,np.ones((len(x),1))))

    cverts = np.array([[0,0],[0,0],[width/3.5, -width/3.5], [1,1]])

    cpts1 = np.array([[0]]*3)
    cpts2 = cpts1

# Start Rendering Loop
    for k in range(N):
        tmpPose = poses[k][:,:]

        # Zero z-translation -- be careful with this!!!!
        if zeroZ == 1:
            tmpPose[2,3] = 0

        # Determine the pose at frame k
        femTmpVerts = inv(tmpPose) @ fVerts
        femVertices = femTmpVerts[0:3,:].T
        tibTmpVerts = tVerts[0:3,:].T
        ballTmpVerts1 = inv(tmpPose) @ bverts1.T
        ballTmpVerts1 = ballTmpVerts1[0:3,:].T

        ballTmpVerts2 = inv(tmpPose) @ bverts2.T
        ballTmpVerts2 = ballTmpVerts2[0:3,:].T

        cvertsTmp = inv(tmpPose) @ cverts

        # Determine center points and concatenate output vectors
        if k == 0:
            cpts1 = cvertsTmp[0:3,0]
            cpts2 = cvertsTmp[0:3,1]
            cpts1 = cpts1.reshape(1,3)
            cpts2 = cpts2.reshape(1,3)
            kin = [tmpPose]
            kin = np.array(kin).reshape(4,4)
        else:
            tmp = cvertsTmp[0:3,0]
            cpts1 = np.vstack((cpts1, tmp))
            tmp = cvertsTmp[0:3, 1]
            cpts2 = np.vstack((cpts2, tmp))
            kin = np.hstack((kin, tmpPose))
        Pint = np.array([[0]]*3)

        # Get lowest points on each condyle
        medialCondyle = femTmpVerts[:,medialIndices]
        lateralCondyle = femTmpVerts[:, lateralIndices]

        # Lowest point (minimum y)
        index = np.argmin(medialCondyle[1,:])
        medialLowest[k,:] = medialCondyle[0:3,index].T
        index = np.argmin(lateralCondyle[1,:])
        lateralLowest[k, :] = lateralCondyle[0:3, index].T

        # Calculate Center of Rotation from !! condyle centers !!
        if k != 0:
            Pint = __intersect(cpts1.T, cpts2.T)  
            Pint = np.array(Pint)

        # Rendering Code
        b1 = vedo.pointcloud.fitSphere(ballTmpVerts1)
        b2 = vedo.pointcloud.fitSphere(ballTmpVerts2)
        b1.color([256/256, 128/256, 0/256])
        b2.color([256/256, 128/256, 0/256])

        fc = vedo.mesh.Mesh([femVertices, ffaces], c = [236/256, 219/256, 162/256], alpha = 0.8)
        tc = vedo.mesh.Mesh([tibTmpVerts, tfaces], c = [236/256, 219/256, 162/256])

        # Upper Left
        plt.show(fc,b1, b2,tc, at = 0, bg = clr)

        # Upper Right

        fcl = fc.clone()
        b1l = b1.clone()
        b2l = b2.clone()
        tcl = tc.clone()

        fcl.rotateY(180)
        tcl.rotateY(180)
        b1l.rotateY(180)
        b2l.rotateY(180)

        plt.show(fcl,tcl, b1l, b2l, at = 1, bg = clr)

        # Lower Left
        fcl = fc.clone()
        b1l = b1.clone()
        b2l = b2.clone()
        tcl = tc.clone()

        fcl.rotateY(90)
        b1l.rotateY(90)
        b2l.rotateY(90)
        tcl.rotateY(90)

        plt.show(fcl,b1l,b2l,tcl, at = 2, bg = clr)

        # Lower Right
        fcl = fc.clone().alpha(0.2)
        b1l = b1.clone()
        b2l = b2.clone()
        tcl = tc.clone()

        c1 = vedo.pointcloud.Points(cpts1, r = 8, c = 'blue')
        c2 = vedo.pointcloud.Points(cpts2, r = 8, c = 'red')

        fcl.rotateY(90)
        b1l.rotateY(90)
        b2l.rotateY(90)
        tcl.rotateY(90)
        c1.rotateY(90)
        c2.rotateY(90)

        fcl.rotateX(90)
        b1l.rotateX(90)
        b2l.rotateX(90)
        tcl.rotateX(90)
        c1.rotateX(90)
        c2.rotateX(90)

        fcl.rotateZ(180)
        b1l.rotateZ(180)
        b2l.rotateZ(180)
        tcl.rotateZ(180)
        c1.rotateZ(180)
        c2.rotateZ(180)
        
        # Ignore Pint if it is out of range
        if Pint[2] < 50 and Pint[2] > -50:
            p = vedo.pointcloud.Point(pos = Pint, r = 15, c ='blue')
            p.rotateY(90)
            p.rotateX(90)
            p.rotateZ(180)
            plt.show(fcl,b1l,b2l,tcl, c1, c2, p, at = 3, bg = clr)
        else:
            plt.show(fcl,b1l,b2l,tcl, c1, c2, at = 3, bg = clr)

        video.addFrame()
        
        

    video.close()

    fileMOV = homeDir + videoFileName + ".mov"
    # Convert to .mov and back again, otherwise it won't encode correctly for windows media player
    a = subprocess.call(['ffmpeg', '-i', videoFileMP4, '-f', 'mov', fileMOV, '-y', '-nostats', '-loglevel', 'panic'])
    a = subprocess.call(['ffmpeg', '-i', fileMOV, '-f', 'mov', videoFileMP4, '-y', '-nostats', '-loglevel', 'panic'])  
    # os.remove(fileMOV)

    dict = {
        "video": 1,
        "kin": result,
        "med": medialLowest,
        "lat": lateralLowest,
        "CoR": Pint,
        "fdims": fdims,
        "tdims": tdims
    }
    sci.savemat(homeDir + dataFileName, mdict = {"kneeData": dict})
    np.savetxt(homeDir + "/kin_gt.csv", result, delimiter = ',', fmt = '%.4f')
