import process
import os
import glob
# Example Call
import matplotlib

homeDir = "C:/Users/ajensen123/Desktop/JTML/Lima/"
MOVEMENT_LIST = os.listdir(homeDir)

for movt in MOVEMENT_LIST:
    data_dir = homeDir + movt + "/"
    fem_kin = data_dir + "/fem.jts"
    tib_kin = data_dir + "/tib.jts"
    fem_stl = data_dir + glob.glob1(data_dir, "*fem*.stl")[0]
    tib_stl = data_dir + glob.glob1(data_dir, "*tib*.stl")[0]
    video = movt + "_video_gt"
    side = 'left'
    file = movt + "_data_gt.mat"
    
    #process.kneeData(fem_stl, tib_stl, fem_kin, tib_kin, side, video, file, data_dir)