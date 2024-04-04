import SimpleITK as sitk 
import numpy as np 
import pandas as pd 
import os 
import cv2
from pathlib import Path 
import subprocess
# from proSeg.proSeg import runSegmentation

class MedImageProcessingUtil(object):
    def __init__(self):
        pass 

    @staticmethod
    def copyImageHeaders(toImage,fromImage):
        toImage.SetSpacing(fromImage.GetSpacing())
        toImage.SetDirection(fromImage.GetDirection())
        toImage.SetOrigin(fromImage.GetOrigin())
        return toImage

    @staticmethod
    def cv2inpaint(ipatch,mpatch = None):

        if ipatch.ndim == 2: 
            ipatch = ipatch.astype(np.float32)
        
        elif ipatch.ndim == 3: 
            ipatch = ipatch.astype(np.uint8)
        
        if ipatch[ipatch == 0].size > 0 :
            ipatch = np.pad(ipatch, 1, mode='constant', constant_values=0)

            if mpatch is None:
                mpatch = np.zeros(ipatch.shape)
                mpatch[ipatch == 0] = 1 
                
            else:
                mpatch = np.pad(mpatch, 1, mode='constant', constant_values=0)

            mpatch = mpatch.astype(np.uint8)

            dst = cv2.inpaint(ipatch,mpatch,3,cv2.INPAINT_NS)
            dst = dst[1:-1,1:-1]
        else:
            dst = ipatch
    
        return dst


    @staticmethod
    def segmentProstate(inputpath,capsulefilepath,zonefilepath):
        runSegmentation(inputpath, capsulefilepath, zonefilepath)
        # exec(open(str(Path(fr"""python /Volumes/GoogleDrive/My Drive/Projects/Code_general/proSeg/proSeg.py "{inputpath}" "{capsulefilepath}" "{zonefilepath}" """))).read())

        # cmd = 'C:/Scripts/proSeg/proSeg.exe "{}" "{}" "{}"'.format(inputpath,capsulefilepath,zonefilepath)
        # subprocess.call(cmd,shell=True)


    @staticmethod
    def registerImages(fixedpath,movingpath,parameterpath,outputfolder,fixedmaskpath=None,movingmaskpath=None):

        cmd = None 

        # fixedpath = os.path.abspath(fixedpath)
        # movingpath = os.path.abspath(movingpath)
        # parameterpath = os.path.abspath(parameterpath)
        # outputfolder = os.path.abspath(outputfolder)
        
        # if fixedmaskpath is not None:
        #     fixedmaskpath = os.path.abspath(fixedmaskpath)

        # if movingmaskpath is not None:
        #     movingmaskpath = os.path.abspath(movingmaskpath)

        cmd = "export PATH=/Users/amogh3892/Documents/Softwares/elastix/elastix_macosx64_v4/bin:$PATH\nexport DYLD_LIBRARY_PATH=/Users/amogh3892/Documents/Softwares/elastix/elastix_macosx64_v4/lib:$DYLD_LIBRARY_PATH\n"

        if (fixedmaskpath and movingmaskpath) is not None:
            cmd = cmd + r'elastix -f "{}" -fMask "{}" -m "{}" -mMask "{}" -p "{}" -out "{}"'.format(fixedpath,fixedmaskpath,movingpath,movingmaskpath,parameterpath,outputfolder)
        elif fixedmaskpath is not None :
            cmd = cmd + r'elastix -f "{}" -fMask "{}" -m "{}" -p "{}" -out "{}"'.format(fixedpath,fixedmaskpath,movingpath,parameterpath,outputfolder)
        else:
            cmd = cmd + r'elastix -f "{}" -m "{}" -p "{}" -out "{}"'.format(fixedpath,movingpath,parameterpath,outputfolder)
            
        if not cmd is None:
            subprocess.call([cmd],shell=True) 
            # subprocess.call(['cmd', '/c', cmd])
        else:
            print("Masks not provided properly")

    @staticmethod
    def transformImages(movingpath,transformationpath,outputfolder,mask=None):
    
        cmd = "export PATH=/Users/amogh3892/Documents/Softwares/elastix/elastix_macosx64_v4/bin:$PATH\nexport DYLD_LIBRARY_PATH=/Users/amogh3892/Documents/Softwares/elastix/elastix_macosx64_v4/lib:$DYLD_LIBRARY_PATH\n"


        if mask is not None:
            # addline = "(FinalBSplineInterpolationOrder 0)"

            with open(fr"{transformationpath}/TransformParameters.0.txt","r") as infile:
                trans = infile.read()
            infile.close()

            # trans = trans + "\n" + addline
            trans = trans.replace('(ResampleInterpolator "FinalLinearInterpolator")','(ResampleInterpolator "FinalNearestNeighborInterpolator")')
            trans = trans.replace('(ResampleInterpolator "FinalBSplineInterpolator")','(ResampleInterpolator "FinalNearestNeighborInterpolator")')
            trans = trans.replace("(FinalBSplineInterpolationOrder 3)","(FinalBSplineInterpolationOrder 0)")

            with open(fr"{transformationpath}/TransformParametersMASK.0.txt","w") as infile:
                infile.writelines(trans)
            infile.close()

            cmd = cmd + "transformix -in {} -tp {}/TransformParametersMASK.0.txt -out {}".format(movingpath,transformationpath,outputfolder)
            # subprocess.call(['cmd', '/c', cmd])
            subprocess.call([cmd],shell=True) 


        else:
            cmd = cmd + "transformix -in {} -tp {}/TransformParameters.0.txt -out {}".format(movingpath,transformationpath,outputfolder)
            # subprocess.call(['cmd', '/c', cmd])
            subprocess.call([cmd],shell=True) 


        # cmd2 = "transformix -in {} -tp {}/TransformParametersMASK.0.txt -out {}/gt1".format(gtpath2,outputfolder,outputfolder)



