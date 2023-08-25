import raycasting
import SRDF
import camera

ITERATION_NUM = 1000

def main():
    #main pipeline loop

    #load files, create class

    ImageSet = camera.Image()
    cameraSet = camera.Camera()

    #initial depth maps
    ImageSet.getdepthmaps()

    #loop starts
    for x in range(ITERATION_NUM):
        #raycasting combined with sampling + SRDF
        #do i need to always generate new rays?
        #make the raycasting and overall function 
        raycasting.raysampling(cameraSet)
        SRDF.srdf()

        #update depthmaps
        ImageSet.updatedepthmaps()

    #output

main()