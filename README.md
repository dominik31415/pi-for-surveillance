# pi-for-surveillance
Using a raspberry pi &camera as surveillance system that only triggers on predefined objects (in my case pedestrians)

# Requirements
* python 3.6
* standard modules, OpenCV 2.4
* raspberry pi with picamera module (I used NOIR v2)
* the only "exotic" OpenCV functions used are: createBackgroundSubtractorMOG2, CascadeClassifier, HOGDescriptor
* static camera setup

# What does it do?
This script uses the picamera module to continuously record videos. Frames are repeatedly analyzed for motion & object detection (the script uses an HAAR detector trained on pedestrians). The videos are only stored when a pedestrian was detected.

# Details
Object detection is computationally (somewhat) expensive and a bit too much to handle for a raspberry out of the box. For this reason I wrote my script to rely on multiprocessing to spread out video acquisition, object detection and video storage onto multiple cores.
There are four core processes, executing different roles:

| process name                |role|
|----------------|-------------------------------|
|main|launches all other processes      |
|acquireVideos|main process interacting with camera. it has two tasks: video acquisition and short exposure frame grabbing (for lower motion blur). The single frames are repeatedly forwarded for object detection in findMovement(). videos are usually discarded unless the state flag was set to True by analyze()
|findMovement|listens to the stream of grabbed single frames, performs a createBackgroundSubtractorMOG2 background removal and forwards ROIs to analyze() when movement was detected |
|analyze|listens to the stream of singled out ROIs and applies two different human detectors, HOG and HAAR based (which are delivered with OpenCV and the attached HS.xml file). If one of the ROIs contains a pedestrian, the state flag is set to true. Also saves relevant snapshots for trouble shooting |
|detect|helper function for analyze() |
|saveVideos|Listens to the stream of singled out video clips and saves them on disk. |

# Comments
* The detect() function contains several hyper-parameters used for object detection. After tuning them correctly the system generally works very reliably. It has an extremely low false positive rate (cars being the main exception). 
* The system struggles at low light (dusk & night) but works well during dawn & daytime or when there is a direct light source illuminating the area. Using an IR light source & camera works too.
* I recommend to combine this script with a cronjob that restarts the pi & script daily
* A second cronjob could clean old videos from disk
