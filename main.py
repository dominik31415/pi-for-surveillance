import io
import picamera
import numpy as np
import time
from datetime import datetime
import multiprocessing as mp
import cv2

# ---------------------------------------------
#  main function for acquiring videos and single frame to be used in object detection
#  generally it is a loop of acquiring videos. At regular intervals the video stream is interrupted to submit a single frame for activity detection
#  @param saveQueue: queue of video clips singled out for saving on disk (if state == True), populated by this function
#  @param bkgQueue: queue of single frames, submitted for activity detection, populated by this function
#  @param state: boolean (consumed), when true activity is detected. the acquired clips will be pushed into the saveQueue (otherwise they are discarded)
#  @return None
#  
def acquireVideos(saveQueue, bkgQueue, state):
    logFile  = "/home/pi/Desktop/cam/logAcquire.txt"
    msg = "Launch acquisition process at " + time.strftime("%m%d_%H%M")
    print(msg)
    with open(logFile,'wa') as f:
        f.write(msg)

    camera = picamera.PiCamera(resolution = (800,608), framerate = 25, sensor_mode = 4)
    camera.exposure_mode = 'night'
    camera.drc_strength = 'medium'
    camera.iso = 800
    time.sleep(5) # Wait for the automatic gain control to finish

    dt_s = 2.3 #wait time between two frames to be submitted for activity detection
    clipLength_s = 60 #length of standard video clip in seconds
    with camera as cam:

        while True:
            #new stream
            t = 0
            stream = io.BytesIO()
            while t < clipLength_s:
                cam.start_recording(stream, format='h264')
                
                # if activity was detected, reduce the number of interruptions
                if state.value == 1:
                    cam.wait_recording(10)
                    t = t + 10
                else:
                    cam.wait_recording(dt_s)
                    t = t + dt_s
                cam.stop_recording()

                # interrupt video stream and acquire image for activity detection
                frame = np.empty(800*608*3,np.uint8)
                cam.capture(frame,'rgb', use_video_port=True)
                frame = np.reshape(frame, (608,800,3))
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                bkgQueue.put(frame)

                #this is a heuristic way of adjusting dt_s based on far behind the analysisQueue is
                print("time: %f, queued for analysis: %d" % (t, analysisQueue.qsize()))
                qsize = analysisQueue.qsize()
                if qsize > 4:
                    dt_s = dt_s + 0.5
                elif qsize < 2 and dt_s > 0.5:
                    dt_s = dt_s - 0.5
                else:
                    pass

			# activity was detected --> save this clip
            if state.value == 1:
                saveQueue.put(stream)
                state.value = 0
            else:
                print("discard uneventful stream")

            currentMin = datetime.now().minute
            if 0 == currentMin % 10:
                msg = "Acquisition alive at " + time.strftime("%m%d_%H%M")
                with open(logFile,'wa') as f:
                    f.write(msg)

    saveQueue.put("DONE")
    bkgQueue.put("DONE")
    analysisQueue.put("DONE")
    print("... terminate acquisition process")
    with open(logFile,'wa') as f:
        f.write("... terminate acquisition process")
    return

# ---------------------------------------------
#  main loop for saving clips to disk. the file names encode time & date
#  @param saveQueue : queue of video clips (consumed by this function)
#  @return None
def saveVideos(saveQueue):
    msg = "Launch saving process at " + time.strftime("%m%d_%H%M")
    print(msg)

    onOff = True
    while onOff:
        clip = saveQueue.get()
        if type("DONE") == type(clip):
            onOff = False
        else:
            baseName = time.strftime("%m%d_%H%M%S")
            fileName = "/home/pi/Desktop/data/"+ baseName +".h264"
            print("Saving clip at %s" % baseName)
            with io.open(fileName, 'wb') as outFile:
                outFile.write(clip.getvalue())
                clip.close()


    print("... terminate save process")
    return


#--------------------------------------------------
#  main loop for detecting movement activity
#  this function uses a standard (dynamic) background removal method to find moving objects
#  moving objects are cropped to their bounding boxes and submitted for object detection
#  @param bkgQueue : queue of single frames (consumed by this function)
#  @param analysisQueue : queue of cropped frames (produced by this function), containing only bounding boxes with detected movement
#  @return None
fgbg = cv2.createBackgroundSubtractorMOG2(history = 20)
kernel = np.ones((5,3))
def findMovement(bkgQueue, analysisQueue):
    logFile  = "/home/pi/Desktop/cam/logBKG.txt"
    msg = "Launch bkg process at " + time.strftime("%m%d_%H%M")
    print(msg)
    with open(logFile,'wa') as f:
        f.write(msg)

    onOff = True
    while onOff:
        frame = bkgQueue.get()
        if type("DONE") == type(frame):
            onOff = False
        else:
            frameSmall = cv2.resize(frame,(20,15),interpolation=cv2.INTER_LINEAR)
            mask = fgbg.apply(frameSmall)
            mask = (mask > 0).astype(np.uint8)  # remove background        
            mask = cv2.dilate(mask, kernel) # dilate it a bit
            
            tmp0,allContours,tmp1 = cv2.findContours(mask, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE) # find blobs
            for c in allContours:
                rect = cv2.boundingRect(c)
                rect = np.array(rect, np.uint16)*40
                (x,y,w,h) = rect
                if w < 250: #objects too large are usually a miss-trigger
                    snippet = frame[y:y+h,x:x+w]
                    analysisQueue.put(snippet) # submit those blobs for object detection

            currentMin = datetime.now().minute
            if 0 == currentMin % 10:
                msg = "Movement alive at " + time.strftime("%m%d_%H%M")
                with open(logFile,'wa') as f:
                    f.write(msg)

    print("... terminate movement  process")
    return



# --------------------------------
#  main function for detecting human-like objects
# this function uses two different detectors (built in with open CV)
#  @param img : image, ideally cropped to the ROI
#  @return boolean, true when a human-like object was detected
# the various integers used here probably have to adapted to each use case
classifier = cv2.CascadeClassifier("/home/pi/Desktop/cam/HS.xml") # weights for Haar detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
def detect(img):
    triggered = False
    objects = classifier.detectMultiScale(img, 2, 13)
    if len(objects) > 0:
        triggered = True
    else:
		# two stage setup to keep CPU usage low
        (objects, weights) =  hog.detectMultiScale(img, winStride = (4,4), padding = (8,8), scale = 1.15)
        if len(objects) > 0:
            triggered = True

    if triggered:
        #FOUND A TRIGGER!
        #save snippet if an object was detected, for future reference
        print("TRIGGERED")
        baseName = time.strftime("%m%d_%H%M%S")
        fileName = "/home/pi/Desktop/data/det"+baseName +".jpg"
        cv2.imwrite(fileName,img)
        return True

    # nothing to report
    return False


# ---------------------------------------------
#  main loop for detecting activity
#  @param analysisQueue : queue of cropped frames, each containing mostly a moving object (consumed by this function)
#  @param state : boolean, set to true by this function, when the activity was caused by a human figure
#  @return None
def analyze(analysisQueue, state):
    logFile  = "/home/pi/Desktop/cam/logDetect.txt"
    msg = "Launch detection process at " + time.strftime("%m%d_%H%M")
    print(msg)
    with open(logFile,'wa') as f:
        f.write(msg)

    while True:
        snippet = analysisQueue.get()
        if type("DONE") == type(snippet):
            break
        print("analyzing frame...")

		# this helper function does all the heavy lifting
        if detect(snippet):
            state.value = 1

        currentMin = datetime.now().minute
        msg = "Detection alive at " + time.strftime("%m%d_%H%M")
        with open(logFile,'wa') as f:
            f.write(msg)


    print("... terminate analysis process")
    return


# ---------------------------------------------
if __name__ == '__main__':
    saveQueue = mp.Queue()
    bkgQueue = mp.Queue()
    analysisQueue = mp.Queue()
    state = mp.Value('i', 0)
    pAll = []

    pAll.append(mp.Process(target=acquireVideos, args=(saveQueue,bkgQueue,state)))
    pAll.append(mp.Process(target=findMovement, args=(bkgQueue, analysisQueue,)))
    pAll.append(mp.Process(target=analyze, args=(analysisQueue,state)))
    pAll.append(mp.Process(target=saveVideos, args=(saveQueue,)))

    for pr in pAll:
        pr.start()

    for pr in pAll:
        pr.join()

