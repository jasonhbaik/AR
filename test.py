import cv2
import depthai as dai
from OpenGL.GL import *
import glfw

vertices = ((1,1),(1,-1),(-1,-1),(-1,1))

#L_LINEAR

#GL_CLAMP_TO_BORDER 

gluntInitDisplayMode()

#primary image window
with dai.Pipeline() as pipeline:
    # Define source and output
    cam = pipeline.create(dai.node.Camera).build()
    videoQueue = cam.requestOutput((640,400)).createOutputQueue()

    # Connect to device and start pipeline
    pipeline.start()
    while pipeline.isRunning():
        videoIn = videoQueue.get()
        assert isinstance(videoIn, dai.ImgFrame)
        cv2.imshow("video", videoIn.getCvFrame())

        if cv2.waitKey(1) == ord("q"):
            
            




# #To get all camera feeds:
# device = dai.Device()
# with dai.Pipeline(device) as pipeline:
#     outputQueues = {}
#     sockets = device.getConnectedCameras()
#     for socket in sockets:
#         cam = pipeline.create(dai.node.Camera).build(socket)
#         outputQueues[str(socket)] = cam.requestFullResolutionOutput().createOutputQueue()

#     pipeline.start()
#     while pipeline.isRunning():
#         for name in outputQueues.keys():
#             queue = outputQueues[name]
#             videoIn = queue.get()
#             assert isinstance(videoIn, dai.ImgFrame)
#             # Visualizing the frame on slower hosts might have overhead
#             cv2.imshow(name, videoIn.getCvFrame())

#         if cv2.waitKey(1) == ord("q"):
#             break
