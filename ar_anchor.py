"""
AR anchor with VIO only (no SLAM/loop closure). See ar_anchor_node.py for
the rendering + gesture-control logic; this file just wires the upstream
DepthAI pipeline.

Gestures (focus the OpenCV window):
    pinch + hold (~0.4 s)         drop the anchor 1.5 m ahead
    pinch inside the picture      drag the whole picture
    pinch near a corner           drag just that corner
    pinch with both hands         scale around the midpoint
    both palms open (~1 s)        clear the anchor
    close the window              quit
"""

import time
import depthai as dai

from ar_anchor_node import ARAnchorNode


with dai.Pipeline() as p:
    fps = 30
    width, height = 640, 400

    left = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=fps)
    right = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps)
    imu = p.create(dai.node.IMU)
    stereo = p.create(dai.node.StereoDepth)
    featureTracker = p.create(dai.node.FeatureTracker)
    odom = p.create(dai.node.RTABMapVIO)
    ar = ARAnchorNode("overlay.png", window_title="AR anchor (VIO)")

    imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 200)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    featureTracker.setHardwareResources(1, 2)
    featureTracker.initialConfig.setCornerDetector(dai.FeatureTrackerConfig.CornerDetector.Type.HARRIS)
    featureTracker.initialConfig.setNumTargetFeatures(1000)
    featureTracker.initialConfig.setMotionEstimator(False)
    featureTracker.initialConfig.FeatureMaintainer.minimumDistanceBetweenFeatures = 49

    stereo.setExtendedDisparity(False)
    stereo.setLeftRightCheck(True)
    stereo.setRectifyEdgeFillColor(0)
    stereo.enableDistortionCorrection(True)
    stereo.initialConfig.setLeftRightCheckThreshold(10)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)

    left.requestOutput((width, height)).link(stereo.left)
    right.requestOutput((width, height)).link(stereo.right)
    stereo.rectifiedLeft.link(featureTracker.inputImage)
    featureTracker.passthroughInputImage.link(odom.rect)
    stereo.depth.link(odom.depth)
    featureTracker.outputFeatures.link(odom.features)
    imu.out.link(odom.imu)

    odom.passthroughRect.link(ar.inputImg)
    odom.transform.link(ar.inputTrans)

    p.start()
    while p.isRunning():
        time.sleep(1)
