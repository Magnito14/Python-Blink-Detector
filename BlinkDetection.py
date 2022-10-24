# https://github.com/Magnito14/Python-Blink-Detector.

import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
from pynput.keyboard import Key, Controller

keyboard = Controller()
vidCap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [0, 40], invert=True)
leftEyeLandmarks = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blinkCounter = 0
counter = 0
blinkColor = (255, 0, 255)
blinkColor2 = (0, 255, 0)

while True:
    success, frame = vidCap.read()
    # Exits the application if frame not found.
    if not success:
        break

    frame, faces = detector.findFaceMesh(frame, draw=False)

    if faces:
        face = faces[0]
        # Draws every landmark on the left eye.
        for i in leftEyeLandmarks:
            cv2.circle(frame, face[i], 5, blinkColor, cv2.FILLED)

        # Vertical and Horizontal keypoints.
        topEyelid = face[159]
        bottomEyelid = face[23]
        leftCorner = face[130]
        rightCorner = face[243]
        lengthVer, _ = detector.findDistance(topEyelid, bottomEyelid)
        lengthHor, _ = detector.findDistance(leftCorner, rightCorner)

        # Draws the vertical line.
        cv2.line(frame, topEyelid, bottomEyelid, blinkColor2, 3)
        # Draws the horizontal line.
        cv2.line(frame, leftCorner, rightCorner, blinkColor2, 3)

        # Blinking ratio.
        ratio = int((lengthVer / lengthHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 5:
            ratioList.pop(0)
        # Blinking ratio average.
        ratioAvg = sum(ratioList) / len(ratioList)

        # Waits 10 frames before accepting blink.
        if ratioAvg < 35 and counter == 0:
            blinkCounter += 1
            blinkColor = (0, 255, 0)
            blinkColor2 = (255, 0, 255)
            counter = 1
        if counter != 0:
            counter += 1
        if counter > 10:
            counter = 0
            blinkColor = (255, 0, 255)
            blinkColor2 = (0, 255, 0)

        # If the eyelids are closed, press the spacebar.
        if ratio < 35:
            keyboard.press(Key.space)
        # If the eyelids are open, release the spacebar.
        if ratio > 35:
            keyboard.release(Key.space)

        # Draws rectangle containing blink counter to screen.
        cvzone.putTextRect(
            frame, f"Blink Count: {blinkCounter}", (28, 45), scale=2, thickness=2
        )

        # Shows the frame and the plot on top of eachother.
        framePlot = plotY.update(ratioAvg, blinkColor)
        frame = cv2.resize(frame, (640, 480))
        frameStack = cvzone.stackImages([frame, framePlot], 1, 1)
    else:
        # Only draw the frame.
        frame = cv2.resize(frame, (640, 480))
        frameStack = cvzone.stackImages([frame, frame], 1, 1)

    frame = cv2.resize(frame, (640, 480))
    # Opens video captured window.
    cv2.imshow("Blink Detector Window", frameStack)
    # Exits the window if "e" key is pressed.
    if cv2.waitKey(1) & 0xFF == ord("e"):
        break
vidCap.release()
cv2.destroyAllWindows()
