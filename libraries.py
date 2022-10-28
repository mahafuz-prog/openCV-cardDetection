import cv2
import numpy as np


def detectCard(img, imgResult):

    # Thresholding main frame
    thresh = getThresh(img, threshLvlLow= 167, threshLvlHigh = 255)
    contours = myContour(thresh)

    i = 0
    for _ in contours:
        area = cv2.contourArea(_)

        # Setting minimum area of contours
        if area >= 5000:
            peri = cv2.arcLength(_, True)
            approx = cv2.approxPolyDP(_, peri*.02, True)
            cv2.drawContours(img, approx, -1, (0,255,0), 8)

            # Specified 4 corner points of a detected card
            if len(approx) == 4:

                # Split detected card from main frame
                cardImage = perspective(approx, imgResult)
                cv2.imshow(f'Card {i+1}', cardImage)
                i+=1

    # Printing total number of detected card on main frame
    cv2.putText(img, f'total cards: {i}', (10,40),cv2.FONT_HERSHEY_SIMPLEX,1, color=(0,255,0), thickness=3)


# Return contours
def myContour(img, mode = cv2.RETR_EXTERNAL, mathod = cv2.CHAIN_APPROX_SIMPLE):
    contours, hierarchy = cv2.findContours(img, mode, mathod)
    return contours


# Return the ordered corner points object
def perspectiveReorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print(myPointsNew)
    return myPointsNew


# Crop & return the selected card
def perspective(approx, img):
    approx = perspectiveReorder(approx)

    # Specified width & height of a card
    width, height = 250, 350
    pts1 = np.float32(approx)
    pts2 = np.float32([[0,0],[width, 0], [0, height],[width,height]])

    mat = cv2.getPerspectiveTransform(pts1, pts2)
    imgCroped = cv2.warpPerspective(img, mat, (width, height))

    return imgCroped
    

# Return thresholded image
def getThresh(img, threshLvlLow= 10, threshLvlHigh = 255):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (1,1), 2)

    retval, thresh = cv2.threshold(imgBlur, threshLvlLow, threshLvlHigh, cv2.THRESH_BINARY)
    return thresh