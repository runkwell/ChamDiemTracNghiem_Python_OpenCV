import cv2
import numpy as np
import utlis

##############################
path = "1.jpg"
widthImg = 700
heightImg = 700
questions = 5
choices = 5
ans = [1,2,0,1,4]
##############################


img = cv2.imread(path)

#PREPROCESSING
img = cv2.resize(img, (widthImg, heightImg))
imgContours = img.copy()
imgFinal = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv2.Canny(imgBlur,10,50)

#FINDING AL CO NTOUR
contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,contours,-1,(0,255,0),10)
#FIND RECTANGLES
rectCon = utlis.rectContour(contours)
biggestContour = utlis.getCornerPoints(rectCon[0])
gradePoints = utlis.getCornerPoints(rectCon[1])
#print(biggestContour)

if biggestContour.size !=0 and gradePoints.size !=0:
    cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),20)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)

    biggestContour = utlis.reorder(biggestContour)
    gradePoints = utlis.reorder(gradePoints)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg], [widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColored= cv2.warpPerspective(img,matrix,(widthImg,heightImg))

    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
    #cv2.imshow("Grade", imgGradeDisplay)

    #APPLY THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray,170,255,cv2.THRESH_BINARY_INV)[1]


    boxes = utlis.splitBoxes(imgThresh)  # GET INDIVIDUAL BOXES

    myPixelVal = np.zeros((questions, choices))
    countR = 0
    countC = 0

    for image in boxes:
        # cv2.imshow(str(countR)+str(countC),image)
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if (countC == choices): countC = 0;countR += 1
   # print(myPixelVal)

    # tim cac o duoc danh tren to dap an
    myIndex = []
    for x in range (0,questions) :
        arr = myPixelVal[x]
        #print("arr",arr)
        myIndexVal = np.where(arr==np.amax(arr))
        #print(myIndexVal[0])
        myIndex.append((myIndexVal[0][0]))
    #print(myIndex)

    # CHAM DIEM
    grading = []
    for x in range (0,questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else: grading.append(0)
    #print(grading)
    score = sum(grading)/questions*100 #tim diem so
    print(score)

    #DISPLAY ANSWERS
    imgResult = imgWarpColored.copy()
    imgResult = utlis.showAnswers(imgResult,myIndex,grading,ans,questions,choices)
    imgRawDrawing = np.zeros_like(imgWarpColored)
    imgRawDrawing = utlis.showAnswers(imgRawDrawing,myIndex,grading,ans,questions,choices)
    invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
    imgInvWarp = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))

    imgRawGrade = np.zeros_like(imgGradeDisplay)
    cv2.putText(imgRawGrade,str(int(score))+"%",(60,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
    invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))


    imgFinal = cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)


imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur,imgCanny],
              [imgContours, imgBiggestContours,imgWarpColored,imgThresh],
              [imgResult, imgRawDrawing,imgInvWarp,imgFinal])
lables = [["Original","Gray", "BLur", "Canny"],
          ["Contour","Biggest Con", "Warp", "Thresthold"],
          ["Result", "Raw Drawing","Inv warp", "Final"]]
imgStacked = utlis.stackImages(imageArray,0.3,lables)

cv2.imshow("Final Result",imgFinal)
cv2.imshow("Stacked Images",imgStacked)
cv2.waitKey(0)
