import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    # read image
    ret, img= cap.read()

    # get hand data from the rectangle sub window on the screen
    cv2.rectangle(img, (270,270), (50,50), (0,255,0),0)
    crop_img = img[50:270, 50:270]

    # convert to greyscale
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # applying gaussian blur
    value = (31, 31)
    blurred = cv2.GaussianBlur(grey, value, 0)

    # thresholdin: Otsu's Binarization method
    _, thresh1 = cv2.threshold(blurred, 100, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # show thresholded image
    cv2.imshow('Thresholded', thresh1)

    # check OpenCV version to avoid unpacking error
    image, contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # find contour with max area
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    
    # create bounding rectangle around the contour (can skip below two lines)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt)
    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(cnt)

    print("areacnt",areacnt)
    # find the percentage of area not covered by hand in convex hull
    arearatio = ((areahull - areacnt) / areacnt) * 100
    print("arearatio", arearatio)
    # drawing contours
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)

    # finding convexity defects
    defects = cv2.convexityDefects(cnt, hull)

    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    # applying Cosine Rule to find angle for all defects (between fingers)
    # with angle > 90 degrees and ignore defects
    l= 0
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        pt = (100,100)

        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        s = (a + b + c) / 2
        ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
        # Distance between points and convex hull
        d = (2 * ar) / a

        # apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignore angles > 90 and highlight rest with blue dots
        if angle <= 90 and d > 20:
            l += 1
            cv2.circle(crop_img, far, 3, [255, 0, 0], -1)

            # draw lines around hand
        cv2.line(crop_img, start, end, [0, 255, 0], 2)

        #dist = cv2.pointPolygonTest(cnt,far,True)

        # draw a line from start to end i.e. the convex points (finger tips)
        # (can skip this part)
        #cv2.circle(crop_img,far,5,[0,0,255],-1)
    l+=1

    # define actions required
    font = cv2.FONT_HERSHEY_SIMPLEX
    if l == 1:
        if areacnt > 30000:
            cv2.putText(img, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        else:
            if arearatio < 7:
                cv2.putText(img, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            elif arearatio < 15 :
                cv2.putText(img, 'Best of luck', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

            else:
                cv2.putText(img, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    elif l == 2:
        cv2.putText(img, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    elif l == 3:

        if arearatio < 30:
            cv2.putText(img, 'ok', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(img, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    elif l == 4:
        cv2.putText(img, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    elif l == 5:
        cv2.putText(img, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    elif l == 6:
        cv2.putText(img, 'reposition', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    else:
        cv2.putText(img, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    # show appropriate images in windows
    def my_mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:  # here event is left mouse button double-clicked
            print(x,y)
    cv2.setMouseCallback('Gesture', my_mouse_callback, img)
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()