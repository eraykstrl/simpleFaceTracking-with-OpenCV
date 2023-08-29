import cv2
import numpy as np

cap=cv2.VideoCapture(0)

#reading frame

ret,frame=cap.read()
if ret == False:
    print("UYARI VERİYORUM ALOO")

#detection
face_cascade=cv2.CascadeClassifier('C:\\Users\\koste\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
faceRects=face_cascade.detectMultiScale(frame)

(face_x,face_y,w,h)=tuple(faceRects[0])
track_window=(face_x,face_y,w,h)
## region of interest
roi=frame[face_y:face_y+h , face_x:face_x + w]

hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

roi_hist=cv2.calcHist([hsv_roi],[0],None,[180],[0,180]) ## takip için histogram gerekli
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

#TAKİP İÇİN GEREKLİ DURDRUMA KRİTERLERİ
## count = maximum number of items to calculate
# eps=changing

term_crit=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,5,1)

while True:
    ret,frame=cap.read()

    if ret:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        #to use the histogram in an image
        dst=cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        ret,track_window=cv2.meanShift(dst,track_window,term_crit)
        x,y,width,height=track_window

        img2=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        cv2.imshow("Following",img2)

        if cv2.waitKey(1) & 0xFF==ord("q"):break

cap.release()
cv2.destroyAllWindows()