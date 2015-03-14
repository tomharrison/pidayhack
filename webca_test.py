import cv2
import numpy as np
import scipy
import math
from scipy import ndimage


def compare_pi(our_pi):
    global pi
    error = (abs(pi - our_pi) / pi) * 100
    return error

def calc_radius(C,A):
    global pi
    r1 = (C/pi)/2
    r2 = math.sqrt(A/pi)
    r3 = (r1+r2)/2
    return r3

#Calculate our version of Pi
def calc_pi(C,A):
    return float(C**2)/float(4*A)

def im_thresh(im):
    average = np.mean(im)
    bins,edges = np.histogram(im,range(256))
    # find peak below average
    maxB = 0
    maxE = 0
    #print bins,len(bins)
    #print edges,len(edges)
    for i in xrange(len(bins)):
        e = edges[i]
        b = bins[i]
        if e>=(average-32):
            continue
        if b>maxB:
            maxB = b
            maxE = e
    im2 = np.zeros(im.shape)
    im2[im<(maxE+24)] = 255
    #im2[im>(maxE-16)] = 255
    struct = ndimage.generate_binary_structure(2,50)
    ndimage.morphology.binary_closing(im2,struct,output=im2)
    return im2

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    gray = im_thresh(gray)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
