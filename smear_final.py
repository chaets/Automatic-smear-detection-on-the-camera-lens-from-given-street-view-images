import sys
#sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2
import os
from PIL import Image
import imutils
import sys
import glob
from matplotlib import pyplot as plt
import numpy as np
import time


def resize(cams):
    path = r"/Users/aacharya/Desktop/sample_drive/cam_"
    for i in range(len(cams) - 2):
        #note: since cam_4 doesn't exist we just skip to cam_5
        if (i == 4):
            i+=1
        for j in os.listdir(path+ str(i)):
            #DS STORE is a MAC problem
            if(not j.endswith(".DS_Store")):
                img = Image.open(path+ str(i) + "/" + j)
                x, y = os.path.splitext(path + str(i) +"/" + j )
                imgResize = img.resize((650,650), Image.ANTIALIAS)
                imgResize.save(x + '-resized.jpg', 'JPEG', quality=90)

def count_files(dir):
    # returns # files in directory
    return len([1 for x in list(os.scandir(dir)) if x.is_file()])

def avg(means, cams):
    path = r"/Users/aacharya/Desktop/sample_drive/cam_"
    path2 = r"/Users/aacharya/Desktop/sample_drive/"
    count = 0
    count_lst = []
    for i in range(len(cams) - 2):
        # skips cam_4 because doesn't exist
        if (i == 4):
            count_lst.append(str(i+1) + "- " + str(count_files(path + str(i + 1)) ))
        else:
            count_lst.append(str(i) + "- "+ str(count_files(path + str(i))))

    for i in range(len(cams) - 2):

        if(i == 4):
            i += 1
        for j in os.listdir(path+ str(i)):
            # use resized images to get averages
                if (not j.endswith(".DS_Store") and j.endswith('resized.jpg')):
                    image = cv2.imread(path + str(i) + "/" + j)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    arr = np.array(gray, dtype=np.float)
                    means = means + arr

        count = [k for k in count_lst if k.startswith(str(i) + "- ")]
        count = int(''.join(count)[3:]) / 2
        print(count)

        means = means / count
        means = np.array(np.round(means), dtype=np.uint8)

        Image.fromarray(means).save(path2 + "avgs/" +"cam" + str(i) + '-AVG.jpg','JPEG', quality = 100)


path = r"/Users/aacharya/Desktop/sample_drive/"
cams = os.listdir(r"/Users/aacharya/Desktop/sample_drive")

# making folders to put our intermediete results and outputs- ONLY RUN ONCE
os.mkdir(path + "outputs/")
os.mkdir(path + "avgs/")
print("made directories for our results")

cams.remove('.DS_Store')

#only need to run this one time, after that comment out otherwise ur code takes forever to run each time
#and causes repeat images

resize(cams)
print("done re-resizing images, now lets get average values")

means = np.zeros((650, 650), np.float)
avg(means, cams)
print("done getting averages, next up is thresholds")

avg_lst = os.listdir(path + "avgs")

# we add some random images in this list to be used for some visual purposes from each camera
random_img = []
random_img.append(path + "cam_0/"+ '393409999-resized.jpg')
random_img.append(path + "cam_1/"+ '393408634-resized.jpg')
random_img.append(path + "cam_2/"+ '393413141-resized.jpg')
random_img.append(path + "cam_3/"+ '393413080-resized.jpg')
random_img.append(path + "cam_5/"+ '393413095-resized.jpg')

x = 0
for i in range(len(avg_lst)):
    if i == 4:
        # skip 4th cam, make it 5 - made x var for some syntax purposes
        x = i + 1
    else:
        x = i
    # read in average pic and random image

    rand = cv2.imread(random_img[i])
    curr = cv2.imread(path + "avgs/" + avg_lst[i], 0)

    # apply adaptive threshold
    th3 = cv2.adaptiveThreshold(curr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 6)
    blur_image = cv2.GaussianBlur(th3, (5, 9), 0)
    warped = blur_image.astype("uint8") * 255

    # Detecting the edges in the image - using dilation to remove noise and horizantal lines from avg images
    kernel= np.ones((15, 17), np.uint8)

    edge_detected_image = 255 * np.ones(warped.shape,dtype=warped.dtype) - warped
    edge_detected_image = cv2.dilate(edge_detected_image, kernel, iterations=1)
    edge_detected_image = cv2.Canny(edge_detected_image, 9, 40, apertureSize=7, L2gradient=True)
    edge_detected_image = cv2.dilate(edge_detected_image, kernel, iterations=1)
    plt.imshow(edge_detected_image, cmap='gray')
    plt.axis('off')
    plt.title('Edge Detection'), plt.xticks([]), plt.yticks([])
    plt.savefig(path + "outputs/"+"cam_"+str(x)+'_edge_detected_image.jpg', bbox_inches='tight', pad_inches=0)

    # Detecting contours
    contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    laplacian = cv2.Laplacian(curr, cv2.CV_64F)
    sobelx = cv2.Sobel(curr, cv2.CV_64F, 1, 0, ksize=5)

    plt.imshow(sobelx, cmap='gray')
    plt.axis('off')
    plt.title('Gradient image'), plt.xticks([]), plt.yticks([])
    plt.savefig(path+ "outputs/"+"cam_"+ str(x) +'_Gradient.jpg' , bbox_inches='tight', pad_inches=0)

    list = []

    for m in contours:
        list.append(m)
    mask = np.zeros((650, 650, 1), np.float)
    img = cv2.drawContours(rand, contours, -1, (0, 255, 255), 2)
    k = cv2.drawContours(mask, contours, -1, (255, 255, 255), 30)
    img3 = cv2.drawContours(curr, contours, -1, (0, 255, 255), 3)

    # Saving Smear file and masked image
    cv2.imwrite(path +"outputs/"+"cam_" + str(x)+'_SmearOnAverageImage.jpg', img3)
    cv2.imwrite(path+"outputs/"+"cam_" + str(x) +'_MaskedImage.jpg', mask)

    # Resultant Image with smear

    plt.subplot(2, 2, 1), plt.imshow(rand, cmap='gray')
    plt.title('Original image with smear detection'), plt.xticks([]), plt.yticks([])
    plt.axis('off')

    plt.savefig(path+"outputs/"+"cam_" + str(x) +'_FinalResult.jpg')
    if len(list)> 0:
        print("Smear Detected. Result in" + path+"cam_" + str(x)+"_FinalResult.jpg")
    else:
        print("Smear not detected  Result in" + path+"cam_" + str(x)+"_FinalResult.jpg")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("done processesing for cam " + str(x))