import cv2 as cv
import os
from matplotlib import pyplot as plt

def main():
    img_list = [r'C:\Users\nathan\PycharmProjects\pythonProject\Multi_Modal_Image_Classifier\IMG_20211027_170109.jpg',
                r'C:\Users\nathan\PycharmProjects\pythonProject\Multi_Modal_Image_Classifier\IMG_20211027_170115.jpg',
                r'C:\Users\nathan\PycharmProjects\pythonProject\Multi_Modal_Image_Classifier\IMG_20211027_170124.jpg',
                r'C:\Users\nathan\PycharmProjects\pythonProject\Multi_Modal_Image_Classifier\IMG_20211027_170127.jpg',
                r'C:\Users\nathan\PycharmProjects\pythonProject\Multi_Modal_Image_Classifier\IMG_20211027_170134.jpg',
                r'C:\Users\nathan\PycharmProjects\pythonProject\Multi_Modal_Image_Classifier\IMG_20211027_170203.jpg',
                r'C:\Users\nathan\PycharmProjects\pythonProject\Multi_Modal_Image_Classifier\IMG_20211027_170211.jpg',
                r'C:\Users\nathan\PycharmProjects\pythonProject\Multi_Modal_Image_Classifier\IMG_20211027_170237.jpg',
                r'C:\Users\nathan\PycharmProjects\pythonProject\Multi_Modal_Image_Classifier\IMG_20211027_170236.jpg',
                r'C:\Users\nathan\PycharmProjects\pythonProject\Multi_Modal_Image_Classifier\IMG_20211027_170303.jpg',
                r'C:\Users\nathan\PycharmProjects\pythonProject\Multi_Modal_Image_Classifier\IMG_20211027_170156.jpg']
    comparator_img = cv.imread(img_list[10])
    test_img = cv.imread(img_list[9])

    sift = cv.SIFT_create()

    '''
    the index paramters sets up using the various algoritims such 
    KDtree or kmeans as a method of finding out which is similar
    '''
    index_params = dict(algorithm=1, trees=5)
    query_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, query_params)


    kp1, des1 = sift.detectAndCompute(comparator_img, None)
    kp2, des2 = sift.detectAndCompute(test_img, None)

    matches = flann.knnMatch(des1, des2, k = 2)

    matchesMask = [[0, 0] for i in range(len(matches))]

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            matchesMask[i] = [1, 0]
   
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    img3 = cv.drawMatchesKnn(comparator_img, kp1, test_img, kp2, matches, None, **draw_params)
    plt.title('.8 threshold')
    plt.imshow(img3)
    plt.show()


if __name__ == '__main__':
    main()