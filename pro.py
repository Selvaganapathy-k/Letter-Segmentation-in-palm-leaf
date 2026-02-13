# Importing required liberies
import cv2

# Reading the image as img

img = cv2.imread("scan0001_1bin.png")

# pip install opencv-python==4.5.3


# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

# Converting to binary using AdaptiveThreshMean
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 21, 11)
cv2.imshow("Thresh", thresh)

# Converting to binary using Threshold
# _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

# Converting to inverse image for finding the contours
thresh = ~thresh

# Find the contours in the image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Converting back to normal image for our convenience
thresh = ~thresh

# Displaying the image
cv2.imshow("test1", thresh)

# Making the copy of the thresh for future use
thresh2 = thresh.copy()

# Define a minimum area for noise
min_noise = 30

# Define a minimum area threshold
min_area = 50

# Define a maximum area threshold
max_area = 50000

# coping the img for future use
img_copy = img.copy()

# array for boundingbox and coordinates
bounding_boxes = []
x1 = []
y1 = []
x2 = []
y2 = []

# --------------------------------------------Loop for detecting the characters-----------------------------------------

# Loop over the contours
for contour in contours:

    # Calculate the area of the contour
    area = cv2.contourArea(contour)

    # If the area is less than the minimum area threshold, draw a white rectangle over it to erase it

    # if area < min_noise:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(thresh2, (x, y), (x + w, y + h), (255, 255, 255), -1)

    # For printing the seperate characters inside another character
    if area < min_area:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), -1)

    # Detecting the characters
    if area > min_area and area < max_area:
        # If the area is greater than or equal to the minimum area threshold, draw a green bounding box around it
        x, y, w, h = cv2.boundingRect(contour)

        # Drawing the bounding box
        cv2.rectangle(thresh2, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Creating the bounding box array
        bounding_boxes.append((x, y, w, h))

        # creating the coordinates array
        x1.append(x)
        x2.append(w)
        y1.append(y)
        y2.append(h)

# -----------------------------------loop for croping the boxes----------------------------------

# Defining a increamenting value for loop
loop_inc = 0

cv2.imshow("Noise removed", thresh)
cv2.imshow("character removed", thresh2)

# Defining the loop
while loop_inc < len(x1):

    # Croping the detected image
    im = thresh2[y1[loop_inc]:y1[loop_inc] + y2[loop_inc], x1[loop_inc]:x1[loop_inc] + x2[loop_inc]]

    # Changing the output window size if required
    # im_big = cv2.resize(im, (1200, 600))

    # Displaying each segmented image if required
    cv2.imshow(f"image{loop_inc}",im)

    # Writing each segmented image if required
    cv2.imwrite(f"./Seg/1-{loop_inc}.png", im)

    # Preprocessing the resulten image for increasing the quality
    erosion = cv2.erode(im, None, iterations=4)
    dilate = cv2.dilate(erosion, None, iterations=2)

    # Increamenting the variable
    loop_inc += 1

# Displaying the image and writing it in desired location
cv2.imshow("Final_Image", thresh2)
cv2.imwrite("../Seg/Final_Image2.png", thresh2)

# Used to freeze the image displayed
cv2.waitKey(0)

# To destroy previously opened window
cv2.destroyAllWindows()
