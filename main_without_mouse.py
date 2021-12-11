import cv2
import time
import numpy as np

cam = cv2.VideoCapture(1)

cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)

bounds = {
    "red": {"lower": (0, 150, 140), "upper": (15, 255, 255)},
    "blue": {"lower": (90, 160, 160), "upper": (110, 255, 255)},
    "green": {"lower": (51, 90, 120), "upper": (70, 255, 255)},
    "yellow": {"lower": (20, 80, 160), "upper": (35, 255, 255)}
}

prev_time = time.time()
curr_time = time.time()
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
radius = 1
d = 5.6 * 10 ** -2


def ball_detect(bound: str):
    mask = cv2.inRange(hsv, bounds[bound]['lower'], bounds[bound]['upper'])
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        (curr_x, curr_y), radius = cv2.minEnclosingCircle(c)
        if radius > 10:
            return (curr_x, curr_y), radius

    return None


while cam.isOpened():
    ret, image = cam.read()
    curr_time = time.time()
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    green_circle = ball_detect('green')
    yellow_circle = ball_detect('yellow')
    red_circle = ball_detect('red')
    blue_circle = ball_detect('blue')

    if green_circle is not None:
        cv2.circle(image, (int(green_circle[0][0]), int(green_circle[0][1])), int(green_circle[1]), (0, 255, 0), 3)
        cv2.circle(image, (int(green_circle[0][0]), int(green_circle[0][1])), 3, (0, 255, 0), 3)

    if yellow_circle is not None:
        cv2.circle(image, (int(yellow_circle[0][0]), int(yellow_circle[0][1])), int(yellow_circle[1]), (0, 0, 255), 3)
        cv2.circle(image, (int(yellow_circle[0][0]), int(yellow_circle[0][1])), 3, (0, 255, 255), 3)

    if red_circle is not None:
        cv2.circle(image, (int(red_circle[0][0]), int(red_circle[0][1])), int(red_circle[1]), (0, 0, 255), 3)
        cv2.circle(image, (int(red_circle[0][0]), int(red_circle[0][1])), 3, (0, 255, 255), 3)

    if blue_circle is not None:
        cv2.circle(image, (int(blue_circle[0][0]), int(blue_circle[0][1])), int(blue_circle[1]), (0, 0, 255), 3)
        cv2.circle(image, (int(blue_circle[0][0]), int(blue_circle[0][1])), 3, (0, 255, 255), 3)


    time_diff = curr_time - prev_time
    pxl_per_m = d / radius
    dist = ((prev_x - curr_x) ** 2 + (prev_y - curr_y) ** 2) ** 0.5
    speed = dist / time_diff * pxl_per_m

    cv2.putText(
        image,
        "Speed = {0:.5f}m/s".format(speed),
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Camera", image)
    # cv2.imshow("Mask", mask)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    prev_time = curr_time
    prev_x = curr_x
    prev_y = curr_y

cam.release()
cv2.destroyAllWindows()
