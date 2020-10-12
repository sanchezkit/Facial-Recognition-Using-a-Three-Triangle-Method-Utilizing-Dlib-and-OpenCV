import dlib
import numpy as np
import cv2
import time


class alignments:
    def __init__(self, predictor, predictor_shape_np, image):
        self.predictor = predictor
        self.predictor_shape_np = predictor_shape_np
        self.image = image

    def align(self):
        point2 = self.predictor_shape_np[2]
        point3 = self.predictor_shape_np[3]
        point0 = self.predictor_shape_np[0]

        # Formula to get Angle between two points
        # dX = x2 - x1
        # dY = y2 - y1
        # to get the angle we calculate the arctan of this
        # arctan(dX, dY)
        dX = point3[0] - point2[0]
        dY = point3[1] - point2[1]
        # print(dX)
        # print(dY)
        angle = np.degrees(np.arctan2(dY, dX))
        # print("Angle: " + str(angle))

        # Euclidian Distance
        # We need to get the distance betwen two lines to properly scale our image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        # print("Dist: " + str(dist))
        dEye = 0.7 * 366
        scale = dEye / dist
        # print("Scale: " + str(scale))
        median_point = (point0[0], point0[1])
        # print(median_point)
        median = cv2.getRotationMatrix2D(median_point, angle, scale)

        # Translation matrix
        tX = 366 * 0.5
        tY = 366 * 0.25
        median[0, 2] += (tX - median_point[0])
        median[1, 2] += (tY - median_point[1])
        output = cv2.warpAffine(self.image, median, (366, 366), flags=cv2.INTER_CUBIC)

        return output
