import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')

while True:

    ret, frame = cam.read()

    if ret is False:
        break
    # TASK 2
    height = frame.shape[0]
    width = frame.shape[1]

    new_height = int(height / 2.5)
    new_width = int(width / 2.5)

    frame = cv2.resize(frame, (new_width, new_height))
    # TASK 3
    gray_frame = np.zeros((new_height, new_width), dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            pixel_tuple = frame[i, j]
            b = pixel_tuple[0]
            g = pixel_tuple[1]
            r = pixel_tuple[2]

            gray_value = int(0.144 * b + 0.587 * g + 0.299 * r)

            gray_frame[i, j] = gray_value
    # TASK 4
    upper_right = (int(new_width * 0.52), int(new_height * 0.76))
    upper_left = (int(new_width * 0.46), int(new_height * 0.76))
    lower_left = (0, new_height - 1)
    lower_right = (new_width - 1, new_height - 1)

    trapezoid_points = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)

    trapezoid_frame = np.zeros((new_height, new_width), dtype=np.uint8)
    cv2.fillConvexPoly(trapezoid_frame, trapezoid_points, 1)
    road_frame = gray_frame * trapezoid_frame
    # TASK 5
    screen_points = np.array([(new_width - 1, 0), (0, 0), (0, new_height - 1), (new_width - 1, new_height - 1)],
                             dtype=np.float32)
    trapezoid_points_float = np.float32(trapezoid_points)
    perspective = cv2.getPerspectiveTransform(trapezoid_points_float, screen_points)
    top_down_view = cv2.warpPerspective(road_frame, perspective, (new_width, new_height))
    # TASK 6
    blurred_frame = cv2.blur(top_down_view, (5, 5))
    # TASK 7
    sobel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]], dtype=np.float32)
    sobel_orizontal = np.transpose(sobel_vertical)

    blurred_float = np.float32(blurred_frame)

    horizontal_edges = cv2.filter2D(blurred_float, -1, sobel_orizontal)
    vertical_edges = cv2.filter2D(blurred_float, -1, sobel_vertical)

    sobel_combined = np.sqrt(horizontal_edges ** 2 + vertical_edges ** 2)

    sobel_frame = cv2.convertScaleAbs(sobel_combined)
    # TASK 8
    threshold = int(90 / 2)

    binary_frame = sobel_frame.copy()

    binary_frame[binary_frame < threshold] = 0
    binary_frame[binary_frame >= threshold] = 255
    # TASK 9
    clean_frame = binary_frame.copy()

    height, width = clean_frame.shape
    edge_width = int(width * 0.05)

    clean_frame[:, :edge_width] = 0
    clean_frame[:, -edge_width:] = 0

    middle = width // 2

    left_side = clean_frame[:, :middle]
    right_side = clean_frame[:, middle:]

    left_points = np.argwhere(left_side == 255)
    right_points = np.argwhere(right_side == 255)

    left_ys = left_points[:, 0]
    left_xs = left_points[:, 1]

    right_ys = right_points[:, 0]
    right_xs = right_points[:, 1] + middle
    # TASK 10

    # cv2.imshow("Original resized", frame)
    # cv2.imshow("Grayscale manual", gray_frame)
    # cv2.imshow("Trapezoid", trapezoid_frame * 255)
    # cv2.imshow("Road only", road_frame)
    # cv2.imshow("Top down", top_down_view)
    # cv2.imshow("Blurred", blurred_frame)

    # cv2.imshow("Sobel horizontal", cv2.convertScaleAbs(horizontal_edges))
    # cv2.imshow("Sobel vertical", cv2.convertScaleAbs(vertical_edges))
    # cv2.imshow("Sobel combined", sobel_frame)
    cv2.imshow("Binarized", binary_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()