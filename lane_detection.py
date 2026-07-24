import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')
left_top_x_prev = 0
left_bottom_x_prev = 0
right_top_x_prev = 0
right_bottom_x_prev = 0
while True:

    ret, frame = cam.read()

    if ret is False:
        break
#TASK 2
    height = frame.shape[0]
    width = frame.shape[1]

    new_height = int(height/2.5)
    new_width = int(width/2.5)

    frame = cv2.resize(frame, (new_width, new_height))
# TASK 3
    gray_frame = np.zeros((new_height, new_width), dtype=np.uint8)
    for i in range (new_height):
        for j in range (new_width):
            pixel_tuple = frame[i,j]
            b = pixel_tuple[0]
            g = pixel_tuple[1]
            r = pixel_tuple[2]

            gray_value = int(0.144*b + 0.587*g + 0.299*r)

            gray_frame[i,j] = gray_value
#TASK 4
    upper_right = (int(new_width*0.52), int(new_height*0.76))
    upper_left = (int(new_width*0.46), int(new_height*0.76))
    lower_left = (0, new_height-1)
    lower_right = (new_width-1, new_height-1)

    trapezoid_points = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)

    trapezoid_frame = np.zeros((new_height, new_width), dtype=np.uint8)
    cv2.fillConvexPoly(trapezoid_frame, trapezoid_points, 1)
    road_frame = gray_frame * trapezoid_frame
#TASK 5
    screen_points = np.array([(new_width - 1, 0), (0, 0), (0, new_height - 1), (new_width - 1, new_height - 1)], dtype=np.float32)
    trapezoid_points_float = np.float32(trapezoid_points)
    screen_points_float = np.float32(screen_points)
    perspective = cv2.getPerspectiveTransform(trapezoid_points_float, screen_points)
    top_down_view = cv2.warpPerspective(road_frame, perspective, (new_width, new_height))
#TASK 6
    blurred_frame = cv2.blur(top_down_view, (5, 5))
#TASK 7
    sobel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]], dtype=np.float32)
    sobel_orizontal = np.transpose(sobel_vertical)

    blurred_float = np.float32(blurred_frame)

    horizontal_edges = cv2.filter2D(blurred_float, -1, sobel_orizontal)
    vertical_edges = cv2.filter2D(blurred_float, -1, sobel_vertical)

    sobel_combined = np.sqrt(horizontal_edges**2 + vertical_edges**2)

    sobel_frame = cv2.convertScaleAbs(sobel_combined)
#TASK 8
    threshold = int(150/2)

    binary_frame = sobel_frame.copy()

    binary_frame[binary_frame < threshold] = 0
    binary_frame[binary_frame >= threshold] = 255
#TASK 9
    clean_frame = binary_frame.copy()

    height, width = clean_frame.shape
    edge_width = int(width*0.15)

    clean_frame[:,:edge_width] = 0
    clean_frame[:,-edge_width:] = 0

    middle = width//2

    left_side = clean_frame[:,:middle]
    right_side = clean_frame[:,middle:]

    left_points = np.argwhere(left_side==255)
    right_points = np.argwhere(right_side==255)

    left_ys = left_points[:,0]
    left_xs = left_points[:,1]

    right_ys = right_points[:,0]
    right_xs = right_points[:,1] +middle
#TASK 10
    left_b, left_a = np.polynomial.polynomial.polyfit(left_xs, left_ys,deg=1)
    right_b, right_a = np.polynomial.polynomial.polyfit(right_xs, right_ys,deg=1)

    left_top_y = 0
    left_bottom_y = height
    right_top_y = 0
    right_bottom_y = height

    left_top_x = int((left_top_y - left_b) / left_a)
    left_bottom_x = int((left_bottom_y - left_b) / left_a)
    right_top_x = int((right_top_y - right_b) / right_a)
    right_bottom_x = int((right_bottom_y - right_b) / right_a)

    if -10**8 <=left_top_x <= 10**8:
        left_top_x_prev = left_top_x
        left_bottom_x_prev = left_bottom_x
    else:
        left_top_x = left_top_x_prev
        left_bottom_x = left_bottom_x_prev

    if -10**8 <=right_top_x <= 10**8:
        right_top_x_prev = right_top_x
        right_bottom_x_prev = right_bottom_x
    else:
        right_top_x = right_top_x_prev
        right_bottom_x = right_bottom_x_prev

    cv2.line(binary_frame, (left_top_x, left_top_y), (left_bottom_x, left_bottom_y), (200, 0, 0), 5)
    cv2.line(binary_frame, (right_top_x, right_top_y), (right_bottom_x, right_bottom_y), (100, 0, 0), 5)

#TASK 11
    blank_frame_left = np.zeros((height, width), dtype=np.uint8)
    blank_frame_right = np.zeros((height, width), dtype=np.uint8)
    cv2.line(blank_frame_left, (left_top_x, left_top_y), (left_bottom_x, left_bottom_y), (255, 0, 0), 3)
    cv2.line(blank_frame_right, (right_top_x, right_top_y), (right_bottom_x, right_bottom_y), (255, 0, 0), 3)
    magic_matrix_inverse = cv2.getPerspectiveTransform(screen_points_float, trapezoid_points_float)
    warp_left = cv2.warpPerspective(blank_frame_left, magic_matrix_inverse, (width, height))
    warp_right =cv2.warpPerspective(blank_frame_right, magic_matrix_inverse, (width, height))
    left_warped_coords = np.argwhere(warp_left > 0)
    left_final_y = left_warped_coords[:, 0]
    left_final_x = left_warped_coords[:, 1]
    right_warped_coords = np.argwhere(warp_right > 0)
    right_final_y = right_warped_coords[:, 0]
    right_final_x = right_warped_coords[:, 1]
    final_frame = frame.copy()
    final_frame[left_final_y, left_final_x] = (0, 0, 255)
    final_frame[right_final_y, right_final_x] = (0, 255, 0)

    cv2.imshow('Forma Finala', final_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()