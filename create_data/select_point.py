"""
reference: https://gaussian37.github.io/vision-opencv-coordinate_extraction/

print("\n")
print("1. 입력한 파라미터인 이미지 경로(--path)에서 이미지들을 차례대로 읽어옵니다.")
print("2. 키보드에서 'n'을 누르면(next 약자) 다음 이미지로 넘어갑니다. 이 때, 작업한 점의 좌표가 저장 됩니다.")
print("3. 키보드에서 'b'를 누르면(back 약자) 직전에 입력한 좌표를 취소한다.")
print("4. 이미지 경로에 존재하는 모든 이미지에 작업을 마친 경우 또는 'q'를 누르면(quit 약자) 프로그램이 종료됩니다.")
print("\n")
print("출력 포맷 : 이미지명,점의갯수,y1,x1,y2,x2,...")
print("\n")
"""

import os
import argparse
import cv2
from datetime import datetime

dir_del = None
clicked_points = []
clone = None

def MouseLeftClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((y, x))
        image = clone.copy()
        for point in clicked_points:
            cv2.circle(image, (point[1], point[0]), 8, (0, 255, 255), thickness = -1)
        cv2.imshow("image", image)

def main(args):
    global clone, clicked_points
    image_names = os.listdir(args.path)

    if len(args.path.split('\\')) > 1: dir_del = '\\'
    else :                        dir_del = '/'

    folder_name = args.path.split(dir_del)[-1]

    now = datetime.now()
    now_str = "%s%02d%02d_%02d%02d%02d" % (now.year - 2000, now.month, now.day, now.hour, now.minute, now.second)   

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", MouseLeftClick)

    for idx, image_name in enumerate(image_names):
        image_path = args.path + dir_del + image_name
        image = cv2.imread(image_path)

        clone = image.copy()

        flag = False

        while True:
            cv2.imshow("image", image)
            key = cv2.waitKey(0)

            if key == ord('n'):
                file_write = open('../data/annotation_text_files/' + now_str + '_' + folder_name + '.txt', 'a+')
                text_output = image_name
                text_output += "," + str(len(clicked_points))
                for points in clicked_points:
                    text_output += "," + str(points[0]) + "," + str(points[1])
                text_output += '\n'
                file_write.write(text_output)
                
                clicked_points = []
                file_write.close()

                break

            if key == ord('b'):
                if len(clicked_points) > 0:
                    clicked_points.pop()
                    image = clone.copy()
                    for point in clicked_points:
                        cv2.circle(image, (point[1], point[0]), 8, (0, 255, 255), thickness = -1)
                    cv2.imshow("image", image)

            if key == ord('q'):
                flag = True
                break
        if flag:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", required=True, help="Enter the image files path")
    
    args = parser.parse_args()
    main(args)
