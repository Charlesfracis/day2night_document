import subprocess
import cv2
bad_pics_path = '0306_700_night.txt'
lines = [line.split('\n')[0] for line in open(bad_pics_path).readlines()]
for i in range(0, len(lines)):
    img_path = lines[i]
    img_name = lines[i].split('/')[-1]
    try:
        mat = cv2.imread(img_path)
        cv2.imwrite('./0306_700_night/{}'.format(img_name), mat)
    except:
        print(img_path, 'not exist')
