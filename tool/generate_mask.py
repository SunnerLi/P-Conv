from time import gmtime, strftime
from util import now
import numpy as np
import argparse
import random
import time
import cv2
import os

"""
    This script will generate the binary mask and store in the target folder
    You should notice: The folder will not be cleaned before generating
"""

# ---------------------------------------------------------------------------------------------------------
#                   Define the fundemental function of wondering
# ---------------------------------------------------------------------------------------------------------
def isOutside(y, x, height, width, thickness):
    # Check if the result position touch the negative region (or too large)
    if y - thickness < 0 or x - thickness < 0:
        return True
    if y + thickness > height or x + thickness > width:
        return True
    return False

def isAtMargin(y, x, height, width, thickness, _range = 25):
    # Check if the result position touch the outside region
    if (y - thickness) < _range:
        return True
    if (y + thickness) > height - _range:
        return True
    if (x - thickness) < _range:
        return True
    if (x + thickness) > width - _range:
        return True
    return False

def determineThickness(min_thick = 2, max_thick = 8):
    return random.randint(min_thick, max_thick)    

def determinePosition(height, width, thickness, margin = False):
    # --------------------------------
    # 決定筆劃粗系(2~10)
    # while True:
    #     決定初始地點
    #     if (margin 且 地點（往外範圍）距離邊界<50) or (not margin 且 地點（往外範圍）距離邊界>50):
    #         break
    # --------------------------------
    while True:
        y_det, x_det = random.randint(0, height), random.randint(0, width)
        if isOutside(y_det, x_det, height, width, thickness):
            continue
        at_margin = isAtMargin(y_det, x_det, height, width, thickness)
        if (margin and at_margin) or (not margin and not at_margin):
            break
    return y_det, x_det

# ---------------------------------------------------------------------------------------------------------

def parse():
    """
        Parse the argument, create the folder and check if the argument is valid
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio_min', default = 0.01, type = float, help = "The minumun ratio of the mask")
    parser.add_argument('--ratio_max', default = 0.1, type = float, help = "The maximun ratio of the mask")
    parser.add_argument('--size', default = '256x256', type = str, help = "The size of mask you want to generate")
    parser.add_argument('--margin', default = False, type = bool, help = "If the mask touch the margin region")
    parser.add_argument('--verbose', default = False, type = bool, help = "If show the detail of generating")
    parser.add_argument('--num', default = 1, type = int, help = "The number of mask you want to generate")
    parser.add_argument('--path', default = './gen_mask', type = str, help = "The folder you want to store into")
    args = parser.parse_args()

    # Check if the folder is exist
    if not os.path.exists(args.path):
        os.mkdir(args.path)

    # Check if the size is correct
    try:
        height, width = args.size.split('x')
    except Exception:
        print("Invalid format of size argument, you should type just like 256x256...")
        exit()
    return args

def __draw(img, y, x, thickness):
    """
        Manhaton distance or uclidian distance
    """
    height, width, _ = np.shape(img)
    distance_type = random.random()
    if distance_type < 0.3:
        for i in range(height):
            for j in range(width):
                if ((y - i) ** 2 + (x - j) ** 2) <= (thickness + 5) ** 2:
                    img[i][j] = 0                    
    elif distance_type < 0.6:
        for i in range(height):
            for j in range(width):
                if abs(y - i) + abs(x - j) <= thickness + 5:
                    img[i][j] = 0   
    else:
        for i in range(height):
            for j in range(width):
                if abs((y - i) ** 3) + abs((x - j) ** 3) <= (thickness + 5) ** 3:
                    img[i][j] = 0   
    return img

def draw(args, ratio = 0.1, verbose = False):
    # Define direction constant
    # -----------------------------------------------------------------
    # The direction order: [N, NE, E, SE, S, SW, W, NW]
    #           7 0 1
    #           6   2
    #           5 4 3
    # -----------------------------------------------------------------
    Y_DIR = [-1, -1, 0, 1, 1, 1, 0, -1]
    X_DIR = [0, 1, 1, 1, 0, -1, -1, -1]

    # Start
    height, width = args.size.split('x')
    height, width = int(height), int(width)
    origin_mask = np.ones([height, width, 1])

    # 初始化
    # 呼叫determinePosition獲取初始座標
    thickness = determineThickness()
    y, x = determinePosition(height, width, thickness, margin = args.margin)

    # Keep drawing
    mask = origin_mask
    while True:
        # Draw
        # --------------------------------    
        # while True:
        #     決定一個方向
        #     根據現在位置、比畫粗系和方向，算出結果座標
        #     if margin or 結果座標（往外範圍）距離邊界>50:
        #         break
        # 把該座標附近地區塗黑
        # if 機率落在要更改位置(0.05):
        #     呼叫determinePosition重新獲取座標
        #     改變比畫粗系(2~10)
        # else:
        #     現在座標 <- 結果座標
        # if 機率落在改變比畫粗系(0.1)：
        #     改變比畫粗系(2~10)
        while True:
            dir_idx = random.randint(0, 7)
            y_, x_ = y + 2 * Y_DIR[dir_idx] * thickness, x + 2 * X_DIR[dir_idx] * thickness
            if args.margin or not isAtMargin(y_, x_, height, width, thickness):
                break
        mask = __draw(mask, y_, x_, thickness)
        if random.random() < 0.1:
            thickness = determineThickness()
            y, x = determinePosition(height, width, thickness, margin = args.margin)
        else:
            y, x = y_, x_
        if random.random() < 0.1:    
            thickness = determineThickness()

        # Show information
        if verbose:
            print("thick: ", thickness, "\tx: ", x, "\ty: ", y, "\tx_: ", x_, "\ty_: ", y_, "\tdir: ", dir_idx, "\tratio: ", 1 - (np.sum(mask) / (height * width)))

        # End if the mask ratio is enough
        if 1 - (np.sum(mask) / (height * width)) > ratio:
            break
    return mask

if __name__ == '__main__':
    args = parse()
    for i in range(args.num):
        ratio_list = np.random.uniform(args.ratio_min, args.ratio_max, size = args.num)
        print("Ratio Range: [%f ~ %f] \t Margin: %d \t Progress %d / %d \t Mask ratio: %.4f" % \
            (args.ratio_min, args.ratio_max, args.margin, i+1, args.num, ratio_list[i]), end = '')
        print("\tTime: ", now())
        mask = draw(args, ratio_list[i], verbose = args.verbose)
        # cv2.imshow('show', mask)
        # cv2.waitKey()
        img_name = os.path.join(args.path, str(time.time()).replace('.', '') + '.png')
        mask = mask.astype(np.uint8) * 255
        cv2.imwrite(img_name, mask)