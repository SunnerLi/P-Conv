import numpy as np
import argparse
import time
import cv2
import os

"""
    This script will generate the binary mask and store in the target folder
    You should notice: The folder will not be cleaned before generating
"""

def parse():
    """
        Parse the argument, create the folder and check if the argument is valid
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', default = 0.1, type = float, help = "The ratio of the mask")
    parser.add_argument('--size', default = '256x256', type = str, help = "The size of mask you want to generate")
    parser.add_argument('--margin', default = False, type = bool, help = "If the mask touch the margin region")
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

def determinePosition():
    # --------------------------------
    # 決定筆劃粗系(2~10)
    # while True:
    #     決定初始地點
    #     if (margin 且 地點（往外範圍）距離邊界<50) or (not margin 且 地點（往外範圍）距離邊界>50):
    #         break
    # --------------------------------
    pass

def draw(args):
    height, width = args.size.split('x')
    height, width = int(height), int(width)
    origin_mask = np.ones([height, width, 1])

    # 初始化
    # 呼叫determinePosition獲取初始座標

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
        pass

        # End if the mask ratio is enough
        if 1 - (np.sum(origin_mask) / (height * width)) > args.ratio:
            break
        exit()
    return origin_mask

if __name__ == '__main__':
    args = parse()
    for i in range(args.num):
        mask = draw(args)
        cv2.imshow('show', mask)
        # cv2.waitKey()
        # cv2.imwrite(str(time.time()).replace('.', ''), mask)