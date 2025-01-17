#--------------------------------------------------------#
#   该文件用于调整输入彩色图片的后缀
#--------------------------------------------------------#
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

#--------------------------------------------------------#
#   Origin_JPEGImages_path   原始标签所在的路径
#   Out_JPEGImages_path      输出标签所在的路径
#--------------------------------------------------------#
Origin_SegmentationClass_path   = "VOCdevkit/VOC2007/SegmentationClass_Origin"
Out_SegmentationClass_path      = "VOCdevkit/VOC2007/SegmentationClassss"

if __name__ == "__main__":
    if not os.path.exists(Out_SegmentationClass_path):
        os.makedirs(Out_SegmentationClass_path)

    #---------------------------#
    #   遍历标签并赋值
    #---------------------------#
    image_names = os.listdir(Origin_SegmentationClass_path)
    print("正在遍历全部图片。")
    for image_name in tqdm(image_names):
        image   = Image.open(os.path.join(Origin_SegmentationClass_path, image_name))
        new_image_name = image_name.replace("Lable", "")
        image.save(os.path.join(Out_SegmentationClass_path, os.path.splitext(new_image_name)[0] + '.png'))


