import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 随机缩放
def random_scale(image, label, scale_range=(0.8, 1.2)):
    
    scale = random.uniform(*scale_range)                                     # 随机生成缩放比例
    height, width = image.shape[:2]                                          # 获取原始图像尺寸
    new_size = (int(width * scale), int(height * scale))                     # 计算新的图像尺寸
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)      # 按新尺寸缩放图像
    label = cv2.resize(label, new_size, interpolation=cv2.INTER_NEAREST)     # 按新尺寸缩放标签图像
    return image, label

# 随机噪音
def random_noise(image, noise_type='salt_pepper', mean=0, var=0.1, amount=0.02):

    if noise_type == 'gaussian':
        # 高斯噪音
        row, col, ch = image.shape
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        image = np.clip(image + gauss, 0, 255).astype(np.uint8)
        
    elif noise_type == 'salt_pepper':
        # 椒盐噪音
        image = np.copy(image)
        num_salt = int(amount * image.size * 0.5)
        num_pepper = int(amount * image.size * 0.5)

        # 添加盐噪声
        salt_coords = [random.randint(0, i-1) for i in image.shape]
        image[salt_coords[0], salt_coords[1], :] = 255

        # 添加胡椒噪声
        pepper_coords = [random.randint(0, i-1) for i in image.shape]
        image[pepper_coords[0], pepper_coords[1], :] = 0
        
    else:
        raise ValueError("Unsupported noise type. Use 'gaussian' or 'salt_pepper'.")
    return image

# 随机亮度
def random_brightness(image, brightness_range=(0.7, 1.3)):
    enhancer = ImageEnhance.Brightness(Image.fromarray(image))           # 创建亮度增强对象
    brightness = random.uniform(*brightness_range)                       # 在指定范围内生成随机亮度因子
    image = enhancer.enhance(brightness)                                 # 调整图像亮度
    return np.array(image)                                               # 转换图像格式并返回

# 随机翻转
def random_flip(image, label):
    # 随机水平翻转
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)

    # 随机垂直翻转
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
        label = cv2.flip(label, 0)

    return image, label

# 随机裁剪
def random_crop(image, label, crop_size=(256, 256)):
    height, width = image.shape[:2]
    crop_h, crop_w = crop_size                          # 获取图像的高度和宽度

    if height > crop_h and width > crop_w:              # 获取图像的高度和宽度
        top = random.randint(0, height - crop_h)        #  随机生成裁剪的左上角坐标
        left = random.randint(0, width - crop_w)
        image = image[top: top + crop_h, left: left + crop_w]   # 从图像和标签中裁剪出指定大小的区域

        label = label[top: top + crop_h, left: left + crop_w]
    else:
        image = cv2.resize(image, crop_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, crop_size, interpolation=cv2.INTER_NEAREST)

    return image, label

# 随机旋转
def random_rotation(image, label, angle_range=(-30, 30)):
    angle = random.uniform(*angle_range)               # 在指定角度范围内随机生成旋转角度
    height, width = image.shape[:2]                    # 获取图像的高度和宽度
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)        # 计算旋转矩阵
    image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)    # 应用旋转变换到图像
    label = cv2.warpAffine(label, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)   # 应用旋转变换到标签
    return image, label                               # 返回旋转后的图像和标签

def augment_image_and_label(image_path, label_path, crop_size=(256, 256)):
    image = cv2.imread(image_path)
    label = cv2.imread(label_path, 0)  # 读取标签图像，灰度模式

    # 随机缩放
    image, label = random_scale(image, label)
    
    # 随机亮度
    image = random_brightness(image)

    # 随机翻转
    image, label = random_flip(image, label)

    # 随机旋转
    image, label = random_rotation(image, label)

    # 随机裁剪
    #image, label = random_crop(image, label, crop_size)

    # 随机噪音
    image = random_noise(image)

    return image, label


# 获取文件路径
image_path = "VOCdevkit/VOC2007/JPEGImages_1class"
label_path = "VOCdevkit/VOC2007/SegmentationClass_1class"

# 增强后输出保存的文件路径
image_enhance_path = 'VOCdevkit/VOC2007/JPEGImages'
label_enhance_path = 'VOCdevkit/VOC2007/SegmentationClass'

# 如果增强后的文件夹不存在，创建它们
os.makedirs(image_enhance_path,exist_ok=True)
os.makedirs(label_enhance_path,exist_ok=True)

# 获取文件夹中所有的文件
file_list_image = sorted(os.listdir(image_path))
print(file_list_image)
file_list_label = sorted(os.listdir(label_path))

enhanced_num = 4

# 遍历文件列表,确保文件列表长度相同，并一一对应
for i in tqdm(range(enhanced_num),desc = "Processing images"):
    for file_name_image,file_name_label in zip(file_list_image,file_list_label):
    # for file_name_image, file_name_label in zip(file_list_image, file_list_label):
        # 拼接文件路径
        file_path_image = os.path.join(image_path, file_name_image)
        file_path_label = os.path.join(label_path,file_name_label)

        augmented_image, augmented_label = augment_image_and_label(file_path_image, file_path_label)

        output_name_image = file_name_image[:-4]+'_enhanced_'+str(i)+'.jpg'
        output_name_label = file_name_label[:-4] + '_enhanced_' + str(i) + '.png'

        print(output_name_image)

        # 保存增强后的图像和标签
        cv2.imwrite(os.path.join(image_enhance_path, output_name_image), augmented_image)
        cv2.imwrite(os.path.join(label_enhance_path, output_name_label), augmented_label)

print("数据增强完成！")
