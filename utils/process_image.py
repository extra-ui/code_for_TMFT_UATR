import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm

def extract_and_resize_images(directory, PSNR):
    resized_images = []
    # 遍历目录中的所有文件
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # 打开图像文件
            with Image.open(file_path) as img:
                img = img.convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img)
                img_array = np.transpose(img_array, (2, 0, 1))
                if PSNR is not None:
                    img_array = add_noise(img_array, PSNR)
                resized_images.append(img_array)
    return resized_images


def show_image(image):
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def add_noise(img, target_psnr):
    img = np.transpose(img, (1, 2, 0)).astype(np.float32)
    max_pixel = 255.0
    current_mse = np.mean((img - np.mean(img)) ** 2)
    desired_mse = (max_pixel ** 2) / (10 ** (target_psnr / 10))
    noise_variance = desired_mse - current_mse
    if noise_variance < 0:
        noise_variance = 0

    noise = np.random.normal(0, np.sqrt(noise_variance), img.shape)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, max_pixel).astype(np.uint8)

    noisy_img = np.transpose(noisy_img, (2, 0, 1))
    return noisy_img

if __name__ == '__main__':
    pass



