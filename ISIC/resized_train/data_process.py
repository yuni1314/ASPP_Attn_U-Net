import numpy as np
from PIL import Image
import os
import tqdm

def resize(img_path,filename,flag=True):
    save_path = ''
    if flag == True:#保存图片
        save_path = '/home/sxd/guopengcheng/ISIC2017-2018/focal_tversky_unet/resized_train/resized_train'
    else:#保存masks
        save_path = '/home/sxd/guopengcheng/ISIC2017-2018/focal_tversky_unet/resized_train/resized_gt'
    with Image.open(img_path) as img:
        img = img.resize((192, 256), Image.ANTIALIAS)
        img.save(os.path.join(save_path, filename))
        if not flag:
            img = img.convert("1")#将图片二值化
        img.save(os.path.join(save_path, filename))

if __name__ == "__main__":
    img_path = "/home/sxd/guopengcheng/ISIC2017-2018/focal_tversky_unet/resized_train/orig_gt"
    filelist = os.listdir(img_path)
    for filename in filelist:
        path = os.path.join(img_path, filename)
        print(path)
        resize(path, filename, False)

