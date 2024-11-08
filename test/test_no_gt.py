import sys
sys.path.append('../')
import torch
import argparse
import torch.nn as nn
import os
import numpy as np
from model.cmformer import CMFormer
import torchvision.utils as tvu
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import blobfile as bf

parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-save_place', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-checkpoint', help='select checkpoint of model', type=str)
parser.add_argument('-val_data_dir', help='test dataset path', type=str)
args = parser.parse_args()

def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)

def list_image_files_recursively(data_dir):
    results = []
    #把数据集目录下的文件和子目录的名称按照字母顺序排序
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        #得到文件的扩展名
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            # 递归地调用这个函数，把子目录下的所有图片文件的路径添加到结果列表中
            results.extend(list_image_files_recursively(full_path))
    return results

def infer(save_place, chk, val_data):
    # --- Gpu device --- #
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # --- Define the network --- #
    net = CMFormer()
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=device_ids)
    try:
        net.load_state_dict(torch.load(chk))
    except:
        raise FileNotFoundError(f"The file at path '{chk}' does not exist.")
    net.eval()
    if os.path.exists(save_place) == False:
        os.makedirs(save_place)
    with torch.no_grad():
        if os.path.isdir(val_data):
            all_images = list_image_files_recursively(val_data)
            for path in all_images:
                input_img = Image.open(path).convert('RGB')
                # Resizing image in the multiple of 8"
                wd_new, ht_new = input_img.size
                if ht_new > wd_new and ht_new > 1024:
                    wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                    ht_new = 1024
                elif ht_new <= wd_new and wd_new > 1024:
                    ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                    wd_new = 1024
                wd_new = int(8 * np.ceil(wd_new / 8.0))
                ht_new = int(8 * np.ceil(ht_new / 8.0))
                input_img = input_img.resize((wd_new, ht_new), Image.LANCZOS)
                input_img = transform_input(input_img)
                input_img = input_img.to(device)
                input_img = input_img.unsqueeze(0)
                #print(f"input_img={input_img.shape}")
                result_img = net(input_img)
                save_image(result_img, os.path.join(save_place, '{}'.format(os.path.basename(path))))
        else:
            path = val_data
            input_img = Image.open(path).convert('RGB')
            # Resizing image in the multiple of 8"
            wd_new, ht_new = input_img.size
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(8 * np.ceil(wd_new / 8.0))
            ht_new = int(8 * np.ceil(ht_new / 8.0))
            input_img = input_img.resize((wd_new, ht_new), Image.LANCZOS)
            input_img = transform_input(input_img)
            input_img = input_img.to(device)
            input_img = input_img.unsqueeze(0)
            result_img = net(input_img)
            save_image(result_img, os.path.join(save_place, '{}'.format(os.path.basename(path))))

if __name__ == "__main__":
    infer(args.save_place, args.checkpoint, args.val_data_dir)