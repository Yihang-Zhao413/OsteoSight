import os
import random
import shutil

img_dir = '/opt/data/private/CUTGAN/datasets/mix_dataset/CX41_olderimage_crop/trainA'
img_exts = ['.jpg', '.png', '.jpeg', '.tif', '.tiff', '.bmp']
out_root = '/opt/data/private/ResShift/training_dataset/CX41_trainA_split'  # 新的数据集输出目录

train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

all_imgs = [f for f in os.listdir(img_dir) if os.path.splitext(f)[-1].lower() in img_exts]
random.shuffle(all_imgs)
n_total = len(all_imgs)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)

split_map = [
    (all_imgs[:n_train], 'train'),
    (all_imgs[n_train:n_train+n_val], 'val'),
    (all_imgs[n_train+n_val:], 'test'),
]

for filelist, split in split_map:
    split_dir = os.path.join(out_root, split)
    os.makedirs(split_dir, exist_ok=True)
    for fname in filelist:
        shutil.copy(os.path.join(img_dir, fname), os.path.join(split_dir, fname))
    print(f'{split} 集合已保存 {len(filelist)} 张图片到 {split_dir}')
