import os
import random

# 你的数据目录
img_dir = '/opt/data/private/CUTGAN/datasets/mix_dataset/CX41_olderimage_crop/trainA'
# 支持的图片后缀
img_exts = ['.jpg', '.png', '.jpeg', '.tif', '.tiff', '.bmp']

# 划分比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 获取所有图片路径
all_imgs = []
for fname in os.listdir(img_dir):
    ext = os.path.splitext(fname)[-1].lower()
    if ext in img_exts:
        all_imgs.append(os.path.join(img_dir, fname))

print(f'共发现图片数: {len(all_imgs)}')

# 随机打乱
random.shuffle(all_imgs)

# 按比例切分
n_total = len(all_imgs)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)
n_test = n_total - n_train - n_val

train_imgs = all_imgs[:n_train]
val_imgs = all_imgs[n_train:n_train+n_val]
test_imgs = all_imgs[n_train+n_val:]

# 保存为txt
def save_list(filelist, filepath):
    with open(filepath, 'w') as f:
        for fp in filelist:
            f.write(fp + '\n')

save_list(train_imgs, '/opt/data/private/ResShift/training_dataset/train.txt')
save_list(val_imgs, '/opt/data/private/ResShift/training_dataset/val.txt')
save_list(test_imgs, '/opt/data/private/ResShift/training_dataset/test.txt')

print(f'已保存: train.txt ({len(train_imgs)}), val.txt ({len(val_imgs)}), test.txt ({len(test_imgs)})')
