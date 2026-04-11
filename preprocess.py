# preprocessing.py
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf

# 尝试引入 ResShift 和 Sparse Deconv 的依赖

from SDRNet.sampler import ResShiftSampler
from SDRNet.utils.util_opts import str2bool
from SDRNet.basicsr.utils.download_util import load_file_from_url
from SDRNet.sparse_recon.sparse_deconv import sparse_deconv




class SDRNetPipeline:
    def __init__(self):
        self.resshift_sampler = None
        self.is_initialized = False

        # ---  参数配置  ---
        self.scale = 4
        self.chop_size = 64
        self.chop_stride = -1
        self.seed = 12345

        self.pixelsize = 538  # (nm)
        self.resolution = 1270  # (nm)
        self.background = 1
        self.tcontinuity = 0
        self.sparsity = 1
        self.fidelity = 400

        # 临时文件路径
        self.temp_in = "./temp_sdr_in.png"
        self.temp_out_dir = "./results/temp_sdr_out"

    def init_resshift(self):
        """懒加载初始化 ResShift"""
        if self.is_initialized: return

        print("Initializing ResShift Model...")
        ckpt_dir = Path('./weights')
        if not ckpt_dir.exists(): ckpt_dir.mkdir()

        # 加载配置
        configs = OmegaConf.load('./SDRNet/configs/realsr_swinunet_realesrgan256_journal.yaml')
        ckpt_path = './SDRNet/weights/resshift_realsrx4_s4_v3.pth'
        vqgan_path = './SDRNet/weights/autoencoder_vq_f4.pth'

        print("CWD =", os.getcwd())
        print("CFG =", configs)
        print("CKPT =", ckpt_path)
        print("VQ =", vqgan_path)

        configs.model.ckpt_path = str(ckpt_path)
        configs.diffusion.params.sf = self.scale
        configs.autoencoder.ckpt_path = str(vqgan_path)

        print("image_size =", configs.model.params.image_size)
        print("lq_size =", configs.model.params.lq_size)
        print("diffusion.sf =", configs.diffusion.params.sf)
        print("degradation.sf =", configs.degradation.sf)
        print("steps =", configs.diffusion.params.steps)

        ...
        # 设置路径
        configs.model.ckpt_path = ckpt_path
        configs.diffusion.params.sf = self.scale
        configs.autoencoder.ckpt_path = vqgan_path

        # 步长计算逻辑，对齐 inference
        scale_factor = 4 // self.scale
        real_chop_size = self.chop_size * scale_factor

        if self.chop_stride < 0:
            self.chop_stride = (self.chop_size - 16) * scale_factor
        else:
            self.chop_stride = self.chop_stride * scale_factor

        print(f"ResShift Config: Chop={real_chop_size}, Stride={self.chop_stride}, AMP=True")

        self.resshift_sampler = ResShiftSampler(
            configs, sf=self.scale, chop_size=real_chop_size, chop_stride=self.chop_stride,
            chop_bs=1, use_amp=True, seed=self.seed,
            padding_offset=configs.model.params.get('lq_size', 64)
        )
        self.is_initialized = True

    def run_resshift(self, img_np):
        """运行 ResShift"""
        self.init_resshift()
        if not os.path.exists(self.temp_out_dir): os.makedirs(self.temp_out_dir)

        # 保存临时文件供读取
        cv2.imwrite(self.temp_in, img_np)
        self.resshift_sampler.inference(self.temp_in, self.temp_out_dir, bs=1, noise_repeat=False)

        # 读取结果
        filename = os.path.basename(self.temp_in)
        # 尝试寻找结果文件
        res_path = os.path.join(self.temp_out_dir, filename)
        if not os.path.exists(res_path):
            # 容错：如果文件名变了，取文件夹里第一个
            files = os.listdir(self.temp_out_dir)
            if files: res_path = os.path.join(self.temp_out_dir, files[0])

        return cv2.imread(res_path) if os.path.exists(res_path) else img_np

    def run_sparse_deconv(self, img_np):
        """运行 Sparse Deconv"""
        # (H,W,C) -> (C,H,W)
        img_trans = np.transpose(img_np, [2, 0, 1])

        img_recon = sparse_deconv(
            img_trans, self.resolution / self.pixelsize,
            background=self.background, tcontinuity=self.tcontinuity,
            sparsity=self.sparsity, fidelity=self.fidelity
        )

        # 后处理 (C,H,W) -> (H,W,C)
        img_recon = np.transpose(img_recon, (1, 2, 0))
        img_recon = np.clip(img_recon * 1.1, 0, 255)
        return img_recon.astype(np.uint8)

    def process(self, img_np):
        """执行完整流水线"""
        print("--- Step 1 ---")
        img_s1 = self.run_resshift(img_np)
        print("--- Step 2 ---")
        img_s1_re = cv2.resize(img_s1, (924, 924), interpolation=cv2.INTER_LINEAR)
        img_s2 = self.run_sparse_deconv(img_s1_re)
        return img_s2