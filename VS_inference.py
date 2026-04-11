import torch
from WSCON.options.test_options import TestOptions
from WSCON.models import create_model
from WSCON.data.base_dataset import get_transform
from PIL import Image
import numpy as np
import cv2
import WSCON.util.util as util


class VirtualStainingModel:
    def __init__(self):
        # 1. 初始化基础参数
        self.opt = TestOptions().parse()
        self.opt.num_threads = 0
        self.opt.batch_size = 1
        self.opt.serial_batches = True
        self.opt.no_flip = True
        self.opt.display_id = -1
        # 默认模型名称，稍后会被覆盖
        self.opt.name = 'default'
        self.model = None

        # 预处理通常是一样的，可以在这里初始化，也可以在 switch_model 里根据需要变
        self.transform = get_transform(self.opt, grayscale=(self.opt.input_nc == 1))



    def load_network(self, model_name):
        """
        核心方法：根据传入的模型文件夹名称加载模型
        model_name: 对应 checkpoints 文件夹下的子文件夹名，例如 'wscon_actin'
        """
        print(f"Loading Model: {model_name}...")

        # 如果模型名没变且模型已加载，直接返回，不做无用功
        if self.model is not None and self.opt.name == model_name:
            print("Model already loaded.")
            return

        # 更新参数中的模型名称
        self.opt.name = model_name

        # 释放旧显存 (可选，防止显存爆炸)
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

        # 创建并初始化新模型
        self.model = create_model(self.opt)
        self.model.setup(self.opt)
        self.model.eval()
        print(f"Model {model_name} Loaded Successfully.")

    def predict(self, image_path_or_array):
        # ... (保持之前的代码不变) ...
        # A. 预处理
        if isinstance(image_path_or_array, str):
            A_img = Image.open(image_path_or_array).convert('RGB')
        else:
            if len(image_path_or_array.shape) == 2:
                A_img = Image.fromarray(image_path_or_array)
            else:
                A_img = Image.fromarray(cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB))

        A = self.transform(A_img)
        print(A)
        A = A.unsqueeze(0)

        # 准备数据字典
        data = {'A': A, 'A_paths': ""}
        # 注意：这里不需要 B，因为我们是推理模式

        # 设置输入并推理
        try:
            self.model.set_input(data)
            with torch.no_grad():
                self.model.test()

            # ================= [修改开始] =================
            # 原来的代码：
            # visuals = self.model.get_current_visuals()  <-- 这行会报错，因为它试图找 real_B

            # 新的代码：直接获取 fake_B (生成的 B 图)
            # 在大多数 GAN 模型中，生成的图像都保存在 self.model.fake_B 中
            if hasattr(self.model, 'fake_B'):
                result_tensor = self.model.fake_B
            elif hasattr(self.model, 'fake_A'):  # 防止方向反了
                result_tensor = self.model.fake_A
            else:
                # 万一模型没生成，打印一下拥有的属性以便调试
                print(f"Error: No output found. Model attributes: {self.model.__dict__.keys()}")
                return np.zeros((800, 800, 3), dtype=np.uint8)

            return util.tensor2im(result_tensor)
            # ================= [修改结束] =================

        except Exception as e:
            print(f"Error during prediction step: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((512, 512, 3), dtype=np.uint8)