import os.path

import matplotlib.pyplot as plt

from WSCON.data.base_dataset import BaseDataset, get_transform,get_params
from WSCON.data.image_folder import make_dataset
from PIL import Image
import WSCON.util.util as util


class AlignedDataset(BaseDataset):
    """
    This dataset class can load aligned (paired) datasets.

    It requires a directory to host training images from both domains A and B in the same order.
    The dataset can be accessed by providing '--dataroot /path/to/data'.
    For example, the data directory could contain 'trainA' and 'trainB' directories,
    where the corresponding images are in the same order in each directory.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        # Ensure both directories have the same number of images and load them in a consistent order
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'

        # Ensure that both datasets A and B have the same size
        assert len(self.A_paths) == len(self.B_paths), "The datasets A and B must have the same number of images"

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths, and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % len(self.A_paths)]  # make sure index is within the range
        B_path = self.B_paths[index % len(self.B_paths)]  # use the same index for domain B

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Apply image transformation
        # For CUT/FastCUT mode, if in the fine-tuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation as in CycleGAN.
        transform_params = get_params(self.opt, A_img.size)
        transform = get_transform(self.opt, transform_params)

        A = transform(A_img)
        B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        Since the datasets A and B are aligned, we can return either one as both should be of the same length.
        """
        return len(self.A_paths)  # or len(self.B_paths), they are equal
