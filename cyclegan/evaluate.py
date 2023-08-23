import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
from medpy.metric.binary import dc, assd

import model  # import your PyTorch model here
from stats_func import *  # make sure to convert these functions to PyTorch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CHECKPOINT_PATH = '' # model path
BASE_FID = '' # folder path of test files
TESTFILE_FID = '' # path of the .txt file storing the test filenames
TEST_MODALITY = 'CT'
KEEP_RATE = 1.0
IS_TRAINING = False
BATCH_SIZE = 128

data_size = [256, 256, 1]
label_size = [256, 256, 1]

contour_map = {
    "bg": 0,
    "la_myo": 1,
    "la_blood": 2,
    "lv_blood": 3,
    "aa": 4,
}

class MyDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])['arr_0']
        label = np.load(self.file_list[idx])['arr_1']
        return torch.from_numpy(data).float(), torch.from_numpy(label).long()

class SIFA:
    """The SIFA module."""

    def __init__(self, config):

        self.keep_rate = KEEP_RATE
        self.is_training = IS_TRAINING
        self.checkpoint_path = CHECKPOINT_PATH
        self.batch_size = BATCH_SIZE

        self._pool_size = int(config['pool_size'])
        self._skip = bool(config['skip'])
        self._num_cls = int(config['num_cls'])

        self.base_fd = BASE_FID
        self.test_fid = os.path.join(BASE_FID, TESTFILE_FID)

        self.model = model.SIFA(config)  # Assuming SIFA is the model's name
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.model.eval()

    def read_lists(self, fid):
        """read test file list """
        with open(fid, 'r') as fd:
            _list = fd.readlines()

        my_list = [os.path.join(self.base_fd, _item.strip()) for _item in _list]
        return my_list

    def test(self):
        """Test Function."""

        test_list = self.read_lists(self.test_fid)
        test_dataset = MyDataset(test_list)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        dice_list = []
        assd_list = []
        for idx, (data, label) in enumerate(test_loader):

            data, label = data.to(device), label.to(device)

            with torch.no_grad():
                output = self.model(data)

            dice_list.append(dc(output.detach().cpu().numpy(), label.detach().cpu().numpy()))
            assd_list.append(assd(output.detach().cpu().numpy(), label.detach().cpu().numpy()))

        # Continue with your evaluation as before

def main(config_filename):

    with open(config_filename) as config_file:
        config = json.load(config_file)

    sifa_model = SIFA(config)
    sifa_model.test()


if __name__ == '__main__':
    main(config_filename='./config_param.json')