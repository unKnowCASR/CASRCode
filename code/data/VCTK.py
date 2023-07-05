import os
from data import srdata


class DL(srdata.SRData):
    def __init__(self, args, name='VCTK', train=True, benchmark=False):
        super(DL, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(DL, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'VCTK_train_HR')
        self.dir_lr = os.path.join(self.apath, 'VCTK_train_LR_bicubic')

