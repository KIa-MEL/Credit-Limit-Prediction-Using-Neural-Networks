import torch

class Global(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Global, cls).__new__(cls)
            cls.__device = torch.device('cpu')
        return cls.instance
    @staticmethod
    def set_device_mode(mode):
        Global.__device = torch.device(mode)

    def get_device(moed):
        return Global.__device
