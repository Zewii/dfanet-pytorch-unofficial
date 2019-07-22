import torch
from models.enet import ENet
import os
from configs import config_factory



if __name__=='__main__':
    save_pth = os.path.join(config_factory['resnet_cityscapes'].respth, 'model_final.pth')
    model = ENet(nb_classes=19)
    model.load_state_dict(torch.load(save_pth))
    model.eval()
    example = torch.rand(2, 3, 1024, 1024).cpu()
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(os.path.join(config_factory['resnet_cityscapes'].respth,"model_dfanet_1024.pt"))

