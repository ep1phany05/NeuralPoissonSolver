import cv2
import torch
import numpy as np

from src.models.diner.diner import DINER
from src.solver import to_matlab
from src.solver.grad_operator import compute_grad, combine_grads, blend_grads


def create_inr(model_path, h, w, ch, device):
    model = DINER(hash_table_length=h * w, out_features=ch).to(device)
    model.load_pretrained(model_path, device)
    output = model(None)
    output["model_out"] = output["model_out"].view(h, w, ch)
    return output["model_out"]


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# src_grad_x, src_grad_y = capture_inr_grads("data/2d/scene_1/src.pth", 500, 500, 3, device)
# tgt_grad_x, tgt_grad_y = capture_inr_grads("data/2d/scene_1/tgt.pth", 300, 400, 3, device)
# cv2.imwrite("src_grad_x.png", src_grad_x.detach().cpu().numpy()[:, :, ::-1] * 255)
# cv2.imwrite("src_grad_y.png", src_grad_y.detach().cpu().numpy()[:, :, ::-1] * 255)
# cv2.imwrite("tgt_grad_x.png", tgt_grad_x.detach().cpu().numpy()[:, :, ::-1] * 255)
# cv2.imwrite("tgt_grad_y.png", tgt_grad_y.detach().cpu().numpy()[:, :, ::-1] * 255)

src = create_inr("data/2d/scene_1/src.pth", 500, 500, 3, device)
tgt = create_inr("data/2d/scene_1/tgt.pth", 300, 400, 3, device)
iou = torch.Tensor(cv2.imread("data/2d/scene_1/iou.png", cv2.IMREAD_GRAYSCALE))
cfg = np.load("data/2d/scene_1/iou.npy", allow_pickle=True).item()

print(src.shape, tgt.shape, iou.shape)
filled_iou, cmb_grad_x, cmb_grad_y = blend_grads(src, tgt, cfg['p'], iou, "max")
print(cmb_grad_x.shape, filled_iou.shape)
