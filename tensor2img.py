import cv2
import torch
import matplotlib.pyplot as plt

t = torch.randn(1, 64, 192, 640)


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((0, 1, 2))
    return img


# def show_from_cv(img, title=None):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.figure()
#     plt.imshow(img)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)


def show_from_tensor(tensor, title=None):
    img = tensor.clone()
    img = tensor_to_np(img)
    plt.figure()
    for i in img:
        plt.imshow(i)
        if title is not None:
            plt.title(title)
        plt.pause(1)


show_from_tensor(t)
