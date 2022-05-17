import numpy as np
import torch
from util import showtensor
import scipy.ndimage as nd
from torch.autograd import Variable
from numpy import linalg as LA


def objective_L2(dst, guide_features):
    return dst.data


def make_step(img, model, num_iterations = 20, control=None, distance=objective_L2):
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])

    learning_rate = 1e-1
    max_jitter = 32
    show_every = 10
    guide_features = control

    for i in range(num_iterations):
        shift_x, shift_y = np.random.randint(-max_jitter, max_jitter + 1, 2)
        img = np.roll(np.roll(img, shift_x, -1), shift_y, -2)
        # apply jitter shift
        model.zero_grad()
        img_tensor = torch.Tensor(img)
        if torch.cuda.is_available():
            img_variable = Variable(img_tensor.cuda(), requires_grad=True)
        else:
            img_variable = Variable(img_tensor, requires_grad=True)

        act_value = model.forward(img_variable)

        diff_out = distance(act_value, guide_features)
        act_value.backward(diff_out)

        ratio = np.abs(img_variable.grad.data.cpu().numpy()).mean()
        learning_rate_use = learning_rate #/ ratio
        img_variable.data.add_(img_variable.grad.data * learning_rate_use)
        img = img_variable.data.cpu().numpy()  # b, c, h, w
        img = np.roll(np.roll(img, -shift_x, -1), -shift_y, -2)
        #img[0, :, :, :] = np.clip(img[0, :, :, :], -mean / std, (1 - mean) / std)

        if i == 0 or (i + 1) % show_every == 0:
            showtensor(img)

        #print(f'Frobenius norm of the orig dream difference: {LA.norm(input_img-img)}')

    return img


def dream(model, input_img, num_iterations, control=None, distance=objective_L2):

        return make_step(input_img, model, num_iterations, control, distance=distance)
