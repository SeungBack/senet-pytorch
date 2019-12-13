import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
from model import ResNet, ResBottleNeck, SEBottleNeck, MSEBottleNeck, WSEBottleNeck, DSEBottleNeck,init_weight
from matplotlib.pyplot import imshow, imsave
import numpy as np
import cv2
from torchvision.utils import save_image


def reverse_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    return x


def visualize(img, cam):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        img: (Tensor) shape => (1, 3, H, W)
        cam: (Tensor) shape => (1, 1, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, H, W)
    """

    _, _, H, W = img.shape
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    heatmap = heatmap.float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap + img.cpu()
    result = result.div(result.max())

    return result


class SaveValues():
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class CAM(object):
    """ Class Activation Mapping """

    def __init__(self, model, target_layer):
        """
        Args:
            model: a base model to get CAM which have global pooling and fully connected layer.
            target_layer: conv_layer before Global Average Pooling
        """

        self.model = model
        self.target_layer = target_layer

        # save values of activations and gradients in target_layer
        self.values = SaveValues(self.target_layer)

    def forward(self, x):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # object classification
        score = self.model(x)
        prob = F.softmax(score, dim=1)
        max_prob, idx = torch.max(prob, dim=1)
        print(
            "predicted object ids {}\t probability {}".format(idx.item(), max_prob.item()))

        # cam can be calculated from the weights of linear layer and activations
        weight_fc = list(
            self.model._modules.get('fc').parameters())[0].to('cpu').data
        cam = self.getCAM(self.values, weight_fc, idx.item())

        return cam

    def __call__(self, x):
        return self.forward(x)

    def getCAM(self, values, weight_fc, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        weight_fc: the weight of fully connected layer.  shape => (num_classes, C)
        idx: predicted class id
        cam: class activation map.  shape => (1, num_classes, H, W)
        '''

        cam = F.conv2d(values.activations, weight=weight_fc[:, :, None, None])
        _, _, h, w = cam.shape

        # class activation mapping only for the predicted class
        # cam is normalized with min-max.
        cam = cam[:, idx, :, :]
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        cam = cam.view(1, 1, h, w)

        return cam.data

class GradCAM(CAM):
    """ Grad CAM """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)

        """
        Args:
            model: a base model to get CAM, which need not have global pooling and fully connected layer.
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of the predicted class
        """
        # object classification
        score = self.model(x)
        prob = F.softmax(score)
        max_prob, idx = torch.max(prob, dim=1)
        print(
            "predicted object ids {}\t probability {}".format(idx.item(), max_prob.item()))

        # caluculate cam of the predicted class
        cam = self.getGradCAM(self.values, score, idx.item())

        return cam

    def __call__(self, x):
        return self.forward(x)

    def getGradCAM(self, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''

        self.model.zero_grad()
        score[0, idx].backward(retain_graph=True)
        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape
        alpha = gradients.view(n, c, -1).mean(2)
        alpha = alpha.view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=str, default='res', help="res, se, mse, wse")
    parser.add_argument("--inplanes", type=int, default=64, help="number of in planes of resnet.")
    parser.add_argument("--load_weights", type=str, default=None, help="path to the pretrained weights")
    parser.add_argument("--output_folder", type=str, default='./results', help="path to the dataset root")
    args = parser.parse_args()

    # load image
    val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    val_dataset = datasets.CIFAR100('./dataset', train=False, transform=val_transform, download=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.block == 'res':
        block = ResBottleNeck
    elif args.block == 'se':
        block = SEBottleNeck
    elif args.block == 'mse':
        block = MSEBottleNeck
    elif args.block == 'wse':
        block = WSEBottleNeck
    elif args.block == 'dse':
        block = DSEBottleNeck

    model = ResNet(inplanes=args.inplanes, block=block).to(device)
    model.load_state_dict(torch.load(args.load_weights))
    # remove fc and GAP layer
    model.fc = nn.Sequential()
    model.layers[-1] = nn.Sequential()
    print(model)

    model.eval()

    # last layer in residual
    target_layer = model.layers[-2][-1].residual[-2]
    wrapped_model = GradCAM(model, target_layer)

    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        grad_cam = wrapped_model(images)
        imshow(grad_cam.squeeze().cpu().numpy(), alpha=0.8, cmap='jet')
        img = reverse_normalize(images)
        heatmap = visualize(img.cpu(), grad_cam.cpu())
        save_image(heatmap, './results/grad-cam_{}.png'.format(i))
        save_image(img, './results/grad-cam_{}_original.png'.format(i))
