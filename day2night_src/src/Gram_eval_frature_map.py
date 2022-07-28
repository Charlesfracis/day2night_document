from data import make_dataset
from torchvision import transforms
import cv2
import numpy as np
import os
import torch
from networks import Vgg19
from utils_day2night import get_data_loader_folder, weights_init, get_model_list, vgg_preprocess, resnet_preprocess, load_vgg16, load_resnet18, get_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import ImageFilelist, ImageFolder
import time
from progress.bar import Bar
from sklearn.metrics import mean_squared_error
import os



def load_vgg19(model_dir):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg19.weight')):
        vggpth = torch.load(os.path.join('/data/day2night/day2night/UNIT/models/vgg_conv.pth'))
        vgg = Vgg19()
        for (src, dst) in zip(vggpth, vgg.parameters()):
            dst.data[:] = vggpth[src]
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg19.weight'))
    vgg = Vgg19()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg19.weight')))
    return vgg



def get_data_loader_folder_vgg19(input_folder, batch_size, new_size=True,
                           height=1080, width=1080, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.40760392, 0.45795686, 0.48501961),
                                           (1, 1, 1))]
    # transform_list = [transforms.CenterCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize((1080,1920))] + transform_list if new_size is not None else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform, return_paths=True)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True, num_workers=num_workers)
    return loader

def featuremap_dot_matrix(style_activation, base_featuremap):
    # Output activations for each filter convolution
    # Shape is: activations x channels
    style_activation = style_activation.cpu().numpy().squeeze()
    c, h, w = style_activation.shape
    gram = np.zeros((c, c))
    flat_style_activations = np.reshape(style_activation, [style_activation.shape[0], -1])
    flat_base_featuremap = np.reshape(base_featuremap, [base_featuremap.shape[0], -1])
    # Inner Product between all feature activations scaled by total number of activations
    # Shape is: (channels x activations) x (activations x channels) = channels x channels
    for i in range(c):
        for j in range(c):
            dot_product = flat_style_activations[i].dot(flat_base_featuremap[j])
            norms_product = np.linalg.norm(flat_style_activations[i]) * np.linalg.norm(flat_base_featuremap[j])
            gram[i][j] = dot_product/norms_product
            if np.isnan(gram[i][j]):
                gram[i][j] = 0
    return gram
#
# def get_mse(style_activations, tars):
#     assert len(style_activations) == len(tars), 'length not equal'
#     length = len(style_activations)
#
#     for i in range(length):
#         style_activations_numpy = gram_matrix(style_activations[i])
#         tar = tars[i]
#         mse_res[i].append(mean_squared_error(style_activations_numpy, tar))
#     return 0


def get_folder_content(dir):
    lists = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for i, fname in enumerate(fnames):
            path = os.path.join(root, fname)
            lists.append(path)
        break
    paths = sorted(lists)


    return paths

def read_grams(paths):
    tars = []
    for i in paths:
        tars.append(np.load(i))
    return tars

def accumulate_res(outs, idx, input):
    for i in range(len(input)):
            outs[i].append(input[i])
        # outs[i] = out[i].numpy() + outs[i]
    return 0

def get_base_featuremap(path):
    feature_images_path = []
    features = []
    assert os.path.isdir(path), '%s is not a valid directory' % path
    for root, _, fnames in sorted(os.walk(path)):
        for i, fname in enumerate(fnames):
            path = os.path.join(root, fname)
            feature_images_path.append(path)

    feature_images_path = sorted(feature_images_path)

    for i in feature_images_path:
        feature = np.load(i)
        features.append(feature)
    return features

def get_feature_gram(input_features, base_features):
    assert len(input_features) == len(base_features), 'length not equal'
    gram_result = []
    for i in range(len(input_features)):
        gram = featuremap_dot_matrix(input_features[i], base_features[i])
        gram_result.append(gram)
    return gram_result

def get_mean(input):
    output = []
    for i in input:
        output.append(np.mean(i))
    return output

def get_diag_mean(input):
    output = []
    for i in input:
        diagonal = np.diag(i)
        diag_mean = np.mean(diagonal)
        output.append(diag_mean)
    return output



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    outs_all = []
    outs_diag = []
    base_features = get_base_featuremap('/data/day2night/day2night/UNIT/featuremap/night1000_features_ori')
    data_path = '/data/day2night/dataset/test/night_obstacle'
    loader = get_data_loader_folder_vgg19(data_path, batch_size=1, new_size=True)
    vgg = load_vgg19('/data/day2night/day2night/UNIT/models')
    vgg.to(device)
    vgg.eval()

    for param in vgg.parameters():
        param.requires_grad = False
    img_num = len(loader)
    style_layers = ['r11','r21','r31','r41', 'r51']
    bar = Bar('test', max=img_num)
    with torch.no_grad():
        for idx, (img, path) in enumerate(loader):
            path = path[0]
            img = img.to(device)
            item_name = os.path.split(path)[-1]
            out = vgg(img, style_layers)
            feature_gram = get_feature_gram(out, base_features)
            all_mean = get_mean(feature_gram)
            outs_all.append(all_mean)
            diag_mean = get_diag_mean(feature_gram)
            outs_diag.append(diag_mean)
            bar.next()
            Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                idx, len(loader), total=bar.elapsed_td, eta=bar.eta_td)
            print(Bar.suffix)

    bar.finish()

    # np.save('/data/day2night/day2night/UNIT/featuremap/day_outs_all.npy', outs_all)
    # np.save('/data/day2night/day2night/UNIT/featuremap/day_outs_diag.npy', outs_diag)



print()








