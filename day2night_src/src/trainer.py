"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen, DomainAgnosticClassifier
from utils import weights_init, get_model_list, vgg_preprocess, resnet_preprocess, load_vgg16, load_resnet18, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        # Load ResNet model if needed
        if 'resnet_w' in hyperparameters.keys() and hyperparameters['resnet_w'] > 0:
            self.resnet = load_resnet18(hyperparameters['resnet_model_path'] + '/models')
            self.resnet.eval()
            for param in self.resnet.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_resnet_a = self.compute_resnet_loss(self.resnet, x_ba, x_b) if hyperparameters['resnet_w'] > 0 else 0
        self.loss_gen_resnet_b = self.compute_resnet_loss(self.resnet, x_ab, x_a) if hyperparameters['resnet_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
                              hyperparameters['resnet_w'] * self.loss_gen_resnet_a + \
                              hyperparameters['resnet_w'] * self.loss_gen_resnet_b
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def compute_resnet_loss(self, resnet, img, target):
        img_resnet = resnet_preprocess(img)
        target_resnet = resnet_preprocess(target)
        img_fea = resnet(img_resnet)
        target_fea = resnet(target_resnet)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def snap_clean(self, snap_dir, iterations, save_last=10000, period=50000):
        # Cleaning snapshot directory from old files
        if not os.path.exists(snap_dir):
            return None

        gen_models = [os.path.join(snap_dir, f) for f in os.listdir(snap_dir) if "gen" in f and ".pt" in f]
        dis_models = [os.path.join(snap_dir, f) for f in os.listdir(snap_dir) if "dis" in f and ".pt" in f]

        gen_models.sort()
        dis_models.sort()
        marked_clean = []
        for i, model in enumerate(gen_models):
            m_iter = int(model[-11:-3])
            if i == 0:
                m_prev = 0
                continue
            if m_iter > iterations - save_last:
                break
            if m_iter - m_prev < period:
                marked_clean.append(model)
            while m_iter - m_prev >= period:
                m_prev += period

        for i, model in enumerate(dis_models):
            m_iter = int(model[-11:-3])
            if i == 0:
                m_prev = 0
                continue
            if m_iter > iterations - save_last:
                break
            if m_iter - m_prev < period:
                marked_clean.append(model)
            while m_iter - m_prev >= period:
                m_prev += period

        print(f'Cleaning snapshots: {marked_clean}')
        for f in marked_clean:
            os.remove(f)

    def save(self, snapshot_dir, iterations, smart_override):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
        if smart_override:
            self.snap_clean(snapshot_dir, iterations+1)

class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.gen_a.to(device)
        # from torchsummary import summary
        # summary(self.gen_a.encode(), input_size=(3, 256, 256))
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        # Load ResNet model if needed        
        if 'resnet_w' in hyperparameters.keys() and hyperparameters['resnet_w'] > 0:
            self.resnet = load_resnet18(hyperparameters['resnet_model_path'] + '/models')
            self.resnet.eval()
            for param in self.resnet.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)#Lvae 第二项
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a) #Lvae 第一项
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a) #Lcc 第三项
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)#Lcc 第二项
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_resnet_a = self.compute_resnet_loss(self.resnet, x_ba, x_b) if hyperparameters['resnet_w'] > 0 else 0
        self.loss_gen_resnet_b = self.compute_resnet_loss(self.resnet, x_ab, x_a) if hyperparameters['resnet_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
                              hyperparameters['resnet_w'] * self.loss_gen_resnet_a + \
                              hyperparameters['resnet_w'] * self.loss_gen_resnet_b
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def compute_resnet_loss(self, resnet, img, target):
        img_resnet = resnet_preprocess(img)
        target_resnet = resnet_preprocess(target)
        img_fea = resnet(img_resnet)
        target_fea = resnet(target_resnet)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def snap_clean(self, snap_dir, iterations, save_last=10000, period=50000):
        # Cleaning snapshot directory from old files
        if not os.path.exists(snap_dir):
            return None
        
        gen_models = [os.path.join(snap_dir, f) for f in os.listdir(snap_dir) if "gen" in f and ".pt" in f]
        dis_models = [os.path.join(snap_dir, f) for f in os.listdir(snap_dir) if "dis" in f and ".pt" in f]
        
        gen_models.sort()
        dis_models.sort()
        marked_clean = []
        for i, model in enumerate(gen_models):
            m_iter = int(model[-11:-3])
            if i == 0:
                m_prev = 0
                continue
            if m_iter > iterations - save_last:
                break
            if m_iter - m_prev < period:
                marked_clean.append(model)
            while m_iter - m_prev >= period:
                m_prev += period
        
        for i, model in enumerate(dis_models):
            m_iter = int(model[-11:-3])
            if i == 0:
                m_prev = 0
                continue
            if m_iter > iterations - save_last:
                break
            if m_iter - m_prev < period:
                marked_clean.append(model)
            while m_iter - m_prev >= period:
                m_prev += period
        
        print(f'Cleaning snapshots: {marked_clean}')
        for f in marked_clean:
            os.remove(f)

    def save(self, snapshot_dir, iterations, smart_override):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
        if smart_override:
            self.snap_clean(snapshot_dir, iterations+1)


class Fork_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(Fork_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        # self.netC = DomainAgnosticClassifier(input_nc=hyperparameters['c']['input_dim'])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.gen_a.to(device)
        # from torchsummary import summary
        # summary(self.gen_a.encode(), input_size=(3, 256, 256))
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']

        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        # netC_params = list(self.netC.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        # self.netC_opt = torch.optim.Adam([p for p in netC_params if p.requires_grad],
        #                                 lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])


        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        # self.netC_scheduler = get_scheduler(self.netC_opt, hyperparameters)


        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))
        # self.netC.apply(weights_init('kaiming'))


        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        # Load ResNet model if needed
        if 'resnet_w' in hyperparameters.keys() and hyperparameters['resnet_w'] > 0:
            self.resnet = load_resnet18(hyperparameters['resnet_model_path'] + '/models')
            self.resnet.eval()
            for param in self.resnet.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    # def forward(self, x_a, x_b):
    #     self.eval()
    #     h_a, _ = self.gen_a.encode(x_a)
    #     h_b, _ = self.gen_b.encode(x_b)
    #     x_ba = self.gen_a.decode(h_b)
    #     x_ab = self.gen_b.decode(h_a)
    #     self.train()
    #     return x_ab, x_ba

    def forward(self, input_a, input_b, hyperparameters):
        # encode
        self.x_a = input_a
        self.x_b = input_b
        self.h_a, _ = self.gen_a.encode(self.x_a)
        self.h_b, _ = self.gen_b.encode(self.x_b)
        # decode (within domain)
        self.x_a_recon = self.gen_a.decode(self.h_a)
        self.x_b_recon = self.gen_b.decode(self.h_b)
        # decode (cross domain)
        self.x_ba = self.gen_a.decode(self.h_b)
        self.x_ab = self.gen_b.decode(self.h_a)
        # encode again
        self.h_ba, _ = self.gen_a.encode(self.x_ba)
        self.h_ab, _ = self.gen_b.encode(self.x_ab)
        # decode again (if needed)
        # self.x_ba_recon = self.gen_a.decode(self.h_ba)
        # self.x_ab_recon = self.gen_b.decode(self.h_ab)

        self.x_aba = self.gen_a.decode(self.h_ab) if hyperparameters['recon_x_cyc_w'] > 0 else None
        self.x_bab = self.gen_b.decode(self.h_ba) if hyperparameters['recon_x_cyc_w'] > 0 else None



        # self.h_a_logit = self.netC(self.h_a)
        # self.h_b_logit = self.netC(self.h_b)
        # self.h_ab_logit = self.netC(self.h_a_recon)
        # self.h_ba_logit = self.netC(self.h_b_recon)

        # self.A_label = torch.zeros([self.x_a.size(0)], dtype=torch.long).to(self.device)
        # self.B_label = torch.ones([self.x_b.size(0)], dtype=torch.long).to(self.device)

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, hyperparameters):
        self.gen_opt.zero_grad()
        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(self.x_a_recon, self.x_a)  # Lvae 第二项
        self.loss_gen_recon_x_b = self.recon_criterion(self.x_b_recon, self.x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(self.h_a)  # Lvae 第一项
        self.loss_gen_recon_kl_b = self.__compute_kl(self.h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(self.x_aba, self.x_a)  # Lcc 第三项
        self.loss_gen_cyc_x_b = self.recon_criterion(self.x_bab, self.x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(self.h_ab)  # Lcc 第二项
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(self.h_ba)

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(self.x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(self.x_ab)

        # self.loss_gen_adv_rec_a = self.dis_a.calc_gen_loss(self.x_a_recon)
        # self.loss_gen_adv_rec_b = self.dis_b.calc_gen_loss(self.x_b_recon)
        #
        # self.loss_gen_adv_ref_a = self.dis_a.calc_gen_loss(self.x_aba)
        # self.loss_gen_adv_ref_b = self.dis_b.calc_gen_loss(self.x_bab)


        # self.loss_gen_adv_total = 0.5 * (self.loss_gen_adv_a + self.loss_gen_adv_b) + \
        #                           0.5 * (self.loss_gen_adv_rec_a + self.loss_gen_adv_rec_b) + \
        #                           0.5 * (self.loss_gen_adv_ref_a + self.loss_gen_adv_ref_b)

        self.loss_gen_adv_total = (self.loss_gen_adv_a + self.loss_gen_adv_b)



        # Generator classification loss
        # self.loss_gen_cls = F.cross_entropy(self.h_ba_logit.reshape(-1, 2), self.A_label) * 0.5 + \
        #                   F.cross_entropy(self.h_ab_logit.reshape(-1, 2), self.B_label) * 0.5

        # domain-invariant perceptual loss
        self.loss_gen_percep = torch.mean(
            torch.abs(torch.mean(self.h_a, dim=3) - torch.mean(self.h_ab, dim=3))) + \
                             torch.mean(torch.abs(torch.mean(self.h_b, dim=3) - torch.mean(self.h_ba, dim=3)))

        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, self.x_ba, self.x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, self.x_ab, self.x_a) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_resnet_a = self.compute_resnet_loss(self.resnet, self.x_ba, self.x_b) if hyperparameters[
                                                                                         'resnet_w'] > 0 else 0
        self.loss_gen_resnet_b = self.compute_resnet_loss(self.resnet, self.x_ab, self.x_a) if hyperparameters[
                                                                                         'resnet_w'] > 0 else 0

        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_total + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
                              hyperparameters['resnet_w'] * self.loss_gen_resnet_a + \
                              hyperparameters['resnet_w'] * self.loss_gen_resnet_b + \
                              hyperparameters['percep_w'] * self.loss_gen_percep\
                              # + \
                              # hyperparameters['cls_w'] * self.loss_gen_cls

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def compute_resnet_loss(self, resnet, img, target):
        img_resnet = resnet_preprocess(img)
        target_resnet = resnet_preprocess(target)
        img_fea = resnet(img_resnet)
        target_fea = resnet(target_resnet)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))

            # h_ab, _ = self.gen_b.encode(x_ab[i].unsqueeze(0))
            # h_ba, _ = self.gen_a.encode(x_ba[i].unsqueeze(0))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, hyperparameters):
        self.dis_opt.zero_grad()

        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(self.x_ba.detach(), self.x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(self.x_ab.detach(), self.x_b)
        # self.loss_dis_rec_a = self.dis_a.calc_dis_loss(self.x_a_recon.detach(), self.x_a)
        # self.loss_dis_rec_b = self.dis_b.calc_dis_loss(self.x_b_recon.detach(), self.x_b)
        # self.loss_dis_ref_a = self.dis_a.calc_dis_loss(self.x_aba.detach(), self.x_a)
        # self.loss_dis_ref_b = self.dis_b.calc_dis_loss(self.x_bab.detach(), self.x_b)
        #
        # self.loss_dis =0.5 * (self.loss_dis_a + self.loss_dis_b) + \
        #                     0.5 * (self.loss_dis_rec_a + self.loss_dis_rec_b) + \
        #                         0.5 * (self.loss_dis_ref_a + self.loss_dis_ref_b)

        self.loss_dis = (self.loss_dis_a + self.loss_dis_b)

        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis
        self.loss_dis_total.backward()
        self.dis_opt.step()

    # def C_update(self):
    #     """Calculate domain-agnostic classifier loss for generator and itself"""
    #     self.netC_opt.zero_grad()
    #     self.loss_cls = F.cross_entropy(self.h_a_logit.reshape(-1, 2).detach(), self.A_label)*0.25 +\
    #                         F.cross_entropy(self.h_b_logit.reshape(-1, 2).detach(), self.B_label)*0.25 +\
    #                             F.cross_entropy(self.h_ab_logit.reshape(-1, 2).detach(), self.A_label)*0.25 +\
    #                                 F.cross_entropy(self.h_ba_logit.reshape(-1, 2).detach(), self.B_label)*0.25
    #     self.loss_cls.backward()
    #     self.netC_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        # if self.gen_scheduler is not None:
        #     self.netC_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def snap_clean(self, snap_dir, iterations, save_last=10000, period=50000):
        # Cleaning snapshot directory from old files
        if not os.path.exists(snap_dir):
            return None

        gen_models = [os.path.join(snap_dir, f) for f in os.listdir(snap_dir) if "gen" in f and ".pt" in f]
        dis_models = [os.path.join(snap_dir, f) for f in os.listdir(snap_dir) if "dis" in f and ".pt" in f]

        gen_models.sort()
        dis_models.sort()
        marked_clean = []
        for i, model in enumerate(gen_models):
            m_iter = int(model[-11:-3])
            if i == 0:
                m_prev = 0
                continue
            if m_iter > iterations - save_last:
                break
            if m_iter - m_prev < period:
                marked_clean.append(model)
            while m_iter - m_prev >= period:
                m_prev += period

        for i, model in enumerate(dis_models):
            m_iter = int(model[-11:-3])
            if i == 0:
                m_prev = 0
                continue
            if m_iter > iterations - save_last:
                break
            if m_iter - m_prev < period:
                marked_clean.append(model)
            while m_iter - m_prev >= period:
                m_prev += period

        print(f'Cleaning snapshots: {marked_clean}')
        for f in marked_clean:
            os.remove(f)

    def save(self, snapshot_dir, iterations, smart_override):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
        if smart_override:
            self.snap_clean(snapshot_dir, iterations + 1)
