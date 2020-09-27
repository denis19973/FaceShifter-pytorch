from network.MultiScaleDiscriminator import *
from torch.utils.data import DataLoader
from demo.lib import swap_faces
from face_modules.model import Backbone
from utils.Dataset import FaceEmbed
import torch.nn.functional as F
import torch.optim as optim
from network.aei import *
from apex import amp
import torchvision
import visdom
import torch
import time
import cv2
from utils.visdom_plotter import VisdomLinePlotter

vis = visdom.Visdom(env='faceshifter', port=8097)
plotter = VisdomLinePlotter(vis, env_name='AEI Training')
batch_size = 64
lr_G = 4e-4
lr_D = 4e-4
max_epoch = 2000
show_step = 10
save_visual_check_epoch = 1
save_epoch = 1
model_save_path = './saved_models/'
optim_level = 'O1'

VISUAL_CHECK_SOURCE_PATH = '../for_vis_check/dawkins.jpg'
VISUAL_CHECK_TARGET_PATH = '../for_vis_check/me.jpg'

device = torch.device('cuda')

G = AEI_Net(c_id=512)
D = MultiscaleDiscriminator(input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d)
G = G.to(device)
D = D.to(device)

G.train()
D.train()

arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load('./saved_models/model_ir_se50.pth', map_location=device), strict=False)

opt_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0, 0.999))

G, opt_G = amp.initialize(G, opt_G, opt_level=optim_level)
D, opt_D = amp.initialize(D, opt_D, opt_level=optim_level)

try:
    G.load_state_dict(torch.load('./saved_models/G_latest.pth', map_location=torch.device('cpu')), strict=False)
    D.load_state_dict(torch.load('./saved_models/D_latest.pth', map_location=torch.device('cpu')), strict=False)
except Exception as e:
    print(e)

dataset = FaceEmbed(['../celeba_64/'], same_prob=0.8)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()


def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1 - X).mean()
    else:
        return torch.relu(X + 1).mean()


def get_grid_image(X):
    X = X[:8]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5 + 0.5
    return X


def make_image(Xs, Xt, Y):
    Xs = get_grid_image(Xs)
    Xt = get_grid_image(Xt)
    Y = get_grid_image(Y)
    return torch.cat((Xs, Xt, Y), dim=1).numpy()


def write_image(path, img):
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    cv2.imwrite(path, img)


def swap_and_save_result(epoch, Xs_raw, Xt_raw):
    s2t = swap_faces(Xs_raw, Xt_raw)
    write_image(f'./vis_check/result_s2t_{epoch}.jpg', s2t)
    Xs_raw, Xt_raw = Xt_raw, Xs_raw
    t2s = swap_faces(Xs_raw, Xt_raw)
    write_image(f'./vis_check/result_t2s_{epoch}.jpg', t2s)


Xs_visual_check_raw = cv2.imread(VISUAL_CHECK_SOURCE_PATH)
Xt_visual_check_raw = cv2.imread(VISUAL_CHECK_TARGET_PATH)

print(torch.backends.cudnn.benchmark)
for epoch in range(0, max_epoch):
    for iteration, data in enumerate(dataloader):
        start_time = time.time()
        Xs, Xt, same_person = data
        Xs = Xs.to(device)
        Xt = Xt.to(device)
        with torch.no_grad():
            embed, Xs_feats = arcface(F.interpolate(Xs, [112, 112], mode='bilinear', align_corners=True))
        same_person = same_person.to(device)

        # train G
        opt_G.zero_grad()
        Y, Xt_attr = G(Xt, embed)

        Di = D(Y)
        L_adv = 0

        for di in Di:
            L_adv += hinge_loss(di[0], True)

        Y_aligned = Y
        ZY, Y_feats = arcface(F.interpolate(Y_aligned, [112, 112], mode='bilinear', align_corners=True))
        L_id = (1 - torch.cosine_similarity(embed, ZY, dim=1)).mean()

        Y_attr = G.get_attr(Y)
        L_attr = 0
        for i in range(len(Xt_attr)):
            L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(batch_size, -1), dim=1).mean()
        L_attr /= 2.0

        L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1) * same_person) / (
                    same_person.sum() + 1e-6)

        l_adv = 1
        l_att = 10
        l_id = 1
        l_rec = 10

        lossG = l_adv * L_adv + l_att * L_attr + l_id * L_id + l_rec * L_rec

        with amp.scale_loss(lossG, opt_G) as scaled_loss:
            scaled_loss.backward()

        opt_G.step()

        # train D
        opt_D.zero_grad()
        fake_D = D(Y.detach())
        loss_fake = 0
        for di in fake_D:
            loss_fake += hinge_loss(di[0], False)

        true_D = D(Xs)
        loss_true = 0
        for di in true_D:
            loss_true += hinge_loss(di[0], True)

        lossD = 0.5 * (loss_true.mean() + loss_fake.mean())

        with amp.scale_loss(lossD, opt_D) as scaled_loss:
            scaled_loss.backward()
        opt_D.step()
        batch_time = time.time() - start_time
        image = make_image(Xs, Xt, Y)
        if iteration % show_step == 0:
            vis.image(image[::-1, :, :], opts={'title': 'result'}, win='result')
            cv2.imwrite('./gen_images/latest.jpg', image.transpose([1, 2, 0]))
            print(f'epoch: {epoch}    {iteration} / {len(dataloader)}')
            plotter.plot('loss G, D', 'D', 'Discriminator loss', iteration, lossD.item())
            plotter.plot('loss G, D', 'G', 'Generator loss', iteration, lossG.item())
            plotter.plot('loss adv, id, attr', 'adv', 'Adversarial loss', iteration, L_adv.item())
            plotter.plot('loss adv, id, attr', 'id', 'Identity loss', iteration, L_id.item())
            plotter.plot('loss adv, id, attr', 'attr', 'Attributes loss', iteration, L_attr.item())
            vis.save(['AEI Training'])
            print(f'lossD: {lossD.item()}    lossG: {lossG.item()} batch_time: {batch_time}s')
            print(f'L_adv: {L_adv.item()} L_id: {L_id.item()} L_attr: {L_attr.item()} L_rec: {L_rec.item()}')
        if iteration % 1000 == 0:
            torch.save(G.state_dict(), './saved_models/G_latest.pth')
            torch.save(D.state_dict(), './saved_models/D_latest.pth')
    if epoch % save_visual_check_epoch == 0:
        swap_and_save_result(epoch, Xs_visual_check_raw, Xt_visual_check_raw)
