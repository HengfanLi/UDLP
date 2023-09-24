import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
import cleanlab
import nibabel as nib
import math
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='16_high6_wothres', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

# if args.deterministic:
#     cudnn.benchmark = False
#     cudnn.deterministic = True
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)
def set_seed(seed):	

    np.random.seed(args.seed)
    random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
set_seed(1337)
num_classes = 2
patch_size = (112, 112, 80)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    

    model = create_model()
    ema_model = create_model(ema=True)

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    db_test = LAHeart(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor()
                       ]))
    labeled_idxs = list(range(16))
    unlabeled_idxs = list(range(16, 80))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    
    
    iter_num = 0
    #iter_num = 4000
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    max_probs_thresholds=0.7
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
           
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            af_label_batch = label_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_u = outputs[labeled_bs:]
            outputs_u_soft = torch.softmax(outputs_u, dim=1)
            
            
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
            ema_output_soft = F.softmax(ema_output, dim=1)
            
            pseudo_mask = (ema_output_soft > 0.7).float()
            outputs_weak_masked = ema_output_soft * pseudo_mask
            pseudo_outputs1 = torch.argmax(outputs_weak_masked.detach(), dim=1, keepdim=False)
            max_probs, _ = torch.max(ema_output_soft, dim=1)
            mask1 = (max_probs.ge(max_probs_thresholds)).float()
            ema_output_soft_mask1 = ema_output_soft*mask1
            T = 8
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2, 112, 112, 80]).cuda()
            for i in range(T//2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
           
            preds = preds.reshape(T, stride, 2, 112, 112, 80)
            preds = torch.mean(preds, dim=0)  #(batch, 2, 112,112,80)
           
            preds1 = preds.clone()
            uncertainty1 = -1.0*torch.sum(preds1*torch.log(preds1 + 1e-6), dim=1) #(batch, 1, 112,112,80)
            uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)
            ## calculate the loss
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            supervised_loss = 0.5*(loss_seg+loss_seg_dice)

            consistency_weight = get_current_consistency_weight(iter_num//150)

            consistency_dist = consistency_criterion(outputs[labeled_bs:], ema_output) #(batch, 2, 112,112,80)
           
            threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num, max_iterations))*np.log(2)
            mask = (uncertainty<threshold).float()
            
            high_confidence_loss = ce_loss(outputs_u, pseudo_outputs1)+ dice_loss(outputs_u_soft, pseudo_outputs1.unsqueeze(1))
            high_confidence_loss    = consistency_weight * high_confidence_loss
            consistency_dist = torch.sum(mask1*consistency_dist)/(2*torch.sum(mask1)+1e-16)
            consistency_loss = consistency_weight * consistency_dist
            ###########################################
            noisy_label_batch = af_label_batch#label_batch[args.labeled_bs:]
            with torch.no_grad():
                outputs1 = ema_model(unlabeled_volume_batch)
                outputs_soft_label_u = torch.softmax(outputs1, dim=1)
                _, label_u = torch.max(outputs_soft_label_u, dim=1)
                
                outputs_mask = outputs1*mask
                outputs_mask_soft = torch.softmax(outputs_mask, dim=1)
                max_probs1, _ = torch.max(outputs_mask_soft, dim=1)
                mask2 = (max_probs1.le(0.7)).float()
                outputs_soft12 = outputs_mask_soft*mask2
            masks_np = noisy_label_batch.cpu().detach().numpy()#label_batch[args.labeled_bs:]
            ema_output_soft_np = outputs_soft12.cpu().detach().numpy()

            # 2: identify the noise map
            ema_output_soft_np_accumulated_0 = np.swapaxes(ema_output_soft_np, 1, 2)
            ema_output_soft_np_accumulated_1 = np.swapaxes(ema_output_soft_np_accumulated_0, 2, 3)
            ema_output_soft_np_accumulated_2 = np.swapaxes(ema_output_soft_np_accumulated_1, 3, 4)
            ema_output_soft_np_accumulated_3 = ema_output_soft_np_accumulated_2.reshape(-1, num_classes)
            ema_output_soft_np_accumulated = np.ascontiguousarray(ema_output_soft_np_accumulated_3)
           
            masks_np_accumulated = masks_np.reshape(-1).astype(np.uint8)		
           
            assert masks_np_accumulated.shape[0] == ema_output_soft_np_accumulated.shape[0]

            noise = cleanlab.filter.find_label_issues(masks_np_accumulated, ema_output_soft_np_accumulated, filter_by='both', n_jobs=1)
            confident_maps_np = noise.reshape(-1, patch_size[0], patch_size[1], patch_size[2]).astype(np.uint8)

            smooth_arg = 0.8
            corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np) * smooth_arg
            
            noisy_label_batch = torch.from_numpy(corrected_masks_np).cuda(outputs_soft.device.index)
            loss_ce_weak = ce_loss(outputs[args.labeled_bs:], noisy_label_batch.long())
            loss_dice_weak = dice_loss(outputs_soft[args.labeled_bs:], noisy_label_batch.long().unsqueeze(1))
            
            weak_weight = 8
            weak_supervised = (loss_dice_weak + loss_ce_weak)*weak_weight
            weak_supervised_loss = consistency_weight *weak_supervised
###############################################################################
            drop_percent = 40
            percent_unreliable = (100 - drop_percent) * (1 - epoch_num / max_epoch)
            drop_percent = 100 - percent_unreliable
            target=label_u.clone()
            thresh1 = np.percentile(uncertainty1[target!=2].detach().cpu().numpy().flatten(), drop_percent)
            thresh_mask = uncertainty1.ge(thresh1).bool()*(target != 2).bool()
            target[thresh_mask] = 2
            unsup_loss = F.cross_entropy(outputs_u, target, ignore_index=2)
            unsup_loss = consistency_weight *unsup_loss*5
            
             #############################################
            
            loss = supervised_loss+unsup_loss+high_confidence_loss+weak_supervised_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            if (iter_num%500==0 and iter_num<=2500):
                max_probs_thresholds=max_probs_thresholds-0.01

    
            writer.add_scalar('uncertainty/mean', uncertainty[0,0].mean(), iter_num)
            writer.add_scalar('uncertainty/max', uncertainty[0,0].max(), iter_num)
            writer.add_scalar('uncertainty/min', uncertainty[0,0].min(), iter_num)
            writer.add_scalar('uncertainty/mask_per', torch.sum(mask)/mask.numel(), iter_num)
            writer.add_scalar('uncertainty/threshold', threshold, iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

            logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f,max_probs_thresholds:%f' %
                         (iter_num, loss.item(), consistency_dist.item(), consistency_weight,max_probs_thresholds))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                image = uncertainty[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/uncertainty', grid_image, iter_num)

                mask2 = (uncertainty > threshold).float()
                image = mask2[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/mask', grid_image, iter_num)
                #####
                image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label', grid_image, iter_num)

                image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
