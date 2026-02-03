import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import json
import numpy as np
import random

from diffusers.models import AutoencoderKL
from seed import setup_seed, no_seed, setup_seed_cpu, setup_seed_gpu


def eval():
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_test.yml',#'config/config_test.yml', config_C60_U45
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    parser.add_argument('--latent_space', type=bool, default=False)
    parser.add_argument('--vae_path', type=str, default='./vae/fine-tuning/ema')
    parser.add_argument('--opt_path', type=str, default='./config/data_LSUI.json')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_seed(42, False)
    with open(args.opt_path, 'r') as f:
        text_opt = json.loads(f.read())

    opt = Logger.dict_to_nonedict(Logger.parse(args))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
    wandb_logger = WandbLogger(opt) if opt['enable_wandb'] else None

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    diffusion = Model.create_model(opt, text_opt)
    logger.info('Initial Model Finished')
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')

    if args.latent_space:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)

    current_step = diffusion.begin_step  # current_step = 150000
    train_seeds = diffusion.train_seeds
    val_seeds = diffusion.val_seeds

    logger.info('Begin Model Inference.')
    avg_psnr, avg_ssim = 0, 0
    # current_step = 0
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    # consistent_rand = False
    # no_seed()
    # setup_seed_cpu(14265830959700210199)
    # setup_seed_gpu(1056465172455132)
    # np.random.seed(2147483648)
    setup_seed_cpu(val_seeds[-1]['cpu'])
    setup_seed_gpu(val_seeds[-1]['gpu'])
    sd_cuda = torch.cuda.initial_seed()
    sd = torch.initial_seed()
    # np.random.seed(train_seeds[-1]['numpy'])
    random.setstate(val_seeds[-1]['random_state'])
    np.random.set_state(val_seeds[-1]['np_state'])
    sd_np = np.random.get_state()[1][0]
    sd_randm = random.getstate()[1][0]
    seed = val_seeds[-1]['numpy']
    np.random.seed(seed)
    # state_random = random.getstate()
    logging.info(
        '#{}th {} seed : cpu: {}; gpu: {}; numpy: {} ; randm: {}'.format(current_step, 'train', sd, sd_cuda, sd_np,
                                                                         sd_randm))
    eval_diff = diffusion
    eval_diff.load_network()
    i = 0
    for _, val_data in enumerate(val_loader):
        i = i + 1
        #if i < 248:
        #    continue
        cond_inp = val_data['input'].to(device)
        cond_tar = val_data['target'].to(device)
        if args.latent_space:
            # Map input images to latent space + normalize latents:
            val_data['input'] = vae.encode(cond_inp).latent_dist.sample().mul_(0.18215)
            val_data['target'] = vae.encode(cond_tar).latent_dist.sample().mul_(0.18215)
        idx += 1
        eval_diff.feed_data(val_data,'input', 'val')
        eval_diff.test(continous=False, use_res=False, single=opt['model']['single_step'])
        visuals = eval_diff.get_current_visuals()

        if args.latent_space:
            target_img = val_data['target']
            input_img = val_data['input']
            restore_img = visuals['output']

            restore_img = vae.decode(restore_img / 0.18215).sample
            # target_img = vae.decode(target_img / 0.18215).sample
            input_img = vae.decode(input_img / 0.18215).sample

            restore_img = restore_img.detach().float().cpu()
            # target_img = target_img.detach().float().cpu()
            input_img = input_img.detach().float().cpu()
            depth_img = visuals['depth'].detach().float().cpu()

            depth_orig = val_data['depth']

        else:
            restore_img = visuals['output'].detach().float().cpu()
            depth_img = visuals['depth_norm'].detach().float().cpu()

            depth_orig = val_data['depth']
            target_img = val_data['target']
            input_img = val_data['input']

        restore_img = Metrics.tensor2img(restore_img)  # uint8
        depth_img = Metrics.tensor2img(depth_img, min_max=(0, 1))  # uint8
        depth_orig = Metrics.tensor2img(depth_orig, min_max=(-1, 1))
        target_img = Metrics.tensor2img(target_img)  # uint8
        input_img = Metrics.tensor2img(input_img)  # uint8

        img_name = val_data['name'][0].split('.')[0]
        # generation
        # Metrics.save_img(target_img, '{}/{}_{}_target.png'.format(result_path, current_step, idx))
        Metrics.save_img(restore_img, '{}/{}'.format(result_path, val_data['name'][0]))
        Metrics.save_img(input_img, '{}/{}_input.png'.format(result_path, img_name))
        # Metrics.save_img(depth_orig, '{}/{}_{}_depth_orig.png'.format(result_path, current_step, idx))
        Metrics.save_img(depth_img, '{}/{}_depth_recon.png'.format(result_path, img_name))

        avg_psnr += Metrics.calculate_psnr(restore_img, target_img)
        avg_ssim += Metrics.calculate_ssim(restore_img, target_img)
        logger.info('# Img.{} Validation # PSNR: {:.4e}; SSIM: {:.4e}'.format(idx, avg_psnr / idx, avg_ssim / idx))

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(restore_img, Metrics.tensor2img(visuals['input'][-1]), target_img)

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)


if __name__ == "__main__":
    eval()
