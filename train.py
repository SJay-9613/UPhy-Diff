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
import numpy as np
import json
import random

from diffusers.models import AutoencoderKL
from seed import setup_seed, no_seed, setup_seed_cpu, setup_seed_gpu, setup_seed_not_torch


def log_seed(seeds, current_step, stage='train', state: dict = None):
    sd_cuda = torch.cuda.initial_seed()
    sd = torch.initial_seed()
    if state is None:
        sd_np = np.random.get_state()[1][0]
        sd_randm = random.getstate()[1][0]
        state_random = random.getstate()
        state_np = np.random.get_state()
    else:
        state_np = state['numpy']
        state_random = state['random']
        sd_np = state_np[1][0]
        sd_randm = state_random[1][0]
    seeds.append({'gpu': sd_cuda, 'cpu': sd, 'numpy': sd_np, 'random': sd_randm, 'random_state': state_random, 'np_state': state_np})
    logging.info(
        '#{}th {} seed : cpu: {}; gpu: {}; numpy: {} ; randm: {}'.format(current_step, stage, sd, sd_cuda, sd_np,
                                                                    sd_randm))
    # print('cpu: {}; gpu: {} ; numpy: {} ; os: {}; randm: {}'.format(sd, sd_cuda, sd_np, os.environ['PYTHONHASHSEED'],
    #                                                                 sd_randm))


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=r'./config/config.yml',
                        help='yml file for configuration')
    parser.add_argument('-p', '--phase', type=str, help='Run train(training)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    parser.add_argument('--latent_space', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vae_path', type=str, default='./vae/fine-tuning/ema/6')
    # parser.add_argument('--vae_path', type=str, default='./fusing/autoencoder-kl-dummy')
    parser.add_argument('--opt_path', type=str, default='./config/data.json')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parse configs
    args = parser.parse_args()
    with open(args.opt_path, 'r') as f:
        text_opt = json.loads(f.read())
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    # 此处设置随机种子是为了保证数据集载入时的随机打乱是一致的
    # 25-2-8
    # setup_seed(args.seed, is_torch=False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            # train_set = Data.create_dataset(dataset_opt, phase)
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            # val_set = Data.create_dataset(dataset_opt, phase)
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt, text_opt)
    logger.info('Initial Model Finished')
    args.latent_space = opt['model']['latent_space']
    if args.latent_space == True:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
        vae.eval()

    # Train
    train_seeds = diffusion.train_seeds
    val_seeds = diffusion.val_seeds

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    phase_set = {True: 'target', False: 'input'} # 每10步，选择使用GT还是input训练
    best_psnr = 19
    test_all_psnr = 24.5
    loss = 0
    buff = 0
    # 25-2-8
    # 如果训练和验证的种子数量不一致，(显然训练只会比验证多)，则继续使用验证
    # if opt['model']['finetune_norm']:
    #     val_seed = val_seeds[-1]
    #     no_seed()
    #     np_state = val_seed['np_state']
    #     np.random.set_state(np_state)
    #     log_seed(train_seeds, current_step)
    # else:
    #     if (len(train_seeds) != 0) and (len(train_seeds) == len(val_seeds) + 1):
    #         setup_seed_cpu(train_seeds[-1]['cpu'])
    #         setup_seed_gpu(train_seeds[-1]['gpu'])
    #         sd_cuda = torch.cuda.initial_seed()
    #         sd = torch.initial_seed()
    #         # np.random.seed(train_seeds[-1]['numpy'])
    #         seed = val_seeds[-1]['numpy']
    #         np.random.seed(seed)
    #         sd_np = np.random.get_state()[1][0]
    #         sd_randm = random.getstate()[1][0]
    #         random.setstate(train_seeds[-1]['random_state'])
    #         # state_random = random.getstate()
    #         logging.info(
    #             '#{}th {} seed : cpu: {}; gpu: {}; numpy: {} ; randm: {}'.format(current_step, 'train', sd, sd_cuda, sd_np,
    #                                                                              sd_randm))
    #     else:
    #         # no_seed()
    #         # sd = torch.initial_seed()
    #         # sd_cuda = torch.cuda.initial_seed()
    #         # train_seeds.append({'cpu': sd, 'gpu': sd_cuda})
    #         # logging.info('#{}th train seed : cpu: {}; gpu: {}'.format(current_step, train_seeds[-1]['cpu'], train_seeds[-1]['gpu']))
    #         no_seed()
    #         seed = args.seed
    #         log_seed(train_seeds, current_step)
    a_or_t = True
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            now_phase = True
            for _, train_data in enumerate(train_loader):
                if current_step % 100 == 0:
                    now_phase = not now_phase
                # cond_inp = train_data['input'].to(device)
                # cond_tar = train_data['target'].to(device)
                # if args.latent_space:
                #     # Map input images to latent space + normalize latents:
                #     train_data['input'] = vae.encode(cond_inp).latent_dist.sample().mul_(0.18215)
                #     train_data['target'] = vae.encode(cond_tar).latent_dist.sample().mul_(0.18215)
                current_step += 1
                if current_step > n_iter:
                    break
                if wandb_logger:
                    wandb_logger.log_metrics(len(train_data))
                if current_step % 100 == 0:
                    a_or_t = not a_or_t
                diffusion.feed_data(train_data, a_or_t, 'train')
                diffusion.optimize_parameters(signal=False, use_res=False)    # 先用文本和编码引导，然后再优化先验
                loss += diffusion.get_current_log()['l_pix']
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = {'average_l_pix': loss / opt['train']['print_freq']}
                    loss = 0
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # 保存权重
                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step, train_seeds, val_seeds)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    train_state = {'numpy': np.random.get_state(), 'random': random.getstate()}
                    avg_psnr, avg_ssim = 0.0, 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')

                    # 保存权重
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step, train_seeds, val_seeds)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

                    eval_diffusion = diffusion
                    eval_diffusion.load_network(os.path.join(opt['path']['experiments_root'],'checkpoint', f'I{current_step}_E{current_epoch}'))
                    # 测试的时候读取随机种子
                    # 25-2-8
                    # if opt['model']['finetune_norm']:
                    #     setup_seed_cpu(val_seed['cpu'])
                    #     setup_seed_gpu(val_seed['gpu'])
                    #     sd_cuda = torch.cuda.initial_seed()
                    #     sd = torch.initial_seed()
                    #     # np.random.seed(train_seeds[-1]['numpy'])
                    #     random.setstate(val_seed['random_state'])
                    #     np.random.set_state(val_seed['np_state'])
                    #     sd_np = np.random.get_state()[1][0]
                    #     sd_randm = random.getstate()[1][0]
                    #     #seed = val_seed['numpy']
                    #     #np.random.seed(seed)
                    #     # state_random = random.getstate()
                    #     logging.info(
                    #         '#{}th {} seed : cpu: {}; gpu: {}; numpy: {} ; randm: {}'.format(current_step, 'train', sd,
                    #                                                                          sd_cuda, sd_np,
                    #                                                                          sd_randm))
                    # else:
                    #     no_seed()
                    #     buff += 1
                    #     setup_seed_not_torch(seed + buff)
                    #     # sd_cuda = torch.cuda.initial_seed()
                    #     # sd = torch.initial_seed()
                    #     # val_seeds.append({'cpu': sd, 'gpu': sd_cuda})
                    #     # logging.info('#{}th val seed : cpu: {}; gpu: {} '.format(current_step, sd, sd_cuda))
                    #     log_seed(val_seeds, current_step, 'val')
                    best = 0
                    for val_data in val_loader:
                        if idx == 5:
                            if best < test_all_psnr:
                                break
                            else:
                                test_all_psnr = best
                                logger.info(
                                    '# Validation # First Five PSNR: {:.4e}'.format(best))
                        idx += 1
                        cond_inp = val_data['input'].to(device)
                        # cond_tar = val_data['target'].to(device)
                        if args.latent_space:
                            # Map input images to latent space + normalize latents:
                            val_data['input'] = vae.encode(cond_inp).latent_dist.sample().mul_(0.18215)
                            # val_data['target'] = vae.encode(cond_tar).latent_dist.sample().mul_(0.18215)
                        eval_diffusion.feed_data(val_data, 'input', 'val')
                        eval_diffusion.test(continous=False, use_res=False, single=opt['model']['single_step'])    # single控制是否使用单步预测
                        visuals = eval_diffusion.get_current_visuals()

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
                            depth_orig = val_data['depth']

                            depth_img = visuals['depth_norm'].detach().float().cpu()
                            target_img = val_data['target']
                            input_img = val_data['input']

                        restore_img = Metrics.tensor2img(restore_img)  # uint8
                        depth_img = Metrics.tensor2img(depth_img, min_max=(-1, 1))  # uint8
                        depth_orig = Metrics.tensor2img(depth_orig, min_max=(-1, 1))
                        target_img = Metrics.tensor2img(target_img)  # uint8
                        input_img = Metrics.tensor2img(input_img)  # uint8

                        # generation
                        Metrics.save_img(target_img, '{}/{}_{}_target.png'.format(result_path, current_step, idx))
                        Metrics.save_img(restore_img, '{}/{}_{}_output.png'.format(result_path, current_step, idx))
                        Metrics.save_img(input_img, '{}/{}_{}_input.png'.format(result_path, current_step, idx))
                        Metrics.save_img(depth_orig, '{}/{}_{}_depth_orig.png'.format(result_path, current_step, idx))
                        Metrics.save_img(depth_img, '{}/{}_{}_depth_recon.png'.format(result_path, current_step, idx))
                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate((input_img, restore_img), axis=1), [2, 0, 1]), idx)
                        avg_psnr += Metrics.calculate_psnr(restore_img, target_img)
                        avg_ssim += Metrics.calculate_ssim(restore_img, target_img)
                        if idx <= 5:
                            best = avg_psnr / idx

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}',
                                np.concatenate((input_img, restore_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx
                    del eval_diffusion
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # First Five PSNR: {:.4e} ; PSNR: {:.4e} ; SSIM: {:.4e}'.format(best, avg_psnr, avg_ssim))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, best))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({'validation/val_psnr': avg_psnr, 'validation/val_step': val_step})
                        val_step += 1

                    # no_seed()
                    # sd_cuda = torch.cuda.initial_seed()
                    # sd = torch.initial_seed()
                    # train_seeds.append({'cpu': sd, 'gpu': sd_cuda})
                    # logging.info('#{}th train seed : cpu: {} ; gpu: {} '.format(current_step, sd, sd_cuda))
                    # 25-2-8
                    # log_seed(train_seeds, current_step, state=train_state)

                    # 保存最好的
                    if best_psnr < best:
                        best_psnr = best
                        logger.info('Saving best models and training states.')
                        diffusion.save_best_network(current_epoch, current_step, train_seeds, val_seeds)

                        if wandb_logger and opt['log_wandb_ckpt']:
                            wandb_logger.log_checkpoint(current_epoch, current_step)



            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        raise NotImplementedError('phase should be the train phase')

