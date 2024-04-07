from templates import *
from templates_latent import *

if __name__ == '__main__':
    # gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    conf = ffhq256_autoenc()

    conf.data_name = 'afhq256-dog'
    conf.name = 'afhq256-dog_autoenc_old'
    conf.total_samples = 90_000_000
    conf.sample_every_samples = 500_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.eval_num_images = 1000

    # train(conf, gpus=gpus)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    # gpus = [3]
    # conf.eval_programs = ['infer']
    # train(conf, gpus=gpus, mode='eval')

    # # train the latent DPM
    # # NOTE: only need a single gpu
    gpus = [3]
    conf = ffhq256_autoenc_latent()
    conf.data_name = 'afhq256-dog'
    conf.name = 'afhq256-dog_autoenc_latent_old'
    conf.pretrain = PretrainConfig(
        name='90M',
        path=f'checkpoints/afhq256-dog_autoenc_old/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/afhq256-dog_autoenc_old/latent.pkl'
    conf.total_samples = 40_000_000
    conf.sample_every_samples = 5_000_000    
    train(conf, gpus=gpus)