from templates import *
from templates_latent import *

if __name__ == '__main__':
    # gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    conf = ffhq256_autoenc()

    conf.data_name = 'church256'
    conf.name = 'church256_autoenc'
    conf.total_samples = 90_000_000
    conf.sample_every_samples = 500_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000

    # train(conf, gpus=gpus)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    # gpus = [2]
    # conf.eval_programs = ['infer']
    # train(conf, gpus=gpus, mode='eval')

    # # train the latent DPM
    # # NOTE: only need a single gpu
    gpus = [2]
    conf = ffhq256_autoenc_latent()
    conf.data_name = 'church256'
    conf.name = 'church256_autoenc_latent'
    conf.pretrain = PretrainConfig(
        name='90M',
        path=f'checkpoints/church256_autoenc/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/church256_autoenc/latent.pkl'
    conf.total_samples = 100_000_000
    conf.sample_every_samples = 5_000_000
    train(conf, gpus=gpus)