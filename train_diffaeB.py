from utils.args import make_args
from utils.trainer import DiffFSTrainer


def main(args):
    trainer = DiffFSTrainer(args)
    trainer.train()


if __name__ == '__main__':
    args = make_args()
    main(args)