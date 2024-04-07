from utils.args import make_args
from utils.tester import DiffFSTester


def main(args):
    tester = DiffFSTester(args)
    tester.infer_image_all()


if __name__=='__main__':
    args = make_args()
    main(args)