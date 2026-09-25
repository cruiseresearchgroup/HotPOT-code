"""Pre-train HotPOT with the paper's two-GPU protocol."""
import sys
from .models import build_model, VARIANTS


def main():
    import argparse
    from ._internal.encoder import train_ddp as trainer
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--scale', type=int, choices=(17, 90), default=17)
    parser.add_argument('--variant', choices=VARIANTS, default='full')
    selected, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    trainer.MODEL_CLASS = lambda **kw: build_model(selected.scale, selected.variant, **kw)
    trainer.MODEL_SYSTEM = f'hotpot.{selected.variant}.{selected.scale}'
    trainer.CHECKPOINT_SCHEMA = 'hotpot.pretraining.v1'
    trainer.main()


if __name__ == '__main__':
    main()
