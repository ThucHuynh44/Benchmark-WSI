# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser, BooleanOptionalAction


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    from datasets import NAMES as dataset_names
    from models import get_all_models

    parser.add_argument('--dataset', type=str, required=True,
                        choices=dataset_names,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--exp_desc', type=str, required=True,
                        help='Experiment description.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate.')

    parser.add_argument('--optimizer', type=str, choices=('adamw',), default='adamw',
                        help='Shared optimizer used by every continual method.')

    parser.add_argument('--adam_eps', type=float, default=1.0e-8,
                        help='Numerical epsilon used by the shared AdamW optimizer.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')    

    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Maximum training epochs per task.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size.')

    parser.add_argument(
        '--early_stopping', action=BooleanOptionalAction, default=True,
        help=(
            'Enable validation-loss early stopping. Disabling it still restores '
            'the best validation checkpoint after all epochs.'
        ),
    )
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Non-improving validation epochs tolerated before stopping.')
    parser.add_argument('--early_stopping_min_epoch', type=int, default=1,
                        help='Minimum number of completed epochs before stopping is allowed.')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0,
                        help='Minimum validation-loss decrease counted as improvement.')
    parser.add_argument(
        '--early_stopping_verbose', action=BooleanOptionalAction, default=True,
        help='Print validation checkpoint and early-stopping updates.',
    )
    parser.add_argument(
        '--evaluate_fwt', action=BooleanOptionalAction, default=False,
        help=(
            'Evaluate untrained/future tasks to compute forward transfer. '
            'Disabled by default so test evaluation happens only after training.'
        ),
    )

    parser.add_argument('--backbone', type=str, default='generic_mil',
                        help="MIL backbone: generic_mil, titan, feather, or '<module>:<ClassName>'.")
    parser.add_argument('--feature_dim', type=int, default=768,
                        help='Patch feature dimension stored in the HDF5 file.')
    parser.add_argument('--backbone_hidden_dim', type=int, default=384,
                        help='Hidden dimension used by the default generic MIL backbone.')
    parser.add_argument('--backbone_dropout', type=float, default=0.0,
                        help='Dropout used by the default generic MIL backbone.')
    parser.add_argument('--backbone_kwargs', type=str, default=None,
                        help='JSON object with extra constructor arguments for a custom backbone.')
    parser.add_argument('--backbone_freeze', action='store_true',
                        help='Freeze the pretrained slide encoder and train only its classifier.')
    parser.add_argument('--backbone_max_patches', type=int, default=None,
                        help='Maximum patches per bag; 0/None means the full bag.')
    parser.add_argument('--backbone_cache_dir', type=str, default=None,
                        help='Optional Hugging Face cache directory for native pretrained backbones.')
    parser.add_argument('--backbone_allow_download', action='store_true',
                        help='Allow downloading a missing pinned backbone snapshot.')
    parser.add_argument('--backbone_model_id', type=str, default=None,
                        help='Pinned Hugging Face model ID (normally supplied by the backbone profile).')
    parser.add_argument('--backbone_revision', type=str, default=None,
                        help='Pinned Hugging Face revision (normally supplied by the backbone profile).')
    parser.add_argument('--patch_size_level0_fallback', type=int, default=1024,
                        help='Patch size used only when coords.attrs lacks patch_size_level0.')
    parser.add_argument('--dataset_config', type=str, default=None,
                        help='WSI dataset YAML (defaults to configs/datasets.yaml).')
    parser.add_argument('--reverse_task_order', action='store_true',
                        help='Reverse task_order after reading the YAML file.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader worker processes.')

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--config', type=str, default=None,
                        help='YAML experiment config; explicit CLI arguments override it.')
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')
    parser.add_argument('--ablation_id', type=str, default=None,
                        help='Stable ablation variant identifier written to manifests.')
    parser.add_argument('--ablation_group', type=str, default=None,
                        help='Ablation family used by comparison reports.')
    parser.add_argument('--ablation_config_hash', type=str, default=None,
                        help='Stable hash of the resolved ablation registry entry.')

    parser.add_argument('--non_verbose', action='store_true')
    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging', default=True)
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--folds', type=str, default='all',
                        help="Folds to run: 'all', one ID, comma list, or ranges (0,2,4-6).")
    parser.add_argument('--preflight_only', action='store_true',
                        help='Validate selected folds and exit without creating a model.')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True, default=100,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')
