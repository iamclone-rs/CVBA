import inspect
import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model_LN_prompt import Model
from src.dataset_retrieval import Sketchy
from experiments.options import opts


def _available_cpu_workers():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _effective_num_workers(requested_workers):
    max_workers = max(1, _available_cpu_workers())
    if requested_workers > max_workers:
        print('clamping dataloader workers from %d to %d' % (requested_workers, max_workers))
    return min(requested_workers, max_workers)


def _build_trainer_kwargs(logger, checkpoint_callback, ckpt_path):
    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = dict(
        min_epochs=1,
        max_epochs=2000,
        benchmark=True,
        logger=logger,
        check_val_every_n_epoch=5,
        callbacks=[checkpoint_callback]
    )

    if torch.cuda.is_available():
        if 'gpus' in trainer_init_params:
            trainer_kwargs['gpus'] = -1
        else:
            trainer_kwargs['accelerator'] = 'gpu'
            trainer_kwargs['devices'] = -1
    else:
        if 'gpus' in trainer_init_params:
            trainer_kwargs['gpus'] = 0
        else:
            trainer_kwargs['accelerator'] = 'cpu'
            trainer_kwargs['devices'] = 1

    if ckpt_path is not None and 'resume_from_checkpoint' in trainer_init_params:
        trainer_kwargs['resume_from_checkpoint'] = ckpt_path

    return trainer_kwargs

if __name__ == '__main__':
    dataset_transforms = Sketchy.data_transform(opts)

    train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    val_dataset = Sketchy(opts, dataset_transforms, mode='val', used_cat=train_dataset.all_categories, return_orig=False)

    num_workers = _effective_num_workers(opts.workers)

    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size, num_workers=num_workers)

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='saved_models/%s'%opts.exp_name,
        filename="{epoch:02d}-{val_loss:.4f}",
        mode='min',
        save_last=True)

    ckpt_path = os.path.join('saved_models', opts.exp_name, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    trainer = Trainer(**_build_trainer_kwargs(logger, checkpoint_callback, ckpt_path))

    model = Model()

    print ('beginning training...good luck...')
    fit_params = inspect.signature(Trainer.fit).parameters
    fit_kwargs = {}
    if ckpt_path is not None and 'ckpt_path' in fit_params:
        fit_kwargs['ckpt_path'] = ckpt_path
    trainer.fit(model, train_loader, val_loader, **fit_kwargs)
