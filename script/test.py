from config import configs
import torch
from pathlib import Path
import pytorch_lightning as pl
from logger import LoggingCallback
from trainer import T5FineTuner
from utils import set_seed
import gc

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=configs['output_dir'],
    filename='{epoch}-{valid_loss:.2f}',
    monitor="valid_loss",
    mode="min",
    save_top_k=2,
    auto_insert_metric_name=False,
    verbose=True
)

train_params = dict(
    accumulate_grad_batches=configs['gradient_accumulation_steps'],
    gpus=configs['n_gpu'],
    accelerator="gpu",
    tpu_cores=0,
    max_epochs=configs['num_train_epochs'],
    # early_stop_callback=False,
    precision=32,
    # amp_level=args.opt_level,
    gradient_clip_val=configs['max_grad_norm'],
    enable_checkpointing=True,
    callbacks=[LoggingCallback(), checkpoint_callback],
)

if __name__ == '__main__':
    set_seed(configs['seed'])
    model = T5FineTuner(
        weight_decay=configs['weight_decay'],
        learning_rate=configs['learning_rate'],
        adam_epsilon=configs['adam_epsilon'],
        finetune_batch_size=configs['finetune_batch_size'],
        n_gpu=configs['n_gpu'],
        gradient_accumulation_steps=configs['gradient_accumulation_steps'],
        num_train_epochs=configs['num_train_epochs'],
        warmup_steps=configs['warmup_steps'],
        max_seq_length=configs['max_seq_length'],
        num_beams=configs['num_beams'],
        eval_batch_size=configs['eval_batch_size'],
    )
    model.eval()
    trainer = pl.Trainer(**train_params)

    trainer.test(model=model, ckpt_path=Path(configs["output_dir"])/Path(configs["checkpoint"]))
