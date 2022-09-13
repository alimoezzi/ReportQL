# Networks
import torch
import numpy as np
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
import math

# Optimizer
from transformers.optimization import Adafactor

# Memory Management
from copy import deepcopy

# Configuration
from config import configs

# Dataset
from torch.utils.data import random_split
from reportql_datasets.report_to_reportql_dataset import get_dataset
from torch.utils.data import DataLoader

# Metrics
import evaluate
from torchmetrics.functional import bleu_score


class T5FineTuner(pl.LightningModule):
    def __init__(
            self,
            weight_decay,
            learning_rate,
            adam_epsilon,
            finetune_batch_size,
            n_gpu,
            gradient_accumulation_steps,
            num_train_epochs,
            warmup_steps,
            max_seq_length,
            num_beams,
            eval_batch_size,
            **kwargs
    ):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters()
        # torch.nn.modules.sparse.Embedding.orig__init__ = torch.nn.modules.sparse.Embedding.__init__
        #
        # def bnb_embed_init(self, *args, **kwargs):
        #     torch.nn.modules.sparse.Embedding.orig__init__(self, *args, **kwargs)
        #     GlobalOptimManager.get_instance().register_module_override(self, 'weight', {'optim_bits': 32})
        #
        # torch.nn.modules.sparse.Embedding.__init__ = bnb_embed_init
        self.model = T5ForConditionalGeneration.from_pretrained(
            configs['model_name_or_path'],
            num_beams=configs['num_beams'],
            max_length=configs['max_seq_length'],
            min_length=configs['min_length'],
        )
        self.tokenizer = T5Tokenizer.from_pretrained(configs['tokenizer_name_or_path'])
        self.add_special_tokens()
        dataset = get_dataset(tokenizer=self.tokenizer, type_path="training", args=self.hparams, task=configs['task'])
        dataset_length = len(dataset)
        self.train_dataset, self.val_dataset = random_split(
            dataset, [math.floor(dataset_length*0.8), math.ceil(dataset_length*0.2)]
        )

    def is_logger(self):
        return True

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"].clone().detach()
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        self.log('train_loss', loss, on_step=True)
        return loss

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('avg_train_loss', avg_train_loss, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('valid_loss', loss, on_step=True, sync_dist=True)
        self.log('hp_metric', loss)
        return {
            "valid_loss": loss,
        }

    def validation_epoch_end(self, outputs):
        avg_validation_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        self.log('avg_validation_loss', avg_validation_loss, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, relative_step=False)
        total_steps = (
                (len(self.train_dataset)//(self.hparams.finetune_batch_size*max(1, self.hparams.n_gpu)))
                //self.hparams.gradient_accumulation_steps
                *float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def add_special_tokens(self):
        # new special tokens
        special_tokens_dict = self.tokenizer.special_tokens_map
        special_tokens_dict['mask_token'] = '<mask>'
        special_tokens_dict['additional_special_tokens'] = ['<t>', '</t>', '<a>', '</a>']
        self.tokenizer.add_tokens(['{', '}', '<c>', '</c>', '<size>'])
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset,
                                batch_size=self.hparams.finetune_batch_size if self.task=='finetune' else self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=configs['num_worker'])
        return dataloader

    def _generate_step(self, batch):
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            num_beams=self.hparams.num_beams,
            max_length=self.hparams.max_seq_length,
            # repetition_penalty=2.5,
            repetition_penalty=1.0,
            early_stopping=True
        )

        preds = self.tokenizer.batch_decode(generated_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        target = self.tokenizer.batch_decode(batch["target_ids"], clean_up_tokenization_spaces=True, skip_special_tokens=True)
        return preds, target

    def test_epoch_end(self, outputs):
        avg_loss = np.stack([x["test_loss"] for x in outputs]).mean()
        sacrebleu = evaluate.load("sacrebleu")
        rouge = evaluate.load('rouge')
        rouge_score = rouge.compute(predictions=np.stack([x["preds"] for x in outputs]), references=np.stack([x["target"] for x in outputs]))
        sacrebleu_score = sacrebleu.compute(predictions=np.stack([x["preds"] for x in outputs]), references=np.stack([x["target"] for x in outputs]))
        accuracy = bleu_score(np.stack([x["preds"] for x in outputs]), np.stack([x["target"] for x in outputs]))
        self.log('test_loss', avg_loss, on_epoch=True, sync_dist=True)
        self.log('test_acc', accuracy, on_epoch=True, sync_dist=True)
        self.log('rouge1', rouge_score['rouge1'], on_epoch=True, sync_dist=True)
        self.log('rouge2', rouge_score['rouge2'], on_epoch=True, sync_dist=True)
        self.log('rougeL', rouge_score['rougeL'], on_epoch=True, sync_dist=True)
        self.log('rougeLsum', rouge_score['rougeLsum'], on_epoch=True, sync_dist=True)
        self.log('sacrebleu_score', sacrebleu_score['score'], on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        preds, target = self._generate_step(batch)
        print(preds,  target)
        loss = self._step(batch)
        return {
            "test_loss": loss.cpu().numpy(),
            "preds": preds[0],
            "target": target[0]
        }

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=configs['num_worker'])

    def test_dataloader(self):
        test_dataset = get_dataset(tokenizer=self.tokenizer, type_path="test", args=self.hparams)
        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=configs['num_worker'])
