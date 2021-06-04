import logging
import attrdict
import math
from typing import Dict, Any

import torch

from transformers import (OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizerFast,
                                  GPT2DoubleHeadsModel, GPT2TokenizerFast)

import determined.pytorch as det_torch
import model_hub.huggingface as hf
from data import add_special_tokens_, build_dataset


class ConvoAITrial(hf.BaseTransformerTrial):
    def __init__(self, context: det_torch.PyTorchTrialContext) -> None:
        self.logger = logging.getLogger(__name__)
        self.hparams = attrdict.AttrDict(context.get_hparams())
        self.data_config = attrdict.AttrDict(context.get_data_config())
        self.exp_config = attrdict.AttrDict(context.get_experiment_config())
        self.context = context

        # Parse optimizer and scheduler config
        optimizer_kwargs, scheduler_kwargs = hf.default_parse_optimizer_lr_scheduler_kwargs(
            self.hparams
        )
        tokenizer_class = GPT2TokenizerFast if "gpt2" in self.hparams.model_checkpoint else OpenAIGPTTokenizerFast
        self.tokenizer = tokenizer_class.from_pretrained(self.hparams.model_checkpoint)


        model_class = GPT2DoubleHeadsModel if "gpt2" in self.hparams.model_checkpoint else OpenAIGPTDoubleHeadsModel
        model = model_class.from_pretrained(self.hparams.model_checkpoint)
        self.model = self.context.wrap_model(model)
        self.nll = torch.nn.CrossEntropyLoss(ignore_index=-100).cuda()

        # Add special tokens if they are not already added
        add_special_tokens_(self.model, self.tokenizer)

        # Build train and val datasets
        self.train_dataset, self.val_dataset = build_dataset(self.data_config, self.tokenizer)

        train_length = len(self.train_dataset)
        self.logger.info("training records: {}".format(train_length))
        if (
            "records_per_epoch" in self.exp_config
            and train_length != self.exp_config["records_per_epoch"]
        ):
            self.logger.warning(
                "number of train records {} does not match records_per_epoch of {}".format(
                    train_length, self.exp_config["records_per_epoch"]
                )
            )

        # Build optimizer and lr scheduler
        self.optimizer = self.context.wrap_optimizer(
            hf.build_default_optimizer(self.model, optimizer_kwargs)
        )

        if self.hparams.use_apex_amp:
            self.model, self.optimizer = self.context.configure_apex_amp(
                models=self.model,
                optimizers=self.optimizer,
            )

        self.lr_scheduler = self.context.wrap_lr_scheduler(
            hf.build_default_lr_scheduler(self.optimizer, scheduler_kwargs),
            det_torch.LRScheduler.StepMode.STEP_EVERY_BATCH,
        )
        self.grad_clip_fn = (
            lambda x: torch.nn.utils.clip_grad_norm_(x, optimizer_kwargs.max_grad_norm)
            if optimizer_kwargs.max_grad_norm > 0  # type: ignore
            else None
        )

    def build_training_data_loader(self) -> det_torch.DataLoader:
        print('train len:', len(self.train_dataset))
        return det_torch.DataLoader(
            self.train_dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            num_workers=8
        )

    def build_validation_data_loader(self) -> det_torch.DataLoader:
        print('val len:', len(self.val_dataset))
        return det_torch.DataLoader(
            self.val_dataset,
            batch_size=self.context.get_per_slot_batch_size(),
        )

    def evaluate_batch(self, batch: det_torch.TorchData, batch_idx: int) -> Dict:
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        # if we dont send labels to model, it doesnt return losses
        outputs = self.model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
        )
        lm_logits = outputs.logits
        mc_logits = outputs.mc_logits
        lm_logits_flat_shifted = lm_logits[..., :-1, :].reshape(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = lm_labels[..., 1:].reshape(-1)
        nll = self.nll(lm_logits_flat_shifted, lm_labels_flat_shifted)
        preds = torch.argmax(mc_logits, dim=-1)
        accuracy = (preds == mc_labels).to(torch.float).mean()
        ppl = math.exp(nll)
        return {'nll': nll, 'accuracy': accuracy, 'ppl': ppl}

    def train_batch(self, batch: Any, epoch_idx: int, batch_idx: int) -> Any:
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        outputs = self.model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            mc_labels=mc_labels, labels=lm_labels
        )
        lm_loss = outputs.loss
        mc_loss = outputs.mc_loss

        loss = (lm_loss * self.hparams.lm_coef + mc_loss * self.hparams.mc_coef) 
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer, self.grad_clip_fn)
        return loss
