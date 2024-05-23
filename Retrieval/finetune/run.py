import deepspeed
import logging
import os
import shutil
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

import sys
from arguments import ModelArguments, DataArguments
from arguments import RetrieverTrainingArguments as TrainingArguments
from data import TrainDatasetForEmbedding, EmbedCollator
from modeling import TextEncoderModel
from mytrainer import MyTrainer

import wandb
import datetime
import torch

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set wandb
    if training_args.local_rank in [-1, 0]:
        use_wandb = False
        if training_args.wandb_host and training_args.wandb_host != "-1":
            use_wandb = True

        if use_wandb:
            logger.info("use_wandb=true")
            run_version = training_args.output_dir.split('/')[-1]
            run_version += "-" + str(datetime.date.today())
            wandb.login(host=training_args.wandb_host, key=training_args.wandb_key)
            wandb.init(name=run_version, project=training_args.wandb_project_name, config=training_args)
            wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".sh"))
        else:
            logger.info("use_wandb=false")
            os.environ['WANDB_MODE'] = 'dryrun'

    # Set seed
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    logger.info('Config: %s', config)

    model = TextEncoderModel(model_name_or_path=model_args.model_name_or_path,
                             config=config,
                             normlized=training_args.normlized,
                             negatives_cross_device=training_args.negatives_cross_device,
                             temperature=training_args.temperature)

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EmbedCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len
        ),
        tokenizer=tokenizer,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    final_model_path = training_args.output_dir + '/model/final_step_model'
    trainer.save_model(final_model_path)

    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()



