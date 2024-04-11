import json
import logging
from tqdm import tqdm
from typing import Optional, Dict
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
import transformers
from transformers import set_seed
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = -100

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default="qihoo360/360Zhinao-7B-Base",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    is_concat: bool = field(
        default=False, metadata={"help": "If True, training data will be concat"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    seed: int = field(default=1024)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length,
        system: str = "You are a helpful assistant.",
    ):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.system = system

        self.im_start_id = [self.tokenizer.im_start_id]
        self.im_end_id = [self.tokenizer.im_end_id]
        self.br_id = self.tokenizer.encode('\n')

        for ex in self.data[:5]:
            self.preprocessing(ex, debug=True)

    def __len__(self):
        return len(self.data)
    
    def _tokenize_system_and_user(self, role, content):
        inp_ids = self.im_start_id + self.tokenizer.encode(role) + self.br_id + self.tokenizer.encode(content)  + self.im_end_id + self.br_id
        tgt_ids = self.im_start_id + [IGNORE_TOKEN_ID] * (len(inp_ids)-3) + self.im_end_id + self.br_id
        return inp_ids, tgt_ids

    def _tokenize_assistant(self, role, content):
        inp_ids = self.im_start_id + self.tokenizer.encode(role) + self.br_id + self.tokenizer.encode(content)  + self.im_end_id + self.br_id
        tgt_ids = self.im_start_id + [IGNORE_TOKEN_ID] * len(self.tokenizer.encode(role) + self.br_id) + self.tokenizer.encode(content) + self.im_end_id + self.br_id
        return inp_ids, tgt_ids
    
    def _pad(self, input_ids, targets):
        input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))
        targets += [IGNORE_TOKEN_ID] * (self.model_max_length - len(targets))
        input_ids = input_ids[:self.model_max_length]
        targets = targets[:self.model_max_length]
        return input_ids, targets
    
    def preprocessing(self, example, debug=False):
        input_ids, labels = [], []

        ## system
        system_message = self.system
        if example["conversations"][0]["from"] == "system":
            system_message = example["conversations"][0]["value"]
            example["conversations"] = example["conversations"][1:]
        system_input_ids, system_labels = self._tokenize_system_and_user("system", system_message)
        input_ids += system_input_ids
        labels += system_labels
        assert len(input_ids) == len(labels), "Error: The length of input_ids and labels must be equal!"

        ## conversations
        for message in example["conversations"]:
            role, value = message["from"], message["value"]
            tokenize_cls = self._tokenize_system_and_user if role == "user" else self._tokenize_assistant
            msg_input_ids, msg_labels = tokenize_cls(role, value)
            input_ids += msg_input_ids
            labels += msg_labels
        assert len(input_ids) == len(labels), "Error: The length of input_ids and labels must be equal!"

        if debug:
            logger.info(f"=======================\ninput:\n{self.tokenizer.decode(input_ids)}\nlabels:\n{self.tokenizer.decode([idx for idx in labels if idx != IGNORE_TOKEN_ID])}========================\n")

        ## padding
        input_ids, labels = self._pad(input_ids, labels) 

        ## tensor
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])
    

class SupervisedDatasetConcat(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length,
        system: str = "You are a helpful assistant.",
    ):
        super(SupervisedDatasetConcat, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.system = system

        self.im_start_id = [self.tokenizer.im_start_id]
        self.im_end_id = [self.tokenizer.im_end_id]
        self.br_id = self.tokenizer.encode('\n')

        logger.info("================ before ================")
        data_dict = self.preprocessing(self.data)
        logger.info("================ end ================")
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)
    
    def _tokenize_system_and_user(self, role, content):
        inp_ids = self.im_start_id + self.tokenizer.encode(role) + self.br_id + self.tokenizer.encode(content)  + self.im_end_id + self.br_id
        tgt_ids = self.im_start_id + [IGNORE_TOKEN_ID] * (len(inp_ids)-3) + self.im_end_id + self.br_id
        return inp_ids, tgt_ids

    def _tokenize_assistant(self, role, content):
        inp_ids = self.im_start_id + self.tokenizer.encode(role) + self.br_id + self.tokenizer.encode(content)  + self.im_end_id + self.br_id
        tgt_ids = self.im_start_id + [IGNORE_TOKEN_ID] * len(self.tokenizer.encode(role) + self.br_id) + self.tokenizer.encode(content) + self.im_end_id + self.br_id
        return inp_ids, tgt_ids
    
    def _pad(self, input_ids, targets):
        input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))
        targets += [IGNORE_TOKEN_ID] * (self.model_max_length - len(targets))
        input_ids = input_ids[:self.model_max_length]
        targets = targets[:self.model_max_length]
        return input_ids, targets
    
    def preprocessing(self, examples):
        input_ids, targets = [], []
        input_ids_merge, targets_merge = [], []
        for i in tqdm(range(len(examples))):
            example = examples[i]
            single_input_ids, single_targets = [], []
            
            ## system
            system_message = self.system
            if example["conversations"][0]["from"] == "system":
                system_message = example["conversations"][0]["value"]
                example["conversations"] = example["conversations"][1:]
            system_input_ids, system_labels = self._tokenize_system_and_user("system", system_message)
            single_input_ids += system_input_ids
            single_targets += system_labels

            assert len(single_input_ids) == len(single_targets)

            ## conversations
            for message in example["conversations"]:
                role, value = message["from"], message["value"]
                tokenize_cls = self._tokenize_assistant if role == "assistant" else self._tokenize_system_and_user
                msg_input_ids, msg_labels = tokenize_cls(role, value)
                single_input_ids += msg_input_ids
                single_targets += msg_labels
            
            assert len(single_input_ids) == len(single_targets)

            if i % 10000 == 0:
                logger.info(f"input_ids: {len(input_ids)}, targets: {len(targets)}")
                logger.info(f"=======================\ninput:\n{self.tokenizer.decode(single_input_ids)}\n")
                logger.info(f"=======================\nlabels:\n{self.tokenizer.decode([idx for idx in single_targets if idx != IGNORE_TOKEN_ID])}\n")

            if len(single_input_ids) > self.model_max_length:
                continue

            if len(input_ids_merge) + len(single_input_ids) > self.model_max_length:
                input_ids_merge, targets_merge = self._pad(input_ids_merge, targets_merge) ## padding
                input_ids.append(input_ids_merge)
                targets.append(targets_merge)
                input_ids_merge, targets_merge = [], []
            
            ## concat
            input_ids_merge += single_input_ids
            targets_merge += single_targets
            
        if input_ids_merge:
            input_ids_merge, targets_merge = self._pad(input_ids_merge, targets_merge) ## padding
            input_ids.append(input_ids_merge)
            targets.append(targets_merge)
            input_ids_merge, targets_merge = [], []

        input_ids = torch.LongTensor(input_ids)
        targets = torch.LongTensor(targets)

        return {
            "input_ids": input_ids,
            "labels": targets,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[idx],
            labels=self.labels[idx],
            attention_mask=self.attention_mask[idx],
        )
           

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )


    if data_args.is_concat:
        dataset = SupervisedDatasetConcat(
            data_args.data_path, tokenizer, training_args.model_max_length
        )
    else:
        dataset = SupervisedDataset(
            data_args.data_path, tokenizer, training_args.model_max_length
        )

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()