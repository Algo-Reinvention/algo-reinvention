"""
This code is modified from verl.utils.dataset.rl_dataset
"""


import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

class RetainDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.raw_prompt_key = "prompt"  #@jzhao
        self.response_key = "response"  #@jzhao
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_raw_prompt_length = config.get("max_prompt_length", 1024)
        self.max_response_length = config.get("max_response_length", 512)
        self.max_prompt_length = self.max_raw_prompt_length + self.max_response_length
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        if self.processor is not None:
            self.tokenizer = self.processor.tokenizer
            self.processor = None

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    # NOTE @jzhao: self.dataframe should be huggingface dataset
    # >> dataframe[:5] == {'id': ['0001', '0002', '0003', '0004', '0005'], 'source': ['weibo', 'weibo', ...]}
    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    # NOTE @jzhao: we need to filter prompt>max_prompt_len
    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.raw_prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                raise ValueError("not support multi-modal")
            else:
                def doc2len(doc) -> int:
                    return len(tokenizer.encode(doc[prompt_key], add_special_tokens=False))

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_raw_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_raw_prompt_length} tokens"
            )

            print(f"filter dataset len: {len(dataframe)}")

        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        raise ValueError("Not Implemented")

    # NOTE @jzhao: we need to return {"input_ids", "response_ids", "attention_mask", "position_ids", "loss_mask"}
    # >> origin parquet should be like: [["content": "<raw_sequence>", "prompt"， "response"]...]
    def __getitem__(self, item):
        """"
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        model_inputs = {}

        if self.processor is not None:
            raise NotImplementedError("Not support multi-modal") #@jzhao

        else:
            # NOTE @jzhao: the model_inputs will be 2D tensor though there's only 1 prompt
            model_inputs = self.tokenizer(row_dict['prompt'], return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            prompt_ids, prompt_attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_raw_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
            model_inputs = self.tokenizer(row_dict['response'], return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            response_ids, response_attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_response_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=False,
                truncation="right",
            )
            row_dict["prompt_ids"] = prompt_ids[0]
            row_dict["response_ids"] = response_ids[0]
            row_dict['response'] = self.tokenizer.decode(row_dict["response_ids"])
            prompt_length = prompt_ids.size(-1)
            response_length = response_ids.size(-1)

            raw_sequence = row_dict['prompt'] + row_dict['response']
            input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
            attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            raise NotImplementedError("Not support multi-modal") #@jzhao
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_sequence_ids = self.tokenizer.encode(raw_sequence, add_special_tokens=False)
        if len(raw_sequence_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_sequence_ids = raw_sequence_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_sequence_ids = raw_sequence_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_sequence_ids = raw_sequence_ids[:left_half] + raw_sequence_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_sequence_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_sequence_ids"] = raw_sequence_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_sequence"] = ""

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_sequence  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs

        loss_mask = attention_mask[0].clone()
        loss_mask = loss_mask[prompt_length:]
        # mask out the last token in response
        loss_mask[-1] = 0
        row_dict["loss_mask"] = loss_mask

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
