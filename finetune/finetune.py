# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig
from accelerate.utils import DistributedType
from colorama import Style, Fore
import re
import tqdm
from typing import Dict, Optional, Sequence, List

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    qwen_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    remove_unused_columns: bool = False # Allow custom attributes appear to the batch.


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "attn.c_proj", "w1", "w2"]  ##["in_proj","out_proj","c_fc"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

def get_point(tokenizer, sentence):
    #This function is used to extract the points from the sentence, and get the '<ref>' position in the tokenzied_ids of sentence.
    points, index = [], []
    # sentence = sentence.replace('<point>', TAG_MAP['<point>'])#.replace('<bbox>', TAG_MAP['<bbox>'])
    tokenized_id = tokenizer(sentence).input_ids
    point_id = tokenizer('<ref>').input_ids
    # box_id = tokenizer(TAG_MAP['<bbox>']).input_ids
    for x, y in re.findall(r"<ref>\s*\((\d+),\s*(\d+)\)", sentence):
        points.append([int(x), int(y)])
    if len(points) == 0:
        points.append([-100, -100])
    index = [i for i, token in enumerate(tokenized_id) if token == point_id[0]]
    # index_box = [i for i, token in enumerate(tokenized_id) if token == box_id[0]]

    return points, index

def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        points_conv, points = [], [] # This is used to store the points in the batch.
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                        tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            if role == '<|im_start|>user':
                points_in_s, index = get_point(tokenizer, sentence['value'])
                if points_in_s != [[-100, -100]] and '<ref>' in sentence['value']:
                    assert len(points_in_s) == len(index)
                    assert all(i < 1280 for i in index)
                    # prefix_length = len(tokenizer(role).input_ids + nl_tokens)
                    # _input_id = _input_id[:prefix_length + index[0] + 1] + _input_id[prefix_length + index[0] + 1:] # + [tokenizer('<imgpad>').input_ids[0]] * 256 
                if '<ref>' in sentence['value']:
                    points.extend(points_in_s)
                else:
                    points.extend([[-100, -100]])
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                          _input_id[len(tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            input_id += _input_id #  add after grounding placeholder insertion
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
        points_conv.append(points) # 记录多轮对话中的所有points
    points_all = torch.tensor(points_conv, dtype=torch.int)
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int64)
    # print(torch.ne(input_ids, 151643).sum().item())

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        points=points_all,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        # print(f'index = {i}: \n{repr(self.raw_data[i]["conversations"])}')
        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            points = ret['points'][0],
        )
        self.cached_data_dict[i] = ret

        return ret

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, points = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "points"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_TOKEN_ID)
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        points = torch.nn.utils.rnn.pad_sequence(points,
                                                 batch_first=True,
                                                 padding_value=IGNORE_TOKEN_ID)
        # input_ids = input_ids[:, :self.tokenizer.model_max_length] # TODO: may truncate point
        # labels = labels[:, :self.tokenizer.model_max_length]
        
        ### Print supervised substrings to debug
        # print(self.tokenizer.decode(input_ids[0].where(input_ids[0] != -200, torch.tensor(0))))
        
        # attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        # print(attention_mask[2])
        # print(torch.argmin(attention_mask[2].int()))
        
        # print(labels[2])
        # label_start = torch.argmax((labels[0] > 0).int()); label_end = torch.argmax((labels[0] == 128009).int())
        # print(label_start, label_end)
        
        # print(self.tokenizer.decode(input_ids[0].where(input_ids[0] != -200, torch.tensor(0))[label_start: label_end+1]))
        batch = dict(
            input_ids=input_ids,
            points=points,
            labels=labels,
            attention_mask=attention_mask,
            # best_w = [x['best_w'] for x in instances]
        )
        
        # if 'origin_image_width' in instances[0]:
        #     origin_image_widths = [instance['origin_image_width'] for instance in instances]
        #     origin_image_heights = [instance['origin_image_height'] for instance in instances]
            
        #     batch['origin_image_widths'] = origin_image_widths
        #     batch['origin_image_heights'] = origin_image_heights

                
            
        # if 'image' in instances[0]:
        #     images = [instance['image'] for instance in instances]
        #     # print("____MY_DEBUG_2____",images)
        #     if all(x is not None and x.shape == images[0].shape for x in images):
        #         batch['images'] = torch.stack(images)
        #     else:
        #         max_of_x = 24
        #         padded_x_tensors = []
        #         for x in images:
        #             padding = torch.zeros(max_of_x - x.size(0), x.size(1), x.size(2), dtype=x.dtype)
        #             # 在第一个维度上堆叠填充
        #             padded_x_tensor = torch.cat((padding, x), dim=0)
        #             padded_x_tensors.append(padded_x_tensor)

        #         batch['images'] = torch.stack(padded_x_tensors)
                

                # if local_rank == 0: 
                #     print(len(batch["origin_image_heights"]))
                #     print("batch shape",batch['images'].shape)
                #     print(batch['origin_image_widths'][0])
                #     print(batch["origin_image_heights"][0])
                #     for i in range(8):
                #         print(f"___________________________{i}_________________________________")
                #         for y in range(5):
                #                 print(batch['images'][0][i*3][0][y].item(),end=" ")
                #         print("|",end=" ")
                #         for y in range(5):
                #                 print(batch['images'][0][i*3][335][330+y].item(),end=" ")
                #         print(" ")
        return batch

def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    # max_length = 0
    # # new_json = []
    # for i in tqdm.tqdm(range(len(train_json))):
    #     for conv in train_json[i]["conversations"]:
    #         if conv['from'] == 'human':
    #             conv['from'] = 'user'
    #         elif conv['from'] == 'gpt':
    #             conv['from'] = 'assistant'
    #     # if i < 3:
    #     #     new_json.append(train_json[i])
    #     #     continue
    #     # if i >= 3 and i < 101000:
    #     #     continue
    #     # if '/data0/jingran/workspace/UI_training_data/Ours-Pretrain/images/cpfs01/user/chengkanzhi/seeclick_web_imgs/bb5056fcf6103c0d3db4c84633de3de7.png' in train_json[i]["conversations"][0]['value']:
    #     #     new_json.append(train_json[i])
    #     #     break
    #     # check fist
    #     length = 0
    #     for conv_i in range(len(train_json[i]["conversations"]) // 2):
    #         tokenized = tokenizer(train_json[i]["conversations"][0]['value'])
    #         points_in_s, index = get_point(tokenizer, train_json[i]["conversations"][0]['value'])
    #         if points_in_s != [[-100, -100]] and '<ref>' in train_json[i]["conversations"][0]['value']:
    #             assert len(points_in_s) == len(index), train_json[i]["conversations"][0]['value']
    #             assert all(i < 1280 for i in index), train_json[i]["conversations"][0]['value']
    #             for point in points_in_s:
    #                 assert all(p < 100 and p >=0 for p in point), train_json[i]["conversations"][0]['value']
    #         length += len(tokenized["input_ids"])
    #     if length > max_length:
    #         max_length = length
    # print("max_length: ", max_length) # 739 for mc.json 739+256 = 995
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator = data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    config.vocab_size = 151936 # vocab size指的是实际的词表大小 (Qwen实际大小是151860），即tokenizer中支持的token数。config.json中数值是实际模型中的embedding size的大小，由于计算效率原因，一般会设置为128的倍数（即151936 = 1187 * 128），会比实际的vocab size大一些哈。
    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.qwen_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(
            bits=4, disable_exllama=True
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
    )

    # customized LoRA parameters
    target_modules = []
    target_layer_names = ["visual.conv1", "attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj", "c_attn",
                          "attn.c_proj", "w1", "w2"]
    lora_supported_types = [torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, transformers.pytorch_utils.Conv1D]
    
    for name, module in model.named_modules():
        if any(t_name in name for t_name in target_layer_names) and 'attn_pool' not in name:
            if isinstance(module, tuple(lora_supported_types)):
                target_modules.append(name)
            else:
                print(name + " not satisfy lora")
                break
                # input()
    
    lora_args.lora_target_modules = target_modules

    """
    # print the LoRA parameters
    for name, param in model.named_parameters():
        if any(target in name for target in lora_args.lora_target_modules):
            print(name)
    """

    if not training_args.use_lora:
        if training_args.fix_vit and hasattr(model, 'transformer') and hasattr(model.transformer, 'visual'):
            model.transformer.visual.requires_grad_(False)
            if hasattr(model.transformer.visual, 'attn_pool'):
                model.transformer.visual.attn_pool.requires_grad_(True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.qwen_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    if training_args.use_lora:
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]

        already_lora = model_args.model_name_or_path != model_args.qwen_path

        if already_lora: # Resume from LoRA. Reference: https://discuss.huggingface.co/t/loading-peft-model-from-checkpoint-leading-into-size-missmatch/71944
            rank0_print(Fore.YELLOW + "Resume LoRA finetuning from the config in" + model_args.model_name_or_path + Style.RESET_ALL)
            lora_config = PeftConfig.from_pretrained(model_args.model_name_or_path)
        else:
            lora_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_args.lora_target_modules,
                lora_dropout=lora_args.lora_dropout,
                bias=lora_args.lora_bias,
                task_type="CAUSAL_LM",
                modules_to_save=modules_to_save  # This argument serves for adding new tokens.
            )

        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        if already_lora:
            model = PeftModel.from_pretrained(model, 
                model_args.model_name_or_path,
                is_trainable=True # 👈 here
                ) 
        else:       
            model = get_peft_model(model, lora_config)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    
    rank0_print("#trainable params")
    rank0_print("Lora param:", sum([x.numel() for name, x in model.named_parameters() if 'lora' in name]))
    rank0_print("Ohter LLM param:", sum([x.numel() for name, x in model.named_parameters() if 'lora' not in name]))
    
    if local_rank == 0:
        model.print_trainable_parameters()
    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    ) 

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()
