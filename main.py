# Modified script from: https://github.com/huggingface/transformers

import psutil
import argparse
import glob
import logging
import os
import random
import timeit
import json
import copy
import tarfile
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import transformers
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import (
    SquadResult,
    SquadV1Processor,
    SquadV2Processor,
)
from transformers.trainer_utils import is_main_process
from torch.utils.data import TensorDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def tar_dumper(model_state_dict, archive_name, file_name, tmp_path="tmp.pt"):
    """
    Save model states into a tar file
    Args:
        - model_state_dict (OrderedDict) - weights of your model after model.state_dict()
        - archive_name (str) - path to the tar file, where your states should be placed
        - file_name (str) - name of your states inside the tar file (model.step4000.pt, e.g.)
        - tmp_path (str) - path to temporary save states, will be removed
    """

    torch.save(model_state_dict, tmp_path)

    with tarfile.open(archive_name, "a") as f:
        f.add(tmp_path, file_name)

    os.remove(tmp_path)


def set_seed(args):
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(int(args.seed))


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    """ Filter negative subsequences"""
    if args.filter_negative:
        logger.info("=" * 50 + "\nFILTERING NEGATIVE EXAMPLES\n" + "=" * 50)
        dataset = train_dataset.tensors
        to_filter = {
            "input_ids": dataset[0],
            "attention_mask": dataset[1],
            "token_type_ids": dataset[2],
            "start_positions": dataset[3],
            "end_positions": dataset[4],
        }
        has_answer = list(
            set(range(len(train_dataset)))
            - set(
                torch.nonzero(to_filter["start_positions"] == 0).view(-1).tolist()
            ).intersection(
                set(torch.nonzero(to_filter["end_positions"] == 0).view(-1).tolist())
            )
        )
        tensors = []
        for t in train_dataset.tensors:
            tensors.append(t[has_answer])
        train_dataset = TensorDataset(*tuple(tensors))

    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    custom_max_steps = (
        len(train_dataloader) // args.gradient_accumulation_steps * 2
    )  # when to stop training
    logger.info("custom_max_steps: %d" % custom_max_steps)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    with open(args.p_pickle, "rb") as f:
        is_clean = pickle.load(f)

    clean_idx = [i for i, el in enumerate(is_clean) if el]
    noise_idx = [i for i, el in enumerate(is_clean) if not el]

    global_step, epochs_trained, steps_trained_in_current_epoch = 1, 0, 0
    delta_norm, c_acc_norm, n_acc_norm, all_acc_norm = [], [], [], []
    c_acc_flag, n_acc_flag = False, False

    gradients_acc_clean, gradients_acc_noise, gradients_acc_all = [], [], []
    gradients_clean, gradients_noise = [], []
    grad_cl, grad_n, grad_all = 0, 0, 0

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps
            )

        except ValueError:
            logger.info("Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    # Added here for reproductibility
    set_seed(args)
    if not os.path.exists(args.gradients_dir) and args.output_grads_weights_ids:
        os.makedirs(args.gradients_dir)

    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):

            if step == 0 and not c_acc_flag:
                old_state = model.state_dict()
                flat_init = torch.zeros(1, 108791040)
                pos = 0
                for k, v in old_state.items():
                    if "weight" in k:
                        flat = torch.flatten(v)
                        flat_init[0, pos : pos + flat.shape[0]] = flat
                        pos += flat.shape[0]

            model.train()

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()

            if global_step < custom_max_steps:
                grad = []
                if (
                    inputs["input_ids"].shape[0]
                    == args.per_gpu_train_batch_size * args.n_gpu
                ):
                    if args.output_grads_weights_ids:
                        for name, param in model.named_parameters():
                            if "bias" not in name:
                                grad.append(param.grad.cpu().detach().reshape(1, -1))
                        grad = torch.cat(grad, dim=1)
                        if is_clean[global_step - 1]:
                            grad_cl += grad
                            gradients_acc_clean.append(torch.norm(grad_cl).item())
                            gradients_clean.append(torch.norm(grad).item())
                        else:
                            grad_n += grad
                            gradients_acc_noise.append(torch.norm(grad_n).item())
                            gradients_noise.append(torch.norm(grad).item())

                        grad_all += grad
                        gradients_acc_all.append(torch.norm(grad_all).item())

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                if global_step < custom_max_steps:
                    flat_batch = torch.zeros(1, 108791040)
                    pos = 0
                    for k, v in model.state_dict().items():
                        if "weight" in k:
                            flat = torch.flatten(v)
                            flat_batch[0, pos : pos + flat.shape[0]] = flat
                            pos += flat.shape[0]

                    delta_w = (flat_batch - flat_init)[0, :]
                    delta_norm.append(torch.norm(delta_w, dim=0).item())

                    if step == 0 and not c_acc_flag:
                        all_acc = delta_w
                    else:
                        all_acc += delta_w
                    all_acc_norm.append(torch.norm(all_acc, dim=0).item())

                    if global_step - 1 in clean_idx:
                        if not c_acc_flag:
                            c_acc = delta_w
                            c_acc_flag = True
                        else:
                            c_acc = c_acc + delta_w

                        c_acc_norm.append(torch.norm(c_acc, dim=0).item())
                        n_acc_norm.append(-1)

                    elif global_step - 1 in noise_idx:
                        if not n_acc_flag:
                            n_acc = delta_w
                            n_acc_flag = True
                        else:
                            n_acc = n_acc + delta_w
                        n_acc_norm.append(torch.norm(n_acc, dim=0).item())
                        c_acc_norm.append(-1)
                    else:
                        "error"
                        break

                    flat_init = flat_batch
                global_step += 1
                if args.output_grads_weights_ids and (global_step < custom_max_steps):
                    if global_step % 1000 == 0:
                        torch.save(
                            {
                                "c_acc_norm": c_acc_norm,
                                "n_acc_norm": n_acc_norm,
                                "all_acc_norm": all_acc_norm,
                                "delta_norm": delta_norm,
                                "c_acc_grad_norm": gradients_acc_clean,
                                "n_acc_grad_norm": gradients_acc_noise,
                                "all_acc_grad_norm": gradients_acc_all,
                                "c_grad_norm": gradients_clean,
                                "n_grad_norm": gradients_noise,
                            },
                            os.path.join(args.gradients_dir, f"norms.pt"),
                        )

                if (
                    args.local_rank in [-1, 0]
                    and (global_step - 1) % len(epoch_iterator) == 0
                ):  # len(epoch_iterator)
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        logger.info(
                            "Results on step {}: {}".format(global_step, results)
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                # Save model checkpoint
                if (
                    args.local_rank in [-1, 0]
                    and (global_step - 1) % len(epoch_iterator) == 0
                ):
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step)
                    )
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(
                        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                    )
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir
                    )

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    torch.save(
        {
            "c_acc_norm": c_acc_norm,
            "n_acc_norm": n_acc_norm,
            "all_acc_norm": all_acc_norm,
            "delta_norm": delta_norm,
            "c_acc_grad_norm": gradients_acc_clean,
            "n_acc_grad_norm": gradients_acc_noise,
            "all_acc_grad_norm": gradients_acc_all,
            "c_grad_norm": gradients_clean,
            "n_grad_norm": gradients_noise,
        },
        os.path.join(args.gradients_dir, f"norms.pt"),
    )

    logger.info("end")
    return global_step, tr_loss / global_step


def token_ids_retreive(args, train_dataset, model, tokenizer):
    """ Filter negative subsequences"""
    if args.filter_negative:
        logger.info("=" * 50 + "\nFILTERING NEGATIVE EXAMPLES\n" + "=" * 50)
        dataset = train_dataset.tensors
        to_filter = {
            "input_ids": dataset[0],
            "attention_mask": dataset[1],
            "token_type_ids": dataset[2],
            "start_positions": dataset[3],
            "end_positions": dataset[4],
        }
        has_answer = list(
            set(range(len(train_dataset)))
            - set(
                torch.nonzero(to_filter["start_positions"] == 0).view(-1).tolist()
            ).intersection(
                set(torch.nonzero(to_filter["end_positions"] == 0).view(-1).tolist())
            )
        )
        tensors = []
        for t in train_dataset.tensors:
            tensors.append(t[has_answer])
        train_dataset = TensorDataset(*tuple(tensors))

    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    global_step, epochs_trained, steps_trained_in_current_epoch = 1, 0, 0
    inputs_ids = []
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps
            )

        except ValueError:
            logger.info("Starting fine-tuning.")

    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    # Added here for reproductibility
    set_seed(args)
    if not os.path.exists(args.gradients_dir) and args.output_grads_weights_ids:
        os.makedirs(args.gradients_dir)

    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }
            print(inputs["input_ids"][0])
            inputs_ids.append(inputs["input_ids"])
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                if args.output_grads_weights_ids:
                    if global_step % 100000 == 0:
                        torch.save(
                            inputs_ids,
                            os.path.join(args.gradients_dir, f"inputs_ids.pt"),
                        )
                        logger.info(
                            "100.000 steps: Token ids saved into %s"
                            % args.gradients_dir
                        )
                        inputs_ids = []

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    if args.output_grads_weights_ids:
        torch.save(inputs_ids, os.path.join(args.gradients_dir, f"inputs_ids_.pt"))
        logger.info("Token ids saved into %s" % args.gradients_dir)
    return inputs_ids


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(
        args, tokenizer, evaluate=True, output_examples=True
    )

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in [
                "xlm",
                "roberta",
                "distilbert",
                "camembert",
                "bart",
                "longformer",
            ]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {
                            "langs": (
                                torch.ones(batch[0].shape, dtype=torch.int64)
                                * args.lang_id
                            ).to(args.device)
                        }
                    )
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs.to_tuple()]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info(
        "  Evaluation done in total %f secs (%f sec per example)",
        evalTime,
        evalTime / len(dataset),
    )

    # Compute predictions
    output_prediction_file = os.path.join(
        args.output_dir, "predictions_{}.json".format(prefix)
    )
    output_nbest_file = os.path.join(
        args.output_dir, "nbest_predictions_{}.json".format(prefix)
    )

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(
            args.output_dir, "null_odds_{}.json".format(prefix)
        )
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = (
            model.config.start_n_top
            if hasattr(model, "config")
            else model.module.config.start_n_top
        )
        end_n_top = (
            model.config.end_n_top
            if hasattr(model, "config")
            else model.module.config.end_n_top
        )

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)

    return results


def output_attentions_and_tokens(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(
        args, tokenizer, evaluate=True, output_examples=True
    )

    if not os.path.exists(args.attentions_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.attentions_dir)

    args.attention_batch_size = args.per_gpu_attention_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.attention_batch_size
    )

    # multi-gpu
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("***** Running attention retrieving {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.attention_batch_size)

    attentions = []
    tokenized_sentences = []
    part_num = 0
    for num_iter, batch in enumerate(
        tqdm(eval_dataloader, desc="Retreiving attentions")
    ):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            feature_indices = batch[3]
            outputs = model(**inputs)

            batch_attention = (
                torch.stack(list(outputs["attentions"]), dim=0).cpu().detach().numpy()
            )
            if batch_attention.shape[1] == args.attention_batch_size:
                free_mem = psutil.virtual_memory().free
                if (num_iter % args.save_attentions_step == 0) and (num_iter != 0):
                    attention = torch.flatten(
                        torch.stack([torch.from_numpy(i) for i in attentions], dim=1),
                        start_dim=1,
                        end_dim=2,
                    )
                    logger.info(
                        "** Saving attentions into {} **".format(
                            "attentions_part%d.pt" % part_num
                        )
                    )
                    torch.save(
                        attention,
                        os.path.join(
                            args.attentions_dir, "attentions_part%d.pt" % part_num
                        ),
                    )
                    del attentions
                    tokens = torch.flatten(
                        torch.stack(
                            [torch.from_numpy(i) for i in tokenized_sentences], dim=1
                        ),
                        start_dim=0,
                        end_dim=1,
                    )
                    logger.info(
                        "** Saving token_ids into {} **".format(
                            "token_ids_part%d.pt" % part_num
                        )
                    )
                    torch.save(
                        tokens,
                        os.path.join(
                            args.attentions_dir, "token_ids_part%d.pt" % part_num
                        ),
                    )
                    del tokenized_sentences
                    attentions = []
                    tokenized_sentences = []
                    part_num += 1
                attentions.append(batch_attention)
                tokenized_sentences.append(inputs["input_ids"].cpu().detach().numpy())
            else:
                logger.warning(
                    "The last batch has not been saved set per_gpu_attention_batch_size according to the rule: num_examples % (per_gpu_attention_batch_size * n_gpu) == 0"
                )

            # print(psutil.virtual_memory().free)
    attention = torch.flatten(
        torch.stack([torch.from_numpy(i) for i in attentions], dim=1),
        start_dim=1,
        end_dim=2,
    )
    tokens = torch.flatten(
        torch.stack([torch.from_numpy(i) for i in tokenized_sentences], dim=1),
        start_dim=0,
        end_dim=1,
    )
    part_num += 1
    logger.info(
        "** Saving attentions into {} **".format("attentions_part%d.pt" % part_num)
    )
    torch.save(
        attention, os.path.join(args.attentions_dir, "attentions_part%d.pt" % part_num)
    )
    logger.info(
        "** Saving token_ids into {} **".format("token_ids_part%d.pt" % part_num)
    )
    torch.save(
        tokens, os.path.join(args.attentions_dir, "token_ids_part%d.pt" % part_num)
    )

    return attention, tokens


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and (
            (evaluate and not args.predict_file)
            or (not evaluate and not args.train_file)
        ):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError(
                    "If not data_dir is specified, tensorflow_datasets needs to be installed."
                )

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(
                tfds_examples, evaluate=evaluate
            )
        else:
            processor = (
                SquadV2Processor()
                if args.version_2_with_negative
                else SquadV1Processor()
            )
            if evaluate:
                examples = processor.get_dev_examples(
                    args.data_dir, filename=args.predict_file
                )
            else:
                examples = processor.get_train_examples(
                    args.data_dir, filename=args.train_file
                )

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--retreive_token_ids", action="store_true", help="Get token ids from examples"
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--per_gpu_attention_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for attention retreiving.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument(
        "--logging_steps", type=int, default=500, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    # N
    parser.add_argument(
        "--output_grads_weights_ids",
        action="store_true",
        help="Whether to output gradients and weights (or token ids if token_ids_retreive",
    )
    # N
    parser.add_argument(
        "--freeze_pretrained",
        type=bool,
        default=False,
        help="Whether to freeze pretrained layers of BERT",
    )
    # N
    parser.add_argument(
        "--gradients_dir",
        type=str,
        default="",
        help="Directory for gadients and exanples ",
    )
    # N
    parser.add_argument(
        "--output_attentions",
        type=bool,
        default=False,
        help="Whether to output attentions of the model on dev set",
    )

    # N
    parser.add_argument(
        "--attentions_dir", type=str, default="", help="Directory for attention output",
    )
    # N
    parser.add_argument(
        "--save_attentions_step",
        type=int,
        default=50,
        help="Save attentions after every n steps",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="Can be used for distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="Can be used for distant debugging."
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="multiple threads for converting example to features",
    )
    parser.add_argument(
        "--p_pickle",
        type=str,
        default="",
        help="Path to pickle file with clean inputs ids.",
    )
    parser.add_argument(
        "--filter_negative", type=str, help="Whether to filter examples without answer"
    )
    parser.add_argument(
        "--logging_file_path_name",
        type=str,
        default="training_log",
        help="Path to to log file and filename",
    )
    parser.add_argument(
        "--token_ids_retreive", action="store_true", help="Whether to output token ids "
    )

    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    logger_dir = os.path.dirname(args.logging_file_path_name)
    if logger_dir:
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)

    # Setup logging
    logging.basicConfig(
        filename="%s.log" % args.logging_file_path_name,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    if args.filter_negative == "True":
        logger.info("=" * 50 + "\nFILTERING NEGATIVE EXAMPLES\n" + "=" * 50)
    else:
        logger.info("=" * 50 + "\nNO FILTERING EXAMPLES\n" + "=" * 50)

    logger.info("SEED:%d" % int(args.seed))

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=False,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

    if args.token_ids_retreive:
        train_dataset = load_and_cache_examples(
            args, tokenizer, evaluate=False, output_examples=False
        )
        ids_ = token_ids_retreive(args, train_dataset, model, tokenizer)
        logger.info("Done with token ids")

    else:
        # Training
        if args.do_train:
            train_dataset = load_and_cache_examples(
                args, tokenizer, evaluate=False, output_examples=False
            )
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.output_dir
        )  # , force_download=True)

        # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
        # So we use use_fast=False here for now until Fast-tokenizer-compatible-examples are out
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case, use_fast=False
        )
        model.to(args.device)

    # Retreive attentions and tokens
    if args.output_attentions and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(
                        glob.glob(
                            args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True
                        )
                    )
                )

        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Attentions from the following checkpoint: %s", checkpoints[-1])

        checkpoint = checkpoints[-1]
        # Reload the model
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        model = AutoModelForQuestionAnswering.from_pretrained(
            checkpoint, output_attentions=True
        )  # , force_download=True)
        model.to(args.device)

        attention, tokens = output_attentions_and_tokens(
            args, model, tokenizer, prefix=global_step
        )

        logger.info(
            "**** Successfully retrieved attentions in the {} ****".format(
                args.attentions_dir
            )
        )

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(
                        glob.glob(
                            args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True
                        )
                    )
                )

        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)
            logger.info("Results_current: {}".format(result))
            cur_PATH = os.path.join(args.output_dir, "results.json")
            if not os.path.exists(cur_PATH):
                spot = {}
                with open(cur_PATH, "w") as json_file:
                    json.dump(spot, json_file)
            else:
                with open(cur_PATH, "r") as json_file:
                    data = json.load(json_file)
                for item in result:
                    data[item + " {}".format(global_step)] = result[item]
                with open(cur_PATH, "w") as json_file:
                    json.dump(data, json_file)

            result = dict(
                (k + ("_{}".format(global_step) if global_step else ""), v)
                for k, v in result.items()
            )
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()
