# Copyright (c) 2026 The Cactus Authors

import argparse
import copy
import json
import os
import random
import re
import time
from itertools import chain

import lm_eval.api.model
import lm_eval.api.registry
import lm_eval.tasks
import numpy as np
import torch
import wandb
from lm_eval import evaluator, models
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter
from lm_eval.utils import simple_parse_args_string

from cactus import cactus, interp, rs, tas, topk
from cactus.constant import tracker

MODELS = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B-FP8",
    "yongchanghao/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
]


@register_filter("take_last")
class TakeLastFilter(Filter):
    def apply(self, resps, docs):
        return map(lambda x: x[-1], resps)


@register_filter("remove_thinking_token")
class RemoveThinkingToken(Filter):
    """
    First removes <think>.*?</think> from the response.
    Then deletes all remaining <think> and </think> tokens.
    """

    def apply(self, resps, docs):
        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                # Remove <think>everything, including newlines</think>
                resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL)
                # <think> might not appear in the response, but </think> might appear
                # If so, remove everything before </think>
                resp = re.sub(r".*?</think>", "", resp, flags=re.DOTALL)
                # Remove all remaining <think> tokens
                resp = re.sub(r"<think>", "", resp, flags=re.DOTALL)
                # Remove left blank lines
                resp = resp.lstrip("\n")
                filtered_resp.append(resp)
            return filtered_resp

        filtered_resps = [filter_set(resp) for resp in resps]
        return filtered_resps


def monkey_patch(target_module):
    import vllm.model_executor.layers.rejection_sampler
    import vllm.model_executor.layers.typical_acceptance_sampler

    vllm.model_executor.layers.rejection_sampler.RejectionSampler = target_module
    vllm.model_executor.layers.typical_acceptance_sampler.TypicalAcceptanceSampler = target_module


def naive_model(
    pretrained,
    max_model_len,
    gpu_memory_utilization=0.8,
):
    return models.vllm_causallms.VLLM(
        pretrained=pretrained,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        use_v2_block_manager=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )


def speculative_model(
    pretrained,
    speculative_model,
    num_speculative_tokens,
    max_model_len,
    gpu_memory_utilization=0.8,
    spec_decoding_acceptance_method="rejection_sampler",
    typical_acceptance_alpha=None,
    typical_acceptance_threshold=None,
    delta=1.0,
    top_k=5,
    real_time_al=False,
    enable_experimental_triton=False,
):
    if spec_decoding_acceptance_method == "cactus":
        cactus.CactusRejectionSampler.delta = delta
        cactus.CactusRejectionSampler.real_time_al = real_time_al
        cactus.CactusRejectionSampler.enable_experimental_triton = enable_experimental_triton
        monkey_patch(cactus.CactusRejectionSampler)
        spec_decoding_acceptance_method = "rejection_sampler"
    elif spec_decoding_acceptance_method == "typical_acceptance_sampler":
        monkey_patch(tas.WrappedTypicalAcceptanceSampler)
    elif spec_decoding_acceptance_method == "rejection_sampler":
        rs.WrappedRejectionSampler.real_time_al = real_time_al
        monkey_patch(rs.WrappedRejectionSampler)
    elif spec_decoding_acceptance_method == "interp":
        if delta is not None:
            interp.InterpRejectionSampler.alpha = delta
        interp.InterpRejectionSampler.real_time_al = real_time_al
        monkey_patch(interp.InterpRejectionSampler)
        spec_decoding_acceptance_method = "rejection_sampler"
    elif spec_decoding_acceptance_method == "topk":
        if top_k is not None:
            topk.TopKRejectionSampler.topk = top_k
        topk.TopKRejectionSampler.real_time_al = real_time_al
        monkey_patch(topk.TopKRejectionSampler)
        spec_decoding_acceptance_method = "rejection_sampler"
    else:
        raise ValueError(
            f"Unknown spec_decoding_acceptance_method: {spec_decoding_acceptance_method}"
        )
    return models.vllm_causallms.VLLM(
        pretrained=pretrained,
        num_speculative_tokens=num_speculative_tokens,
        speculative_model=speculative_model,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        speculative_max_model_len=max_model_len,
        spec_decoding_acceptance_method=spec_decoding_acceptance_method,
        use_v2_block_manager=True,
        typical_acceptance_sampler_posterior_threshold=typical_acceptance_threshold,
        typical_acceptance_sampler_posterior_alpha=typical_acceptance_alpha,
    )


def prepare_task(
    model,
    model_args=None,
    tasks=None,
    num_fewshot=None,
    batch_size=None,
    max_batch_size=None,
    device=None,
    gen_kwargs=None,
    random_seed=0,
    numpy_random_seed=1234,
    torch_random_seed=1234,
    fewshot_random_seed=1234,
    metadata=None,
    predict_only=False,
):
    if random_seed is not None:
        random.seed(random_seed)

    if numpy_random_seed is not None:
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        torch.manual_seed(torch_random_seed)

    if tasks is None:
        tasks = []
    if len(tasks) == 0:
        raise ValueError("No tasks specified, or no tasks found. Please verify the task names.")

    if gen_kwargs is not None:
        if isinstance(gen_kwargs, str):
            gen_kwargs = simple_parse_args_string(gen_kwargs)
        if not gen_kwargs:
            gen_kwargs = None

    if isinstance(model, str):
        if model_args is None:
            model_args = ""

        if isinstance(model_args, dict):
            lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )

        else:
            lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )
    else:
        if not isinstance(model, lm_eval.api.model.LM):
            raise TypeError(
                f"The value of `model` passed to simple_evaluate() was of type"
                f" {type(model)}, but is required to be a subclass of"
                f" lm_eval.api.model.LM. This may be because you are passing an"
                f" initialized Hugging Face PreTrainedModel without having wrapped"
                f" it in `lm_eval.models.huggingface.HFLM(pretrained=my_model)` first."
            )

        lm = model

    metadata = (
        simple_parse_args_string(model_args)
        if isinstance(model_args, str)
        else model_args
        if isinstance(model_args, dict)
        else {}
    ) | (metadata or {})
    task_manager = lm_eval.tasks.TaskManager(metadata=metadata)

    task_dict = evaluator.get_task_dict(
        tasks,
        task_manager,
    )

    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }

            else:
                if task_obj.get_config("output_type") == "generate_until":
                    if gen_kwargs is not None:
                        task_obj.set_config(key="generation_kwargs", value=gen_kwargs, update=True)

                    # Custom area: removing "\n" from the until and add "<|eot_id|>"
                    if task_obj.get_config(key="generation_kwargs").get("until", None) is not None:
                        og_until = task_obj.get_config(key="generation_kwargs").get("until")
                        if isinstance(og_until, str):
                            og_until = og_until.replace("\n", "")
                        elif isinstance(og_until, list):
                            for tok in ["\n", "\n\n", "\n\n\n", "\n\n\n\n"]:
                                if tok in og_until:
                                    og_until.remove(tok)

                            for tok in ["<|eot_id|>", "</s>", "<eos>", "<|im_end|>"]:
                                if tok not in og_until:
                                    og_until.append(tok)

                        task_obj.set_config(
                            key="generation_kwargs",
                            value={"until": og_until},
                            update=True,
                        )

                if predict_only:
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs
                if num_fewshot is not None:
                    if task_obj.get_config("num_fewshot") == 0:
                        pass
                    else:
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    # if num_fewshot not provided and task has no default, use 0
                    if task_obj.get_config("num_fewshot") is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)

                # Custom area: adding filters
                flexible_extract_filter = None
                for f in task_obj._filters:
                    f.filters.insert(0, RemoveThinkingToken)  # This is a hack
                    if f.name == "flexible-extract":
                        flexible_extract_filter = copy.deepcopy(f)

                # Add the TakeLastFilter to the flexible_extract_filter
                if flexible_extract_filter is not None:
                    flexible_extract_filter.name = "flexible-extract-last"
                    flexible_extract_filter.filters[-1] = TakeLastFilter
                    task_obj._filters.append(flexible_extract_filter)
                # End of custom area

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    task_dict = _adjust_config(task_dict)

    return lm, task_dict


def lm_evaluate(task, model, gen_kwargs=None):
    tracker.task = task
    model, task_dict = prepare_task(model=model, tasks=[task], gen_kwargs=gen_kwargs)
    results = evaluator.evaluate(
        model,
        task_dict=task_dict,
        confirm_run_unsafe_code=True,
        apply_chat_template=True,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="speculative model")
    parser.add_argument(
        "-t",
        "--tasks",
        type=str,
        nargs="+",
        help="tasks to evaluate",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=MODELS,
        help="model to evaluate",
    )
    parser.add_argument(
        "-s",
        "--speculative-model",
        type=str,
        choices=MODELS,
        help="speculative model to use",
    )

    parser.add_argument(
        "-n",
        "--num-speculative-tokens",
        type=int,
        default=10,
        help="number of speculative tokens to generate",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="maximum model length",
    )
    parser.add_argument(
        "-c",
        "--spec-decoding-acceptance-method",
        type=str,
        default="rejection_sampler",
        choices=[
            "rejection_sampler",
            "typical_acceptance_sampler",
            "cactus",
            "interp",
            "naive",
            "topk",
        ],
        help="speculative decoding acceptance method",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="fraction of GPU memory to use",
    )
    parser.add_argument(
        "--gen-kwargs",
        type=str,
        default="do_sample=True,top_p=0.95,temperature=0.6,top_k=20,max_gen_toks=4096",
        help="generation kwargs",
    )
    parser.add_argument(
        "--t-threshold",
        type=float,
        default=None,
        help="threshold for typical acceptance sampler",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="delta for cactus acceptance method",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="k for top-k acceptance method",
    )
    parser.add_argument(
        "--real-time-al",
        action="store_true",
        help="real-time accepted length: log per-step AL to wandb",
    )
    parser.add_argument(
        "--enable-experimental-triton",
        action="store_true",
        help="enable experimental Triton kernel instead of the default PyTorch path",
    )

    args = parser.parse_args()
    print(args)

    # Validate arguments
    if args.spec_decoding_acceptance_method == "cactus":
        if args.delta is None:
            raise ValueError("--delta must be set when using cactus")

    # Parse tasks
    task_list = []
    for task in args.tasks:
        if "," in task:
            task_list.extend(task.split(","))
        else:
            task_list.append(task)

    # task_manager = tasks.TaskManager()
    tasks_list = list(set(task_list))
    # assert_all_tasks_exist(task_manager, tasks_list)

    # Load model
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    if args.speculative_model is None or args.spec_decoding_acceptance_method == "naive":
        # Use naive model
        model = naive_model(
            pretrained=args.model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        model = speculative_model(
            pretrained=args.model,
            speculative_model=args.speculative_model,
            num_speculative_tokens=args.num_speculative_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            spec_decoding_acceptance_method=args.spec_decoding_acceptance_method,
            typical_acceptance_alpha=args.t_threshold**0.5 if args.t_threshold else None,
            typical_acceptance_threshold=args.t_threshold,
            delta=args.delta,
            top_k=args.topk,
            real_time_al=args.real_time_al,
            enable_experimental_triton=args.enable_experimental_triton,
        )

    # Name the model
    name_form_config = "{}-{}({})-M({})".format(
        args.model,
        (
            args.spec_decoding_acceptance_method
            if args.spec_decoding_acceptance_method != "cactus"
            else f"cactus@d({args.delta})"
        ),
        args.speculative_model,
        args.num_speculative_tokens,
    )

    for special_char in [" ", "/", ".", ":"]:
        name_form_config = name_form_config.replace(special_char, "-")

    # Evaluate
    global_results = {}
    wandb.init(project="cactus", name=name_form_config, config=vars(args))
    for task in tasks_list:
        print(f"Evaluating task: {task}")
        start_time = time.perf_counter()
        results = lm_evaluate(task, model, gen_kwargs=args.gen_kwargs) or {}
        end_time = time.perf_counter()
        results["results"][task]["time"] = end_time - start_time
        results["results"][task]["model"] = name_form_config
        results["results"][task]["avg_accepted_length"] = tracker.avg
        tracker.reset()
        table = wandb.Table(columns=["key", "value"], allow_mixed_types=True)
        _num_res = {}
        for key, value in results["results"][task].items():
            print(f"{key}: {value}")
            table.add_data(key, value)
            if isinstance(value, (int, float)):
                _num_res[f"{task}/{key}"] = value
        wandb.log({task: table})
        wandb.log(_num_res)

        # Save results with less samples
        results["samples"][task] = list(
            chain(results["samples"][k][:3] for k in results["samples"].keys())
        )
        global_results[task] = copy.deepcopy(results)
    wandb.finish()

    # Save results locally
    date_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs("results", exist_ok=True)
    with open(f"results/{date_time}-{name_form_config}.json", "w", encoding="utf-8") as f:
        json.dump(
            global_results,
            f,
            indent=2,
            ensure_ascii=False,
            default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>",
        )
