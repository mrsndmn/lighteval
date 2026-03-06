#!/usr/bin/env python3
"""Run LightEval benchmarks for one or more model checkpoints and save results per benchmark."""

import fcntl
import os
import tempfile

import torch
from transformers import AutoModelForCausalLM

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


class NoopEvaluationTracker(EvaluationTracker):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("output_dir", tempfile.mkdtemp())
        super().__init__(*args, **kwargs)

    def save(self):
        pass


def main() -> None:
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    # LightEval task modules (e.g. ifeval, ifbench) require NLTK punkt. With multi-node/multi-process
    # runs, all processes would download concurrently and race on the same NLTK_DATA dir → FileExistsError.
    # Use a file lock so only one process downloads; others wait and then see the data.
    import nltk

    nltk_data_dir = os.environ.get("NLTK_DATA", os.path.expanduser("~/nltk_data"))
    os.makedirs(nltk_data_dir, exist_ok=True)
    lock_path = os.path.join(nltk_data_dir, ".nltk_download.lock")
    with open(lock_path, "a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            nltk.download("punkt", quiet=True, download_dir=nltk_data_dir)
            nltk.download("punkt_tab", quiet=True, download_dir=nltk_data_dir)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    model_checkpoint = "unsloth/Llama-3.2-3B"
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_checkpoint),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print("hf_model", type(hf_model))

    batch_size = 256
    benchmark = "mmlu"
    max_new_tokens = 128

    evaluation_tracker = NoopEvaluationTracker()
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
    )

    print(benchmark, "batch_size", batch_size, "max_new_tokens", max_new_tokens)
    config = TransformersModelConfig(
        model_name=str(model_checkpoint),
        batch_size=batch_size,
        generation_parameters={
            "max_new_tokens": max_new_tokens,
        },
    )
    model = TransformersModel.from_model(hf_model, config)

    pipeline = Pipeline(
        model=model,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        tasks=benchmark,
        # max_samples=1,
    )
    pipeline.evaluate()
    pipeline.show_results()
    full_results = pipeline.get_results()
    print("full_results", full_results)


if __name__ == "__main__":
    main()
