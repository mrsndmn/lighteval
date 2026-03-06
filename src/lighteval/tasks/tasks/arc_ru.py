"""
name:
Arc (Russian)

dataset:
mrsndmn/ai2_arc_ru

abstract:
7,787 genuine grade-school level, multiple-choice science questions, assembled
to encourage research in advanced question-answering. Russian translation.
Partitioned into Challenge Set and Easy Set.

languages:
russian

tags:
multiple-choice

paper:
https://arxiv.org/abs/1803.05457
"""

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def arc_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=line["choices"]["label"].index(line["answerKey"]),
    )


def record_to_sample(record):
    query = record["question"].strip()
    target = record["answerKey"]
    choices = record["choices"]["text"]

    return Sample(input=query, target=target, choices=choices)


arc_challenge_ru = LightevalTaskConfig(
    name="arc_ru:challenge",
    prompt_function=arc_prompt,
    hf_repo="mrsndmn/ai2_arc_ru",
    hf_subset="ARC-Challenge",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

arc_easy_ru = LightevalTaskConfig(
    name="arc_ru:easy",
    prompt_function=arc_prompt,
    hf_repo="mrsndmn/ai2_arc_ru",
    hf_subset="ARC-Easy",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

TASKS_TABLE = [arc_challenge_ru, arc_easy_ru]
