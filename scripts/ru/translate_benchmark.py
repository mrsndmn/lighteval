#!/usr/bin/env python3
"""
Translate benchmark datasets to Russian via OpenAI Responses API (flex tier).

Supported: Rowan/hellaswag (validation+test) -> mrsndmn/hellaswag_ru;
          allenai/ai2_arc (ARC-Challenge, ARC-Easy validation+test) -> mrsndmn/ai2_arc_ru.

Uses service_tier="flex" for ~50%% cost savings.
https://developers.openai.com/api/docs/guides/flex-processing/

Usage:
  OPENAI_API_KEY=sk-... python scripts/ru/ru_hellaswag.py --dataset hellaswag [--limit N]
  OPENAI_API_KEY=sk-... python scripts/ru/ru_hellaswag.py --dataset ai2_arc [--limit N]
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from openai import DefaultHttpxClient, OpenAI
from tqdm import tqdm


def _call_responses_api(
    client: OpenAI,
    model: str,
    instructions: str,
    input_text: str,
    service_tier: str = "flex",
    timeout: float = 900.0,
) -> str:
    """Call OpenAI Responses API; return response text. Handles 429 fallback to auto."""
    create_kwargs: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "service_tier": service_tier if service_tier in ("flex", "auto") else "auto",
    }
    for attempt in range(5):
        try:
            api_client = client.with_options(timeout=timeout) if timeout else client
            response = api_client.responses.create(**create_kwargs)
            content = getattr(response, "output_text", None)
            if content is None and hasattr(response, "output"):
                out = response.output
                content = out[0].content if out and hasattr(out[0], "content") else ""
            return (content or "").strip()
        except Exception as e:
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            if code == 429 and service_tier == "flex" and attempt >= 2:
                time.sleep(5.0 * (attempt + 1))
                try:
                    fallback = {**create_kwargs, "service_tier": "auto"}
                    api_client = client.with_options(timeout=timeout) if timeout else client
                    response = api_client.responses.create(**fallback)
                    content = getattr(response, "output_text", None)
                    if content is None and hasattr(response, "output"):
                        out = response.output
                        content = out[0].content if out and hasattr(out[0], "content") else ""
                    return (content or "").strip()
                except Exception:
                    pass
            if attempt == 4:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("Unreachable")


def _extract_json_from_content(content: str) -> str:
    """Extract JSON string from model output (handle markdown code blocks)."""
    content = content.strip()
    if "```" in content:
        parts = content.split("```")
        for p in parts:
            p = p.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                return p
    return content


def _parse_hellaswag_response(content: str) -> dict[str, Any]:
    out = json.loads(_extract_json_from_content(content))
    ctx_ru = out["ctx_a"] + " " + out["ctx_b"]
    return {
        "activity_label": out["activity_label"],
        "ctx_a": out["ctx_a"],
        "ctx_b": out["ctx_b"],
        "ctx": ctx_ru,
        "endings": out["endings"],
    }


def _parse_arc_response(content: str) -> dict[str, Any]:
    out = json.loads(_extract_json_from_content(content))
    return {"question": out["question"], "choices_text": out["choices_text"]}


def translate_row_hellaswag(
    client: OpenAI,
    row: dict[str, Any],
    model: str,
    service_tier: str = "flex",
    timeout: float = 900.0,
) -> dict[str, Any]:
    """Translate one HellaSwag row to Russian."""
    instructions = "You are a translator. Reply only with valid JSON, no markdown."
    input_text = """Translate the following HellaSwag item to Russian. Preserve meaning and style.
Return valid JSON only, with these exact keys (all strings; "endings" is an array of 4 strings):
"activity_label", "ctx_a", "ctx_b", "endings"

Input:
- activity_label: %(activity_label)s
- ctx_a: %(ctx_a)s
- ctx_b: %(ctx_b)s
- endings: %(endings)s""" % {
        "activity_label": row["activity_label"],
        "ctx_a": row["ctx_a"],
        "ctx_b": row["ctx_b"],
        "endings": json.dumps(row["endings"], ensure_ascii=False),
    }
    for attempt in range(5):
        try:
            content = _call_responses_api(client, model, instructions, input_text, service_tier, timeout)
            return _parse_hellaswag_response(content)
        except (json.JSONDecodeError, KeyError) as e:
            if attempt == 4:
                raise RuntimeError(f"Translation failed after 5 attempts: {e}") from e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("Unreachable")


def translate_row_arc(
    client: OpenAI,
    row: dict[str, Any],
    model: str,
    service_tier: str = "flex",
    timeout: float = 900.0,
) -> dict[str, Any]:
    """Translate one AI2 ARC row (question + choice texts) to Russian."""
    instructions = "You are a translator. Reply only with valid JSON, no markdown."
    choices = row["choices"]
    labels = choices["label"] if isinstance(choices["label"], list) else [choices["label"]]
    texts = choices["text"] if isinstance(choices["text"], list) else [choices["text"]]
    n_choices = len(texts)
    input_text = """Translate the following multiple-choice question and its answer options to Russian. Preserve meaning.
Return valid JSON only with exactly two keys: "question" (string) and "choices_text" (array of %(n)d strings in the same order as the inputs).

Input:
- question: %(question)s
- choices (in order): %(choices)s""" % {
        "n": n_choices,
        "question": row["question"],
        "choices": json.dumps(texts, ensure_ascii=False),
    }
    for attempt in range(5):
        try:
            content = _call_responses_api(client, model, instructions, input_text, service_tier, timeout)
            parsed = _parse_arc_response(content)
            # Keep original labels (A,B,C,D); only text is translated
            return {
                "question": parsed["question"],
                "choices_text": parsed["choices_text"],
                "labels": labels,
            }
        except (json.JSONDecodeError, KeyError) as e:
            if attempt == 4:
                raise RuntimeError(f"Translation failed after 5 attempts: {e}") from e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("Unreachable")


def _translate_one_hellaswag(
    item: tuple[int, dict[str, Any], str, str, str, float, str | None, float],
) -> tuple[int, dict[str, Any]]:
    """Worker: translate one HellaSwag row."""
    idx, row, api_key, model, service_tier, timeout, proxy, delay = item
    http_client = DefaultHttpxClient(proxy=proxy) if proxy else None
    client = OpenAI(api_key=api_key, http_client=http_client)
    translated = translate_row_hellaswag(client, row, model, service_tier=service_tier, timeout=timeout)
    if delay > 0:
        time.sleep(delay)
    new_row = {
        "ind": row["ind"],
        "activity_label": translated["activity_label"],
        "ctx_a": translated["ctx_a"],
        "ctx_b": translated["ctx_b"],
        "ctx": translated["ctx"],
        "endings": translated["endings"],
        "source_id": row["source_id"],
        "split": row["split"],
        "split_type": row["split_type"],
        "label": row["label"],
    }
    return (idx, new_row)


def _translate_one_arc(
    item: tuple[int, dict[str, Any], str, str, str, float, str | None, float],
) -> tuple[int, dict[str, Any]]:
    """Worker: translate one AI2 ARC row."""
    idx, row, api_key, model, service_tier, timeout, proxy, delay = item
    http_client = DefaultHttpxClient(proxy=proxy) if proxy else None
    client = OpenAI(api_key=api_key, http_client=http_client)
    translated = translate_row_arc(client, row, model, service_tier=service_tier, timeout=timeout)
    if delay > 0:
        time.sleep(delay)
    choices = row["choices"]
    labels = choices["label"] if isinstance(choices["label"], list) else list(choices["label"])
    new_row = {
        "id": row["id"],
        "question": translated["question"],
        "choices": {"label": labels, "text": translated["choices_text"]},
        "answerKey": row["answerKey"],
    }
    return (idx, new_row)


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description="Translate HellaSwag validation/test to Russian and push to HF")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (default: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.1",
        help="Model for Responses API",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows per split (for testing)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between API calls (default: 0.5)",
    )
    parser.add_argument(
        "--service-tier",
        choices=("flex", "auto"),
        default="flex",
        help="Responses API service tier: flex (50%% cheaper) or auto (default: flex)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Request timeout in seconds; use higher for flex (default: 900)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for translation (default: 1)",
    )
    parser.add_argument(
        "--dataset",
        choices=("hellaswag", "ai2_arc"),
        default="hellaswag",
        help="Benchmark to translate: hellaswag or ai2_arc (default: hellaswag)",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Do not push to Hub; only build dataset",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    proxy = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )

    if args.dataset == "hellaswag":
        splits = ["validation"]
        ds = load_dataset("Rowan/hellaswag", split=None, trust_remote_code=True)
        if not all(s in ds for s in splits):
            raise SystemExit("Dataset must contain validation and test splits")
        result = {}
        for split in splits:
            data = ds[split]
            if args.limit is not None:
                data = data.select(range(min(args.limit, len(data))))
            data_list = [dict(row) for row in data]
            n = len(data_list)
            worker_args = (
                args.api_key,
                args.model,
                args.service_tier,
                args.timeout,
                proxy,
                args.delay,
            )
            items = [(i, data_list[i], *worker_args) for i in range(n)]
            if args.workers <= 1:
                http_client = DefaultHttpxClient(proxy=proxy) if proxy else None
                client = OpenAI(api_key=args.api_key, http_client=http_client)
                rows = []
                for row in tqdm(data_list, desc=split):
                    translated = translate_row_hellaswag(
                        client,
                        row,
                        args.model,
                        service_tier=args.service_tier,
                        timeout=args.timeout,
                    )
                    rows.append(
                        {
                            "ind": row["ind"],
                            "activity_label": translated["activity_label"],
                            "ctx_a": translated["ctx_a"],
                            "ctx_b": translated["ctx_b"],
                            "ctx": translated["ctx"],
                            "endings": translated["endings"],
                            "source_id": row["source_id"],
                            "split": row["split"],
                            "split_type": row["split_type"],
                            "label": row["label"],
                        }
                    )
                    if args.delay > 0:
                        time.sleep(args.delay)
            else:
                with Pool(processes=args.workers) as pool:
                    results = list(
                        tqdm(
                            pool.imap(_translate_one_hellaswag, items, chunksize=1),
                            total=n,
                            desc=split,
                        )
                    )
                results.sort(key=lambda x: x[0])
                rows = [r[1] for r in results]
            result[split] = Dataset.from_list(rows)
        out_ds = DatasetDict(result)
        hub_id = "mrsndmn/hellaswag_ru"
        local_path = "hellaswag_ru_local"
    else:
        # ai2_arc: ARC-Challenge and ARC-Easy, each validation + test
        splits = ["test"]
        arc_configs = ["ARC-Challenge", "ARC-Easy"]
        result_by_config = {}
        for config_name in arc_configs:
            ds = load_dataset("allenai/ai2_arc", config_name, split=None, trust_remote_code=True)
            if not all(s in ds for s in splits):
                raise SystemExit(f"ai2_arc {config_name} must contain validation and test")
            result = {}
            for split in splits:
                data = ds[split]
                if args.limit is not None:
                    data = data.select(range(min(args.limit, len(data))))
                data_list = [dict(row) for row in data]
                n = len(data_list)
                worker_args = (
                    args.api_key,
                    args.model,
                    args.service_tier,
                    args.timeout,
                    proxy,
                    args.delay,
                )
                items = [(i, data_list[i], *worker_args) for i in range(n)]
                if args.workers <= 1:
                    http_client = DefaultHttpxClient(proxy=proxy) if proxy else None
                    client = OpenAI(api_key=args.api_key, http_client=http_client)
                    rows = []
                    for row in tqdm(data_list, desc=f"{config_name}/{split}"):
                        translated = translate_row_arc(
                            client,
                            row,
                            args.model,
                            service_tier=args.service_tier,
                            timeout=args.timeout,
                        )
                        choices = row["choices"]
                        labels = choices["label"] if isinstance(choices["label"], list) else list(choices["label"])
                        rows.append(
                            {
                                "id": row["id"],
                                "question": translated["question"],
                                "choices": {"label": labels, "text": translated["choices_text"]},
                                "answerKey": row["answerKey"],
                            }
                        )
                        if args.delay > 0:
                            time.sleep(args.delay)
                else:
                    with Pool(processes=args.workers) as pool:
                        results = list(
                            tqdm(
                                pool.imap(_translate_one_arc, items, chunksize=1),
                                total=n,
                                desc=f"{config_name}/{split}",
                            )
                        )
                    results.sort(key=lambda x: x[0])
                    rows = [r[1] for r in results]
                result[split] = Dataset.from_list(rows)
            result_by_config[config_name] = DatasetDict(result)
        out_ds = result_by_config
        hub_id = "mrsndmn/ai2_arc_ru"
        local_path = "ai2_arc_ru_local"

    if args.no_push:
        if args.dataset == "hellaswag":
            out_ds.save_to_disk(local_path)
            print(f"Saved to {local_path} (no push)")
        else:
            for config_name, ds_dict in out_ds.items():
                path = f"{local_path}_{config_name.replace('-', '_')}"
                ds_dict.save_to_disk(path)
                print(f"Saved {config_name} to {path}")
    else:
        if args.dataset == "hellaswag":
            out_ds.push_to_hub(hub_id)
            print(f"Pushed to {hub_id}")
        else:
            for config_name, ds_dict in out_ds.items():
                ds_dict.push_to_hub(hub_id, config_name=config_name)
            print(f"Pushed to {hub_id} (configs: {list(out_ds.keys())})")


if __name__ == "__main__":
    main()
