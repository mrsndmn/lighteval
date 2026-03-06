#!/usr/bin/env python3
"""
Translate Rowan/hellaswag validation and test splits to Russian via OpenAI API
and push the result to mrsndmn/hellaswag_ru.

Uses the Responses API with service_tier="flex" for ~50%% cost savings
(https://developers.openai.com/api/docs/guides/flex-processing/).

Usage:
  OPENAI_API_KEY=sk-... python scripts/ru/ru_hellaswag.py [--limit N] [--no-push]
  # Push requires: huggingface-cli login or HF_TOKEN

Options:
  --limit N    Translate only N rows per split (for testing).
  --no-push    Save dataset to disk only (hellaswag_ru_local).
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


def _parse_translation_response(content: str) -> dict[str, Any]:
    content = content.strip()
    if "```" in content:
        parts = content.split("```")
        for p in parts:
            p = p.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                content = p
                break
    out = json.loads(content)
    ctx_ru = out["ctx_a"] + " " + out["ctx_b"]
    return {
        "activity_label": out["activity_label"],
        "ctx_a": out["ctx_a"],
        "ctx_b": out["ctx_b"],
        "ctx": ctx_ru,
        "endings": out["endings"],
    }


def translate_row(
    client: OpenAI,
    row: dict[str, Any],
    model: str,
    service_tier: str = "flex",
    timeout: float = 900.0,
) -> dict[str, Any]:
    """Translate one HellaSwag row's text fields to Russian via Responses API (flex tier supported)."""
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

    # Responses API supports service_tier "flex" (50%% cheaper) and "auto"
    # https://developers.openai.com/api/docs/guides/flex-processing/
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
            # Responses API: https://developers.openai.com/api/docs/guides/flex-processing/
            content = getattr(response, "output_text", None)
            if content is None and hasattr(response, "output"):
                out = response.output
                content = out[0].content if out and hasattr(out[0], "content") else ""
            content = content or ""
            return _parse_translation_response(content)
        except (json.JSONDecodeError, KeyError) as e:
            if attempt == 4:
                raise RuntimeError(f"Translation failed after 5 attempts: {e}") from e
            time.sleep(2.0 * (attempt + 1))
        except Exception as e:
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            if code == 429 and service_tier == "flex" and attempt >= 2:
                time.sleep(5.0 * (attempt + 1))
                try:
                    fallback_kwargs = {**create_kwargs, "service_tier": "auto"}
                    api_client = client.with_options(timeout=timeout) if timeout else client
                    response = api_client.responses.create(**fallback_kwargs)
                    content = getattr(response, "output_text", None)
                    if content is None and hasattr(response, "output"):
                        out = response.output
                        content = out[0].content if out and hasattr(out[0], "content") else ""
                    content = content or ""
                    return _parse_translation_response(content)
                except Exception:
                    pass
            if attempt == 4:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("Unreachable")


def _translate_one(
    item: tuple[int, dict[str, Any], str, str, str, float, str | None, float],
) -> tuple[int, dict[str, Any]]:
    """Worker entry: (index, row, api_key, model, service_tier, timeout, proxy, delay) -> (index, new_row)."""
    idx, row, api_key, model, service_tier, timeout, proxy, delay = item
    http_client = DefaultHttpxClient(proxy=proxy) if proxy else None
    client = OpenAI(api_key=api_key, http_client=http_client)
    translated = translate_row(client, row, model, service_tier=service_tier, timeout=timeout)
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


def main() -> None:
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
    ds = load_dataset("Rowan/hellaswag", split=None, trust_remote_code=True)
    splits = ["validation", "test"]
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
            for idx, row in enumerate(tqdm(data_list, desc=split)):
                translated = translate_row(
                    client,
                    row,
                    args.model,
                    service_tier=args.service_tier,
                    timeout=args.timeout,
                )
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
                rows.append(new_row)
                if args.delay > 0:
                    time.sleep(args.delay)
        else:
            with Pool(processes=args.workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(_translate_one, items, chunksize=1),
                        total=n,
                        desc=split,
                    )
                )
            results.sort(key=lambda x: x[0])
            rows = [r[1] for r in results]
        result[split] = Dataset.from_list(rows)

    out_ds = DatasetDict(result)
    if args.no_push:
        out_ds.save_to_disk("hellaswag_ru_local")
        print("Saved to hellaswag_ru_local (no push)")
    else:
        out_ds.push_to_hub("mrsndmn/hellaswag_ru")
        print("Pushed to mrsndmn/hellaswag_ru")


if __name__ == "__main__":
    main()
