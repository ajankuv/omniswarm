"""OmniSwarm CLI — batch-offload a file of tasks ('orchestrate, don't ingest').

`omniswarm submit tasks.txt` processes one task per line through the engine with
bounded concurrency, writes per-task results to a JSONL file, and prints a one-line
summary. Full results live on disk; only the summary goes to stdout.
"""
import argparse
import asyncio
import json
import sys

import httpx

from omniswarm import engine, store
from omniswarm.config import get_settings


async def run_batch(tasks: list[str], task_type: str, out_path: str, concurrency: int, settings) -> list[dict]:
    store.init_db(settings.db_path)
    timeout = httpx.Timeout(
        connect=settings.connect_timeout, read=settings.read_timeout,
        write=settings.read_timeout, pool=settings.connect_timeout,
    )
    limits = httpx.Limits(max_connections=settings.max_connections)
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def worker(task: str) -> dict:
            async with sem:
                req = engine.JobRequest(
                    task_type=task_type,
                    system="You are a precise, helpful assistant. Do exactly what is asked.",
                    user=task,
                )
                res = await engine.process_job(client, settings, req)
                res["task"] = task
                return res

        for fut in asyncio.as_completed([worker(t) for t in tasks]):
            results.append(await fut)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results


def summarize(results: list[dict]) -> dict:
    return {
        "processed": len(results),
        "passed": sum(1 for r in results if r.get("verdict") == "pass"),
        "escalated": sum(1 for r in results if r.get("verdict") == "escalated"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "tokens_saved": sum(r.get("tokens_saved", 0) for r in results),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omniswarm")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit", help="Batch-process a file of tasks (one per line).")
    s.add_argument("input", help="Path to a text file with one task per line.")
    s.add_argument("--task-type", default="general",
                   help="general | summarize | classify | draft | code | reasoning")
    s.add_argument("--out", default="omniswarm-results.jsonl", help="JSONL output path.")
    s.add_argument("--concurrency", type=int, default=4, help="Max concurrent jobs.")
    args = parser.parse_args(argv)

    if args.cmd == "submit":
        with open(args.input) as f:
            tasks = [line.strip() for line in f if line.strip()]
        if not tasks:
            print("no tasks found in input", file=sys.stderr)
            return 1
        settings = get_settings()
        results = asyncio.run(run_batch(tasks, args.task_type, args.out, args.concurrency, settings))
        s = summarize(results)
        print(
            f"processed {s['processed']} | passed {s['passed']} | "
            f"escalated {s['escalated']} | failed {s['failed']} | "
            f"~{s['tokens_saved']} Claude tokens saved"
        )
        print(f"results written to {args.out}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
