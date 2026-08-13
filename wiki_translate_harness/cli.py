"""wiki-translate-harness CLI entrypoint. Wires config, sources, and pipeline together."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from wiki_translate_harness.output import sanitize_filename
from wiki_translate_harness.evaluation import save_evaluation_results
from wiki_translate_harness.config import build_config
from wiki_translate_harness.logging_setup import setup_logging
from wiki_translate_harness.mediawiki import MediaWikiClient, wiki_api_url_for_lang
from wiki_translate_harness.progress import ProgressReporter
from wiki_translate_harness.sources import ArticleInput, parse_source_ref, resolve_static_inputs
from wiki_translate_harness.statistics import StatsTracker
from wiki_translate_harness.benchmark import run_benchmark

app = typer.Typer(
    add_completion=False,
    help="Batch-translate Wikipedia articles into a target-language wiki source, "
    "delegating all translation judgment to a configured Pi skill. Source language is "
    "per-title (a `lang:Title` prefix or full Wikipedia URL, e.g. `sq:Gjergj Arianiti`), "
    "falling back to config.yaml's source_lang; target language is a single run-wide setting.",
)

console = Console()


def _build_overrides(
    model: Optional[str],
    workers: Optional[int],
    temperature: Optional[float],
    max_retries: Optional[int],
    cache: Optional[bool],
    validate: Optional[bool],
    repair: Optional[bool],
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict:
    overrides = {
        "model": model,
        "workers": workers,
        "temperature": temperature,
        "max_retries": max_retries,
        "cache": cache,
        "validate": validate,
        "repair": repair,
        "provider": provider,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if base_url is not None:
        # Applies to whichever provider ends up active for this run.
        effective_provider = provider or "openrouter"
        key = "local_base_url" if effective_provider == "local" else "openrouter_base_url"
        overrides[key] = base_url
    return overrides


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    title: Optional[str] = typer.Option(
        None, "--title", help="Article title, optionally `lang:Title` or a full Wikipedia URL"
    ),
    titles: Optional[Path] = typer.Option(
        None, "--titles", help="Text file, one title per line (each may carry its own `lang:Title` prefix)"
    ),
    category: Optional[str] = typer.Option(
        None, "--category", help="Category name to expand into articles, optionally `lang:Category`"
    ),
    file: Optional[Path] = typer.Option(None, "--file", help="Local .wiki/.txt file with raw source wikitext"),
    directory: Optional[Path] = typer.Option(None, "--directory", help="Directory of local .wiki/.txt files"),
    model: Optional[str] = typer.Option(
        None, "--model", help="Model id, e.g. deepseek/deepseek-v3 (OpenRouter) or a local model name"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="'openrouter' (default) or 'local' (any OpenAI-compatible server)"
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Override the active provider's base URL (pair with --provider local)"
    ),
    workers: Optional[int] = typer.Option(None, "--workers", help="Concurrent section-translation workers"),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    max_retries: Optional[int] = typer.Option(None, "--max-retries"),
    cache: Optional[bool] = typer.Option(None, "--cache/--no-cache"),
    validate: Optional[bool] = typer.Option(None, "--validate/--no-validate"),
    repair: Optional[bool] = typer.Option(None, "--repair/--no-repair"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Path to config.yaml"),
    force: bool = typer.Option(False, "--force", help="Re-translate even if output .wiki already exists"),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    if not any([title, titles, category, file, directory]):
        console.print(ctx.get_help())
        raise typer.Exit(code=1)

    overrides = _build_overrides(
        model, workers, temperature, max_retries, cache, validate, repair, provider, base_url
    )
    cfg = build_config(config_path, overrides)

    logger = setup_logging(cfg.log_dir)

    async def _run() -> StatsTracker:
        from wiki_translate_harness.pipeline import run_pipeline  # local import: avoids import cost on --help

        inputs: list[ArticleInput] = resolve_static_inputs(title, titles, file, directory)

        if category:
            cat_lang, cat_name = parse_source_ref(category)
            effective_lang = cat_lang or cfg.source_lang
            api_url = (
                cfg.source_wiki_api
                if effective_lang == cfg.source_lang and cfg.source_wiki_api
                else wiki_api_url_for_lang(effective_lang)
            )
            mw_client = MediaWikiClient(api_url, cfg.user_agent, effective_lang)
            try:
                member_titles = await mw_client.fetch_category_members(cat_name)
            finally:
                await mw_client.aclose()
            inputs.extend(ArticleInput(title=t, source_lang=effective_lang) for t in member_titles)

        logger.info(
            "Starting run: %d article(s) requested, provider=%s, model=%s, workers=%d",
            len(inputs),
            cfg.provider,
            cfg.model,
            cfg.workers,
        )

        stats_tracker = StatsTracker()
        with ProgressReporter(stats_tracker.stats, cfg.workers, console=console) as reporter:
            result = await run_pipeline(
                cfg, inputs, force=force, reporter=reporter, stats_tracker=stats_tracker
            )
        return result

    stats_tracker = asyncio.run(_run())
    stats = stats_tracker.stats
    console.print(
        f"\nDone. Articles completed: {stats.articles_completed}, failed: {stats.articles_failed}, "
        f"skipped: {stats.articles_skipped}. Estimated cost: ${stats.estimated_cost_usd:.4f}. "
        f"Stats written to {cfg.stats_path}"
    )


@app.command()
def benchmark(
    title: Optional[str] = typer.Option(
        None, "--title", help="Article title to benchmark, optionally `lang:Title` or a full Wikipedia URL"
    ),
    file: Optional[Path] = typer.Option(None, "--file", help="Local .wiki/.txt file instead of a title"),
    model: list[str] = typer.Option(
        ..., "--model", help="OpenRouter model id to include; repeat for multiple models"
    ),
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model", help="OpenRouter model id for evaluation (must not be among the evaluated models)"
    ),
    no_evaluation: bool = typer.Option(
        False, "--no-evaluation", help="Skip the evaluation step even if judge-model is provided"
    ),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Path to config.yaml"),
    output: Path = typer.Option(Path("quality"), "--output", help="Directory for benchmark outputs"),
) -> None:
    """Translate the same article with several OpenRouter models and compare
    translation, runtime, token usage, and estimated cost under quality/.
    
    If --judge-model is provided, a separate evaluation will be performed by that
    model (blind, randomized) to rank translations and provide quality scores.
    """
    if not title and not file:
        console.print("[red]Provide --title or --file[/red]")
        raise typer.Exit(code=1)

    cfg = build_config(config_path, {})
    setup_logging(cfg.log_dir)

    if title:
        title_lang, title_text = parse_source_ref(title)
    else:
        title_lang, title_text = None, file.stem.replace("_", " ") if file else ""
    item = ArticleInput(title=title_text, local_path=file, source_lang=title_lang)

    # Validate judge model
    do_evaluation = bool(judge_model) and not no_evaluation
    if do_evaluation and judge_model in model:
        console.print(f"[yellow]Judge model {judge_model} is also among evaluated models; skipping evaluation.[/yellow]")
        do_evaluation = False
    
    benchmark_data = asyncio.run(run_benchmark(cfg, item, model, output, judge_model if do_evaluation else None))
    
    results = benchmark_data["results"]
    source_title = benchmark_data["source_title"]
    source_wikitext = benchmark_data["source_wikitext"]
    translations = benchmark_data["translations"]
    evaluation_result = benchmark_data["evaluation"]
    
    console.print(f"\nBenchmark results for {source_title!r}:")
    for m, r in results.items():
        console.print(
            f"  {m}: {r['runtime_s']:.1f}s, in={r['tokens_in']} out={r['tokens_out']} "
            f"tokens, cost=${r['estimated_cost_usd']:.4f}, failed_sections={r['failed_sections']}"
        )
    console.print(f"Full report: {output / (source_title.replace(' ', '_') + '_benchmark.json')}")
    
    if evaluation_result:
        # Create article subdirectory
        article_dir = output / sanitize_filename(source_title)
        article_dir.mkdir(parents=True, exist_ok=True)
        
        # Save evaluation results and create comparison.md
        save_evaluation_results(
            result=evaluation_result,
            output_dir=article_dir,
            source_title=source_title,
            translations=translations,
            benchmark_results=results,
        )
        
        console.print(f"\nEvaluation saved to {article_dir}/")
        console.print(f"  - Translations: {article_dir}/translations/")
        console.print(f"  - Evaluation report: {article_dir}/evaluation/sonnet_judge.md")
        console.print(f"  - Comparison: {article_dir}/comparison.md")
        
        # Print ranking
        ranking = evaluation_result.ranking
        if ranking:
            ranked_models = [evaluation_result.label_to_model.get(l, l) for l in ranking]
            console.print(f"\nRanking by judge: {' > '.join(ranking)} ({' > '.join(ranked_models)})")
    elif do_evaluation:
        console.print("[yellow]Evaluation was requested but failed or not enough successful translations.[/yellow]")



if __name__ == "__main__":
    app()
