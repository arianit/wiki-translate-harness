# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- `skill_path` now accepts a single directory or a list of directories, to match the upstream skill being split into `enwiki-sqwiki-translation` (translation), `wikiterms`, and `wikiqa`. `skill_loader.load_skill`/`load_skill_cached` concatenate each directory's `SKILL.md` (and, if `include_skill_references` is set, its `references/*.md`) in the given order; reference filenames are namespaced by source skill directory only when more than one path is loaded, so existing single-path configs are unaffected. `config.example.yaml`'s default now lists all three directories.
- **`wikiqa` (the pre-delivery QA checklist) no longer rides along on every normal translation/repair call.** It's now loaded separately via a new `qa_skill_path` config field and appended (`skill_loader.build_repair_messages`'s new `qa_skill` parameter) only to a repair call — the one case where `validate_wikitext` has already found a real defect. `skill_path`'s default dropped `wikiqa`, keeping only `enwiki-sqwiki-translation` + `wikiterms`. Measured against the real skill files: system-prompt tokens on an ordinary translation call drop ~29% (32.4k → 22.9k, tiktoken cl100k_base estimate); a live `claude_code`-provider translation of a real enwiki lead section (Sustainable architecture) showed the same shape live (54.9k → 40.9k input tokens per the CLI's own usage reporting) with both before/after outputs passing static validation and reading as equally fluent, faithful Albanian. `translate_chunk`/`repair_chunk`/`run_assembly_repair`/benchmark mode all thread the new `qa_skill` parameter through; existing single-skill-path configs without `wikiqa` are unaffected.
- **New compact, growing article-level terminology registry** (`VerifiedFacts.established_renderings` in `verification.py`) narrows one remaining cross-section consistency gap: for a source-language term with no confirmed target-wiki sitelink, the first chunk to translate it typically renders it as `{{ill|display|en|Title}}` — that specific rendering is now recorded (first occurrence wins, mechanically parsed from the model's own output, no translation judgment invented by the harness) and surfaced in `build_verified_facts_block` to every later chunk of the same article mentioning the same term ("already rendered elsewhere in this article as ... — reuse that exact rendering"), instead of each chunk independently reinventing its own phrasing. Confirmed against a real model output using this exact pattern (`{{ill|Shtëpia me energji pozitive|en|Energy-plus-house|lt=...}}`).

## [0.2.0] - 2026-08-06

### Added
- Blind evaluation with judge model in benchmark mode (`--judge-model`)
- New `evaluation.py` module for judge-model quality assessment
- Randomized labeling (A-D) with structured JSON output
- Five evaluation criteria: translation accuracy, Albanian language quality, terminology quality, MediaWiki quality, publication readiness
- Integration with existing benchmark pipeline
- Example benchmark results for "Enji (deity)" article

### Changed
- Updated `benchmark.py` to return translations and support evaluation
- Updated `cli.py` with `--judge-model` and `--no-evaluation` options
- Enhanced README with example benchmark and usage instructions

### Removed
- Test files `albanian_mythology`, `albanian_mythology.missing`, `filter_existing.py` from repository (moved to .gitignore)

## [0.1.0] - 2026-08-04

### Added
- Initial release of wiki-translation-harness
- Batch translation of Wikipedia articles via OpenRouter
- Delegation to enwiki-sqwiki-translation Pi skill
- Translation memory caching
- Fact verification (Wikidata + target wiki)
- Post-processing fixes (citation language, parameter names, short-footnote dedup)
- Report generation
- Benchmark mode for comparing multiple models