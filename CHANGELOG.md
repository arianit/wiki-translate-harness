# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- `skill_path` now accepts a single directory or a list of directories, to match the upstream skill being split into `enwiki-sqwiki-translation` (translation), `wikiterms`, and `wikiqa`. `skill_loader.load_skill`/`load_skill_cached` concatenate each directory's `SKILL.md` (and, if `include_skill_references` is set, its `references/*.md`) in the given order; reference filenames are namespaced by source skill directory only when more than one path is loaded, so existing single-path configs are unaffected. `config.example.yaml`'s default now lists all three directories.

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