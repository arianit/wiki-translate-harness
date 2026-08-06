# wiki-translate-harness

Batch harness that translates Wikipedia articles into a target-language wiki
source (built and tested for English → Albanian sq.wikipedia), via
OpenRouter. All translation judgment is delegated to the
[enwiki-sqwiki-translation](https://github.com/arianit/enwiki-sqwiki-translation)
Pi skill — this harness only fetches, splits, invokes, validates, retries,
caches, verifies facts, estimates cost, and saves. It contains no
translation prompts.

## Example Benchmark: Enji (deity)

A recent benchmark translating the English article **"Enji (deity)"** to Albanian compared four models with blind evaluation by Claude Sonnet 4.5:

**Results:**
- **deepseek/deepseek-v3.2**: 9/10 overall, $0.0620, 220.0 tokens/sec
- **google/gemini-2.5-flash**: 9/10 overall, $0.1284, 621.8 tokens/sec  
- **mistralai/mistral-large**: 8/10 overall, $0.5422, 127.8 tokens/sec
- **qwen/qwen3-235b-a22b**: 6/10 overall, $0.1830, 176.8 tokens/sec

**Judge's ranking (blind):** deepseek/deepseek-v3.2 > google/gemini-2.5-flash > mistralai/mistral-large > qwen/qwen3-235b-a22b

**Total cost:** $1.2825 ($0.9156 translation + $0.3669 evaluation)

**Recommendation:** `deepseek/deepseek-v3.2` offers the best quality/cost balance for English→Albanian Wikipedia translation; `google/gemini-2.5-flash` is 3× faster with identical quality.

*Full results are documented in [GitHub Issue #1](https://github.com/arianit/wiki-translate-harness/issues/1).*

## How skill invocation works

The skill was written to be followed by an interactive agent with shell/tool
access (curl, grep, Write). A single OpenRouter chat-completion call has none
of that. So the harness loads the skill's `SKILL.md` from disk verbatim and
uses it as the system prompt for each per-section OpenRouter call, prefixed
with a small fixed "invocation frame" (see
`wiki_translate_harness/skill_loader.py`) that tells the model it has no
tools/internet in this call and should skip steps that require live
lookups. That frame carries zero translation guidance of its own — every
grammar/vocabulary/convention rule still comes from the skill file.

By default the skill is read from a pinned git revision
(`skill_git_ref: HEAD` in config.yaml) rather than the live working tree, so
local uncommitted edits to the skill's own repo don't silently change
translation behavior mid-run.

Because a single completion call has no tools, the harness does the skill's
live-research steps itself and hands the model verified facts instead (see
**Fact verification** below) — this is what makes the "no tools" limitation
workable in practice.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp config.example.yaml config.yaml   # edit model/workers/skill_path as needed
export OPENROUTER_API_KEY=sk-or-...
export WIKIMEDIA_CONTACT=you@example.com   # or set wikimedia_contact in config.yaml
```

`wikimedia_contact` is required — Wikimedia's
[User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy)
requires automated requests to self-identify with contact info so the
operator can be reached; the harness refuses to start without it (or an
explicit `user_agent` override).

## Usage

```bash
wiki-translate-harness --title "Paris" --model deepseek/deepseek-chat-v3-0324 --workers 4
wiki-translate-harness --titles articles.txt --model qwen/qwen3-235b-a22b
wiki-translate-harness --category "Physics" --model mistralai/mistral-large
wiki-translate-harness --file article.wiki
wiki-translate-harness --directory raw_articles/
```

**Source language is generic, not hardcoded to English.** Target language is
a single run-wide setting (`target_lang` in config.yaml, default `sq`), but
each title can carry its own source wiki — a `lang:Title` prefix
(interwiki-link style) or a full Wikipedia URL, both in `--title` and in
each line of a `--titles` file:

```bash
wiki-translate-harness --title "sq:Gjergj Arianiti"
wiki-translate-harness --title "https://sr.wikipedia.org/wiki/Ниш"
```

With no prefix, `source_lang` from config.yaml is used. `--category` also
accepts a `lang:Category` prefix.

Output is saved as `output/Article_Name.wiki` (UTF-8, no publishing), plus
`output/Article_Name.report.md` (see **Reports** below). Rerun the exact
same command to resume — articles with an existing output file are skipped,
and any chunk already translated is served from the SQLite
translation-memory cache (`cache/translation_memory.sqlite3`) instead of
being re-sent to the model. Pass `--force` to re-translate regardless.

Logs: `logs/run.log` (all activity), `logs/errors.log` (errors only).
Live counters: `stats.json`, updated after every section.

## Fact verification

The skill's own methodology assumes live tool access (batch Wikidata
sitelink checks, curl-based template/parameter checks) that a single
completion call doesn't have. The harness does this work itself
(`wiki_translate_harness/verification.py`) during the planning phase, and
hands the model **verified facts** as plain data alongside each section —
never translation guidance, only what already exists:

- **Link targets**: batch-checked against Wikidata sitelinks; confirmed
  targets are given as `[[X]] -> confirmed as [[Y]]`. For links with no
  target-wiki sitelink, the harness also checks how many *other* Wikipedia
  language editions have an article for the concept — a mechanical proxy
  for "distinct, notable topic worth an interwiki link" vs. "generic word,
  better left unlinked" (e.g. "egg").
- **Templates**: same check, plus — for infobox-shaped templates confirmed
  present — the target wiki's *actual* parameter names, fetched from the
  live template source. Several target-wiki infobox templates keep English
  parameter names with only display labels localized; without this, a model
  will invent translated parameter names that get silently dropped on
  render.
- **Existing target article**: if the article itself already has a
  target-wiki sitelink, this is a rewrite, not a first translation — the
  report flags it prominently, and the existing article's own wikilinks are
  passed along as already-established terminology for the topic.
- **Citation parameter names**: CS1 citation templates (`{{cite web}}`,
  `{{cite book}}`, ...) use fixed English parameter names on essentially
  every Wikipedia regardless of language — the model is told this
  explicitly, since it's easy to mistranslate `|title=` → `|titulli=` etc.,
  which the citation module then silently ignores as unrecognized.

All of this is cached persistently (`cache/verified_facts.sqlite3`) — the
harness's own, growing equivalent of the skill's `sqwiki-verified.md`
reference file, reused across the whole batch and future runs.

## Post-processing (deterministic fixes)

A few defects are common enough, and mechanically fixable enough, that the
harness corrects them after translation rather than relying solely on model
compliance with an instruction:

- **Citation language fill**: any citation missing `|language=` gets one —
  guessed from the title, or (more reliably) by visiting the cited URL and
  reading its declared language. When the two disagree, the title wins
  (academic-publisher/DOI-resolver pages report their own UI language, not
  the cited work's).
- **Citation parameter name fix**: renames known mistranslated CS1 parameter
  names back to English (`|titulli=` → `|title=`, `|botues=` → `|publisher=`,
  etc.), covering inflected and numbered-variant forms.
- **Short-footnote dedup**: `{{sfn}}`/`{{harvnb}}` auto-generate a shared
  anchor from author+year+page. The same source citation split across two
  independently-translated chunks can come back with its `|ps=` quote
  paraphrased slightly differently each time, which breaks that shared
  anchor (MediaWiki requires byte-identical content across all uses). Every
  occurrence sharing an identity is canonicalized to the first one.
- **Leaked commentary detection**: a chunk-level validation check that
  catches the model breaking character — either leaking the skill's
  whole-article "end of file" attribution block into an individual section,
  or fabricating a plausible-looking replacement when given too little real
  content to translate. Treated as a validation failure, triggering the
  same repair-then-fail path as a structural syntax error.

All of these are individually toggleable in config.yaml
(`fill_citation_languages`, `fix_citation_param_names`,
`dedupe_short_footnotes`, `verify_links`).

## Reports

Every completed article gets `output/Article_Name.report.md`: link/template
verification tables (confirmed and not-found, with cross-language notability
counts), confirmed infobox parameters, citation language breakdown, which
sections needed a repair pass, a REWRITE flag with reused terminology if the
target article already exists, and — for English sources — the exact
ready-to-paste `{{Përkthyer nga}}` Talk-page attribution block and edit
summary the skill's own format specifies, built from the real revision ID
and date already captured during fetch.

## Benchmark mode

Compare several models on the same article:

```bash
wiki-translate-harness benchmark --title "Paris" \
  --model deepseek/deepseek-chat-v3-0324 \
  --model qwen/qwen3-235b-a22b \
  --model mistralai/mistral-large \
  --model google/gemini-2.5-flash
```

Writes `quality/<model>/Article_Name.wiki` per model plus
`quality/Article_Name_benchmark.json` with runtime, token usage, and
estimated cost per model.

### Blind evaluation with a judge model

You can optionally have a separate judge model (not among the evaluated ones) rank the translations and provide detailed quality scores:

```bash
wiki-translate-harness benchmark --title "Paris" \
  --model deepseek/deepseek-v3.2 \
  --model qwen/qwen3-235b-a22b \
  --model mistralai/mistral-large \
  --model google/gemini-2.5-flash \
  --judge-model anthropic/claude-sonnet-4.5
```

The judge receives the original English article and the four Albanian translations labeled A‑D in random order, with no indication which model produced which translation. It evaluates each translation on five criteria (translation accuracy, Albanian language quality, terminology quality, MediaWiki quality, publication readiness), assigns scores 1‑10 per category, ranks them, and provides a detailed explanation and recommendation.

Output directory structure:

```
quality/
  Article_Name/
    translations/          # model_a.wiki, model_b.wiki, … (labeled copies)
    evaluation/
      sonnet_judge.md      # full evaluation markdown
      evaluation.json      # structured evaluation data
    comparison.md          # combined benchmark metrics + judge scores
    mapping.json           # which label corresponds to which model
```

Pass `--no-evaluation` to skip the evaluation step even if `--judge-model` is given.

## Reliability

Every network call (OpenRouter, Wikidata, MediaWiki, citation URL fetches)
is wrapped in a hard `asyncio.wait_for` deadline independent of the
underlying HTTP client's own timeout — confirmed necessary in practice, as
httpx's `timeout=` alone did not reliably fire under real network
conditions and could otherwise hang an entire batch run indefinitely on a
single stuck socket.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Known limitations

- The skill's live-research steps that the harness doesn't (yet) replicate
  — nearby-sqwiki-article terminology searches beyond the current article's
  own rewrite target, category-name translation judgment — remain purely
  the model's own judgment call, same as an unverified fact.
- Citation parameter mistranslation and short-footnote reconciliation cover
  the CS1/CS2 template family and the patterns observed in practice; an
  unrecognized Albanian rendering of a parameter name won't be caught until
  added to `ALBANIAN_TO_ENGLISH_CS1_PARAMS`.
