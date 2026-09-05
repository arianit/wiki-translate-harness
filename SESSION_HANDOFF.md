# Session handoff — 2026-09-05 (updated)

Previous handoff's rename-related restart is done and fully verified. This
update covers what happened in the follow-up session.

## Rename verification — done

- No code files anywhere reference the old `wiki_translate_harness` package
  name; only historical log lines (`logs/errors.log`, `logs/run.log`,
  `translation_stdout.log`) and a couple of cosmetic strings (User-Agent
  strings in `filter_existing.py`/`config.local.yaml`, the GitHub issue link
  in README.md pointing at the still-named-`wiki-translate-harness` GitHub
  repo/remote) still say the old name — none of these affect behavior, left
  as-is.
- `~/code/wiki-translate-queue` was *also* renamed, to
  `~/code/wiki-translation-queue` — this wasn't mentioned in the previous
  handoff but is what `wiki-translation-harness queue`'s
  `--queue-repo-dir` default already correctly points at. Confirmed.
- `.venv` reinstalled via `.venv/bin/python -m pip install -e ".[dev]"`
  (plain `pip`/`python` aren't on PATH in this shell — always invoke via
  `.venv/bin/python` or `.venv/bin/pip` explicitly, not bare `python`).
  `rich._emoji_codes` imports fine; the previously-reported crash there
  didn't reproduce (nothing further needed).
- `pytest -q`: **285 passed** (was 284; +1 from this session's new
  regression test).
- `~/.claude/skills/` symlinks (enwiki-sqwiki-translation, wikiqa,
  wikiterms) — reconfirmed intact, pointing at
  `~/code/enwiki-sqwiki-translation-skills/`.

## Ashoka `table_span_mismatch` — root-caused and fixed (real harness bug, not a translation defect)

Investigated by refetching Ashoka's live English source and re-running
`build_chunks` to reproduce chunk 22 exactly (the "External links" section,
containing the "Edicts of Ashoka" wikitable). Confirmed the *pristine,
untranslated English source itself* triggers `table_span_mismatch` on rows
26–27 of that table — a deliberately narrower 2-row trailing sub-block
(a "Year 26, 27 and later" summary row pair that doesn't fill every column,
common real-world Wikipedia authoring, renders fine, not a syntax error).
Since the defect exists in the source, translation could never have fixed
it — every repair attempt was doomed, permanently failing that chunk.

Fixed in `wiki_translation_harness/validator.py`
(`_check_table_span_mismatches`): now only flags an *isolated* single-row
width anomaly (surrounded on both sides by rows matching the table's
dominant width) — the actual signature of a translation-introduced defect
(a dropped colspan/rowspan) — rather than any row that doesn't match,
which also caught legitimate multi-row asymmetric sub-blocks. Added
`tests/fixtures/table_span_trailing_block_not_flagged.wiki` +
`test_fixture_table_span_trailing_block_not_flagged` (existing
`table_span_mismatch_simple`/`table_span_mismatch_rowspan`/
`table_span_ok_rowspan` fixtures all still pass unchanged — their anomalies
are all isolated single rows). Committed as `4970cf7`, pushed.

## Queue state — reset, but draining is currently blocked (not by anything in this repo)

Both `Nagarjuna` and `Ashoka` reset from `FAILED` back to pending in
`~/code/wiki-translation-queue/totranslate.txt` (commits `5d537ab`,
`116b994`, both pushed) — Nagarjuna's failure was the already-fixed skill
symlink issue, Ashoka's was the validator false positive above.

Tried to actually drain the queue
(`wiki-translation-harness queue --max-articles 2 --provider openrouter
--model deepseek/deepseek-v3.2 --live-validate`) to confirm both now
succeed, but hit **OpenRouter HTTP 402 `in_flight_budget_exhausted`** on
every chunk almost immediately. Root cause: **a separate, pre-existing
`wiki-translation-harness --title Venus` process (PID 94422, started ~18:57,
well before this session's queue command) was already running and consuming
the account's OpenRouter in-flight request budget concurrently** — this is
someone/something else's job, not a bug in the queue drain or the validator
fix. Killed my own queue-drain process (PID 95789) cleanly rather than
compete with or kill that pre-existing job; it hadn't written a result yet,
so the stale `Nagarjuna` `CLAIMED` line was reset back to pending too
(commit `116b994`).

**The Venus job (PID 94422) was still running when this session ended** —
check `ps -p 94422` on resume; if it's finished, the OpenRouter in-flight
budget should have freed up and the queue drain command above should be
retried to confirm Nagarjuna and Ashoka now actually complete successfully
(rather than just no-longer-permanently-failing).

## Immediate next steps

1. Check whether PID 94422 (`--title Venus`) is still running; if not,
   retry draining the queue:
   `wiki-translation-harness queue --max-articles 2 --provider openrouter
   --model deepseek/deepseek-v3.2 --live-validate`
2. Confirm Nagarjuna and Ashoka both complete and get pushed as `DONE` (or
   investigate whatever new failure reason shows up, if any).
3. Nothing else outstanding from the previous handoff — Jupiter was already
   fully delivered/QA'd with nothing pending; that section of the old
   handoff is stale and can be disregarded going forward.
