# Translating Wikipedia through LLMs: workshop notebook

Hands-on notebook for the workshop *Translating Wikipedia through LLMs*.
It goes from the simplest step (looking at one Wikipedia page) to the
automated pipeline, and ends with what is not built yet and how to help.

**Read it on GitHub:** open
[`workshop-walkthrough.ipynb`](workshop-walkthrough.ipynb). Nothing runs
there, but all text, commands and links are readable.

## Run it locally

You need Python 3.10+, `git`, and [Claude Code](https://docs.claude.com/en/docs/claude-code/overview)
(`claude`), logged in. The `claude` cells do not work without it.

```bash
git clone https://github.com/arianit/wiki-translate-workshop
cd wiki-translate-workshop
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab workshop-walkthrough.ipynb
```

VS Code with the Jupyter extension also opens the file directly.

Before running anything else, edit the first code cell ("Setup"): your
contact details for Wikimedia (required by their
[User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy)),
the demo article, and where repos get cloned. Cells that write to the
shared translation queue do nothing unless you set `ALLOW_QUEUE_WRITES`
to `"yes"`.

The notebook creates `out/` and `frwiki-sqwiki-translation/` in the folder
you run it from. Both are git-ignored.

## Present it as slides

```bash
jupyter nbconvert workshop-walkthrough.ipynb --to slides --post serve
```

## The project

| Repo | What it is |
|---|---|
| [enwiki-sqwiki-translation](https://github.com/arianit/enwiki-sqwiki-translation) | the translation skill |
| [wiki-translate-harness](https://github.com/arianit/wiki-translate-harness) | the automated pipeline |
| [wiki-translate-queue](https://github.com/arianit/wiki-translate-queue) | shared list of articles to translate |
| [albanian-language-tech-plan](https://github.com/arianit/albanian-language-tech-plan) | the wider plan for Albanian language technology |

Ideas and problems are welcome as issues on any of them.

## Licence

CC BY-SA 4.0, as for the skill. See [LICENSE](LICENSE).
