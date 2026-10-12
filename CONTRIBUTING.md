# Working together

Use the [source quick start](README.md) before changing the project. Keep shared documentation and new interface text in English. Chinese interface translations belong in `xiangqi_lab/static/locales/zh-CN.json`.

## Access and the first change

Connecting a local folder to GitHub enables synchronization. Teammates still need permission to push to the shared repository. Its owner can use **Settings → Collaborators → Add people** and teammates must accept the invitation. Public visibility lets people read a repository; it does not automatically grant write access. See [GitHub's collaborator instructions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository).

Use one branch and one pull request (PR) per task. The shared branch is currently `master`. A PR is a request for a teammate to review and merge your branch into it.

With **GitHub Desktop**:

1. Clone this repository. Before new work, switch to `master` and use Fetch origin / Pull origin to get teammates' changes. Commit existing work on its own branch before switching.
2. Create a new branch, such as `feature/mcts-baseline` or `fix/model-timeout`.
3. Make a focused change and run the relevant checks in the [development guide](docs/DEVELOPMENT.md).
4. Review the Changes list, select only files belonging to the task, and commit with a descriptive summary.
5. Publish the branch / Push origin, then create a pull request targeting `master`. Describe the resulting behavior and how you checked it.
6. Have a teammate review the PR and check its automated test results before merging. Afterwards, pull the updated `master` locally.

The equivalent start from a clean command-line checkout is:

```text
git switch master
git pull --ff-only origin master
git switch -c feature/mcts-baseline
```

After editing and testing, review `git diff` and `git status`, stage specific files, commit, and push your task branch. If you have no write access, fork the repository and submit a PR from your fork.

These are team conventions. Adding this document does not enable GitHub branch protection or prevent direct pushes; the repository owner manages those settings separately.

## Keep changes easy to combine

- Agree on a task in an Issue before overlapping work. Link it in the PR; record the owner and acceptance criteria there.
- Put each policy in its own `xiangqi_lab/models/<name>/` folder with a unique `custom-...` ID. Keep the supplied template runnable for the next teammate.
- Keep training and batch evaluation independent of the browser. Use `rules.py` and `training.py` as the shared environment implementation.
- Discuss observation shapes, action IDs, reward perspective and episode-schema changes before implementing them. Update the API guide and meaningful compatibility tests together.
- Avoid moving the application or engine files as part of an unrelated change: launchers, model paths and the release builder depend on the current layout.
- Preserve existing code style in touched files. Avoid a full-file reformat alongside a behavior change. `.gitattributes` keeps shell scripts on LF and Windows batch scripts on CRLF.

If teammates edited the same lines, resolve the conflict on your task branch, keep the intended changes from both sides, and rerun the affected checks. Do not overwrite the shared branch to remove a conflict.

## What belongs in Git

| Commit | Keep local or share as a separate artifact |
| --- | --- |
| Source, portable model manifests, tests and documentation | `.venv/`, downloaded Python runtimes, machine-specific paths |
| Model dependency instructions, configuration and checkpoint hashes | Large `.pt`, `.pth`, `.onnx`, `.ckpt` and `.safetensors` files |
| Small, deliberate endgame fixtures and experiment summaries | Bulk training data, run logs, videos and generated episodes |
| Existing bundled engine, network and license notices | Rebuilt `dist/` ZIPs and temporary setup files |

Store local experiments under `output/` or `runs/`. Use a `models/local_<name>/` folder for manifests with private machine paths; that convention is ignored by Git but the app still discovers its `model.json`. The web app executes local policy code, so only add code you trust.

Use the [model handoff checklist](docs/MODEL_HANDOFF.md) when sharing a checkpoint. `.gitignore` only affects untracked files; it does not remove something already committed. Review the staged diff before pushing.

## Before review

Run the existing suite for code or interface changes and add a regression test when fixing behavior that needs one. For documentation-only changes, check the instructions, paths and links. Include screenshots for visible interface changes, and record which operating system was actually tested. Mark unfinished experiments as planned, not measured results.

The PR template provides space for checks and limitations. The workflow runs tests; it does not automatically merge, publish a website, upload checkpoints or build release packages.
