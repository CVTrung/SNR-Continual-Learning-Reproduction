# Assignment requirements and submission checklist

Purpose: connect the course deliverables and the user-provided repository checkpoint to this project's structure. Inputs: the course pages below and the provided Moodle prompt. Output: a checklist of intended evidence, not a claim of completion.

Course pages reviewed on 2026-11-04. Moodle owns the actual dates and any course announcements; none are inferred here.

## Repository-link checkpoint

The supplied Moodle prompt asks one team representative to paste one publicly accessible link to the team's GitHub repository containing its experimental source code. It is an informative checkpoint, not a commit deadline: work may continue until the Final Poster deadline.

Repository URL: https://github.com/CVTrung/SNR-Continual-Learning-Reproduction

The initial repository contains a documented scaffold only. It establishes the project structure, not completed experimental source code or results. The prompt does not specify a minimum code milestone or explicitly approve scaffold-only submissions. Pushing to GitHub does not submit the URL on Moodle. Do not report submission or acceptance without evidence.

## Requirements mapped to files (status 2026-11-21, branch milestone2-core-scope)

| Requirement | Owning location / evidence | Status |
| --- | --- | --- |
| One public team repository with development history | README, Git commits, public GitHub URL | Repository public; core-scope work on branch `milestone2-core-scope` |
| Rebuild the method from the paper description | src/methods/snr.py; docs/reproduction_notes.md | Done; author code read for verification only, nothing copied |
| Clear data / task / model boundaries | src/data/, src/tasks/, src/models/ | Done |
| Reproduction scope (core) | docs/experiment_coverage.md; configs/experiments/{main,shakespeare,sensitivity}.yaml | A, B, C implemented; reduced-protocol runs locally; full protocol prepared in notebooks/colab.ipynb |
| Programmatic Hugging Face/Kaggle dataset preparation | scripts/prepare_data.py; docs/data_sources.md | MNIST from HF (pinned revision, sha256 verified); Shakespeare from the pinned authors' text file (no matching HF copy found; deviation recorded) |
| Reproducible configurations, seeds, environment | configs/experiments/; requirements.txt; src/checkpoint.py; per-run JSON | Done; versions pinned from the tested environment |
| Every run tracked in one public W&B project | src/tracking.py (wandb / trackio / none) | Done: public W&B project `snr-reproduction` (https://wandb.ai/vutrung2903-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/snr-reproduction); all finished runs uploaded (scripts/sync_wandb.py); Colab runs log live |
| Compare against paper numbers and explain deviations | scripts/summarize_results.py; results/tables/; docs/reproduction_notes.md | Done for the reduced protocol |
| Author-code verification | docs/author_comparison.md | Static comparison done; author runs not executed |
| Reading + reproduction deck | deliverables/milestone2/ (build_deck.py, pptx) | Built from docs/7.pptx in English, with speaker notes |
| Critique and empirical demonstration (poster, milestone 3) | docs/critical_analysis.md; configs/experiments/critique.yaml | Not started |
| Actual contributions | docs/contributions.md | To be filled by the team |

## Scope decision

**Core scope**, a team decision recorded 2026-11-21 and explained in the deck. The course allows at least the two most important experiments, including the headline comparison, plus a team-designed analysis in place of the omitted ones. Ours:

- **A, headline:** Table 1, Permuted MNIST, all 8 conditions, SGD and Adam.
- **B:** Permuted Shakespeare, Table 2 and Figures 2-4. It shows that SNR alone is not enough in a Transformer and needs L2.
- **C, team analysis:** a sensitivity sweep of SNR over eta, which tests the paper's robustness claim. It also compares the paper's two estimators (geometric vs histogram).

The reason is compute: no local GPU, and the full five-benchmark matrix would need hundreds of GPU-hours. RM, RC, CI, scaling and generalization are out of scope (see docs/experiment_coverage.md).

## Deliverables

| Deliverable | Required format / content | Weight |
| --- | --- | --- |
| Paper registration | Registered/confirmed selection; status is not established by this repository | Pass gate |
| Reading + reproduction deck | One PPTX, 20-40 slides including appendices, under 20 MB; speaker notes on every slide, at most 500 words each; original visuals; numbered IEEE citations at the bottom of the relevant slides, no separate references slide; results linked to W&B | 45% |
| Critique + empirical-demonstration poster | One PPTX containing one portrait A2 slide; theoretically argued issues connected to measured evidence and W&B runs; defend the work at the poster workshop | 50% |
| Repository, tracking, and artifact discipline | Readable code/docs, development history, reproducibility, public W&B evidence, and required formatting | 5% |

When more than two weaknesses are supported both theoretically and empirically, select the two most notable for the poster and describe the remaining supported weaknesses with evidence in README.md. Do not force an unsupported critique into a claimed finding.

## Check before later submissions

- Confirm paper-registration status and exact Moodle dates.
- Verify anonymous access to the repository and the public W&B evidence actually cited.
- Keep README mappings current: experiment -> script/notebook/config -> completed run -> conclusion.
- Label reduced protocols, incomplete work, reference-paper numbers, and team measurements distinctly.
- Check the deck/poster format and Moodle upload/link entries separately; GitHub publication alone is not Moodle submission.

## Sources

- User-provided Moodle repository-link checkpoint wording in this chat.
- [Project overview and grading](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/).
- [Reading/deck requirements](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reading).
- [Reproduction, scope, data, and tracking rules](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction).
- [Critical-analysis requirements](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/analysis).
- [Empirical demonstration and poster requirements](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/empirics).
