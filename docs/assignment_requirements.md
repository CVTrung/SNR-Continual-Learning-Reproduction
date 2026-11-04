# Assignment requirements and submission checklist

Purpose: connect the course deliverables and the user-provided repository checkpoint to this project's structure. Inputs: the course pages below and the provided Moodle prompt. Output: a checklist of intended evidence, not a claim of completion.

Course pages reviewed on 2026-11-04. Moodle owns the actual dates and any course announcements; none are inferred here.

## Repository-link checkpoint

The supplied Moodle prompt asks one team representative to paste one publicly accessible link to the team's GitHub repository containing its experimental source code. It is an informative checkpoint, not a commit deadline: work may continue until the Final Poster deadline.

Repository URL: https://github.com/CVTrung/SNR-Continual-Learning-Reproduction

The initial repository contains a documented scaffold only. It establishes the project structure, not completed experimental source code or results. The prompt does not specify a minimum code milestone or explicitly approve scaffold-only submissions. Pushing to GitHub does not submit the URL on Moodle. Do not report submission or acceptance without evidence.

## Requirements mapped to files

| Requirement | Owning location / intended evidence | Initial status |
| --- | --- | --- |
| One public team repository with development history | Root README, actual Git commits, public GitHub URL | Public repository verified; initial scaffold only |
| Rebuild the method from the paper description | src/methods/; docs/reproduction_notes.md | Design notes only; no copied author implementation |
| Original datasets, task generation, and models have clear boundaries | src/data/, src/tasks/, src/models/ | Responsibilities documented |
| Full reproduction scope | docs/experiment_coverage.md; configs/experiments/ | Full-scope target mapped; no experiments completed |
| Programmatic Hugging Face/Kaggle dataset preparation | src/data/; scripts/prepare_data.py | Local files present; sources/revisions/loading still to verify |
| Reproducible configurations, seeds, and environment | configs/; requirements.txt; src/tracking.py; src/checkpoint.py | PyTorch/local-Colab proposed; versions/configs still deferred |
| Every run tracked in one public W&B project | src/tracking.py; README experiment-to-evidence table | No verified project or runs recorded |
| Compare against paper numbers and explain deviations | scripts/summarize_results.py; results/; docs/reproduction_notes.md | No measured numbers yet |
| Run author code independently for verification and localize differences | notebooks/author_comparison.ipynb; docs/author_comparison.md | Not executed; method copying prohibited |
| Paper understanding and original visual explanations | docs/paper_understanding.md; future reading/reproduction deck | Outline only; no deck authored |
| At least three strengths, two theoretical limitations, and two linked future directions | docs/critical_analysis.md | Investigation prompts, not conclusions |
| Controlled experiments demonstrating argued limitations | configs/experiments/critique.yaml; docs/empirical_demonstration.md | Not selected or run |
| Actual contributions supported by history/artifacts | docs/contributions.md | Members and assignments not entered |

## Scope decision

This project targets full scope: all main paper experiments on the same or equivalent datasets, with equivalent choices justified. The supplementary scale-256 study is tracked explicitly. A folder or short smoke run does not complete an experiment.

The course also permits core scope: at least the two most important experiments, including the headline comparison, with omitted experiments replaced by an additional team-designed analysis. Switching scope requires an explicit team decision and explanation in the deck; do not silently shrink this full-scope plan.

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
