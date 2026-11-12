# Third-party notices

The MIT license in `LICENSE` covers the code and annotations written by the
OmniCode authors. The following parts of this repository come from other
projects and remain under their own terms.

| Path | Origin | License |
| --- | --- | --- |
| `multiswebench_local/multi_swe_bench/` | Modified copy of [multi-swe-bench](https://github.com/multi-swe-bench/multi-swe-bench); the `style_review/` and `CPP_style_review/` harnesses and other changes were added by the OmniCode authors | Apache-2.0 (see `multiswebench_local/LICENSE`) |
| `multiswebench_local/mswebench_dataset/` | Instances from, or built with the pipeline of, [Multi-SWE-bench](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench) | CC0, subject to the licenses of the underlying projects |
| `SWE-bench/`, `SWE-agent/`, `SWE-ReX/`, `synthetic_datagen/Agentless/` (submodules) | Forks of the respective upstream projects | MIT |
| `multi-swe-bench/` (submodule) | Fork of multi-swe-bench | Apache-2.0 |

Parts of `src/` build on and adapt the SWE-bench evaluation harness
(Copyright (c) 2023 Carlos E Jimenez, John Yang, Kilian Lieret, et al., MIT License).

## AutoCodeRover

The scripts in `legacy/AutoCodeRover/` drive [AutoCodeRover](https://github.com/AutoCodeRoverSG/auto-code-rover),
which is distributed under the SONAR Source-Available License v1.0 (not an
open-source license; non-competitive use only). It is not included in this
repository; see `legacy/AutoCodeRover/README.md` for how to obtain it.

## Benchmark data

Task instances under `data/`, `legacy/data/`, `results/`, `baselines/` and
`multiswebench_local/mswebench_dataset/` contain issue text, patches, tests
and code excerpts taken from third-party open-source repositories (and agent
trajectories that quote them). That material remains under the license of the
repository it was taken from; the MIT license applies only to the OmniCode
authors' own contributions (selection, annotations, generated reviews and
bad patches).
