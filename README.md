<div align="center" id="astraflowtop">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/title-dark.svg">
  <img src="./docs/assets/title-light.svg" alt="AstraFlow — Dataflow-Oriented Reinforcement Learning for (Multi-)Agentic LLMs" width="620">
</picture>

[![arXiv](https://img.shields.io/badge/arXiv-2605.15565-b31b1b.svg)](https://arxiv.org/abs/2605.15565)
[![Blog](https://img.shields.io/badge/blog-online-9cf.svg)](https://Infini-AI-Lab.github.io/astraflow/)
[![Docs Site](https://img.shields.io/badge/docs-github%20pages-blue)](https://Infini-AI-Lab.github.io/astraflow/docs/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](./LICENSE)

</div>

<hr>

AstraFlow is a **dataflow-oriented** reinforcement learning system designed for better flexibility and scalability.

AstraFlow **natively** supports the following for LLM RL training **without any feature-specific system engineering**:

- **Fully Async Multi-policy collaborative RL**
- **Elastic heterogeneous cross-region rollouts**
- **Substitutable rollout and trainer service**
- **Composable data algorithms**

<br>

<!-- ## What can AstraFlow enable? -->

<div align="center">
<img src="./docs/assets/raas.gif" width="90%" alt="Elastic RaaS pool of mixed-hardware nodes joining and leaving across regions">
</div>

<p align="center"><i><b>Elastic Heterogeneous Cross-region Rollouts</b>: RaaS instances on mixed hardware and across regions join and leave the rollout pool on demand, with no scheduler- or region-specific code.</i></p>

<br>

<div align="center">
<img src="./docs/assets/astraflow.gif" width="90%" alt="AstraFlow training a multi-policy workflow on an elastic, heterogeneous, cross-region rollout pool">
</div>

<p align="center"><i><b>Fully Async Multi-policy Collaborative RL Training</b>: multiple policies train together, each as an independent trainer with its own data and weight stream.</i></p>

<!-- <p align="center"><i>AstraFlow training a multi-policy workflow on an elastic, heterogeneous, cross-region rollout pool — all at once, with no feature-specific code.</i></p> -->

## News
- **[2026/05]** AstraFlow **v0.1.1** released — CUDA 13 image, SGLang 0.5.12, Megatron weight-sync training backend, and transformers 5 support. See the [project website](https://Infini-AI-Lab.github.io/astraflow/).
- **[2026/05]** AstraFlow **v0.1.0** released — first public release of the full system. See the [project website](https://Infini-AI-Lab.github.io/astraflow/).
- **[2026/05]** AstraFlow paper is on [arXiv](https://arxiv.org/abs/2605.15565).

----

## Getting Started

- [Install AstraFlow](https://Infini-AI-Lab.github.io/astraflow/docs/get-started/installation.html)
- [Quick Start](https://Infini-AI-Lab.github.io/astraflow/docs/get-started/quickstart.html)

## Recipes
AstraFlow currently supports the following recipes. Check the [documentation](https://Infini-AI-Lab.github.io/astraflow/docs/) for more detailed instructions.

| Recipe | Description |
|---|---|
| [`math/`](examples/math/) | RLVR math reasoning — Qwen3-1.7B / 8B, M2PO, full and delta-weight transfer |
| [`math-multi-agent/`](examples/math-multi-agent/) | Actor + verifier collaborative math training |
| [`math-efficient-data/`](examples/math-efficient-data/) | Composable data algorithms — GRESO, dynamic sampling, buffer replay |
| [`code/`](examples/code/) | Code-generation RL — Qwen3-8B, M2PO |
| [`code-multi-agent/`](examples/code-multi-agent/) | Codegen + verifier competitive coding |
| [`terminal-bench/`](examples/terminal-bench/) | Harbor-backed Terminal-Bench eval and RL recipes |
| [`search/`](examples/search/) | Search-augmented agent training with local retrieval |
| [`alfworld/`](examples/alfworld/) | ALFWorld embodied household agent |
| [`webshop/`](examples/webshop/) | WebShop web-navigation shopping agent |

## Roadmap
Near-term focus:

- [ ] **Offline cluster training** — Support training on offline clusters without internet access.
- [ ] **All-in-one launcher** — A launcher helper that streamlines bringing up the AstraFlow, RaaS, and trainer services.
- [ ] **MoE model support** — Extend the training backends to Mixture-of-Experts models.
- [ ] **Terminal-Bench training** — Add a recipe for training agents on Terminal-Bench.
- [ ] **Megatron backend** — Add Megatron-LM as a training backend.
- [ ] **vLLM rollout engine** — Support vLLM alongside SGLang as a rollout engine.

## Citation
If you find AstraFlow useful in your research, please cite:

```bibtex
@article{zheng2026astraflow,
  title   = {AstraFlow: Dataflow-Oriented Reinforcement Learning for Agentic LLMs},
  author  = {Zheng, Haizhong and Di, Yizhuo and Wang, Jiahui and Jin, Shuowei and
             Liu, Xueshen and Wu, Yongji and Mao, Z. Morley and Stoica, Ion and
             Zhao, Jiawei and Chen, Beidi},
  journal = {arXiv preprint arXiv:2605.15565},
  year    = {2026}
}
```

## Acknowledgment
We learned the design and reused code from the following projects: [AReaL](https://github.com/areal-project/AReaL), [verl](https://github.com/verl-project/verl), [AgentBench](https://github.com/THUDM/AgentBench), [ASearcher](https://github.com/inclusionAI/ASearcher), and [M2PO](https://github.com/Infini-AI-Lab/M2PO).
