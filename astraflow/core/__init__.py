"""Shared building blocks for AstraFlow.

Contains cross-cutting components used by the engine packages
(``raas``, ``train_worker``, ``dataflow``):

- ``config``         — experiment/raas/dataflow/trainer YAML loading
- ``weight_manager`` — weight transport between trainer and RaaS
- ``workflow``       — rollout workflows and reward function registry
"""
