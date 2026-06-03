"""TextCraft env + recursive_agent workflow.

Ported from platoon (the recursive agent training framework for stateful
environments). We adapt their TextCraft crafting environment + tasks to
run as an in-process AstraFlow workflow under our SGLang + M2PO + full-FT
infrastructure.

Public entry point: the ``recursive_agent`` workflow, registered via
``@register_workflow`` in ``workflow.py``. Import is done lazily by the
parent ``astraflow.core.workflow.__init__`` rather than here, to avoid
ordering issues during package import.
"""
