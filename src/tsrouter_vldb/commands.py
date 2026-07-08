from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandPlan:
    command: str
    action: str
    stage: int | None
    reuse: str
    artifact_groups: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "command": self.command,
            "action": self.action,
            "stage": self.stage,
            "reuse": self.reuse,
            "artifact_groups": list(self.artifact_groups),
        }


COMMAND_ARTIFACT_GROUPS = {
    "tsfm": ("tsfm_results_stage20",),
    "profile": ("profile_sources", "tsrouter_core_stage20"),
    "route": ("tsfm_results_stage20", "tsrouter_core_stage20", "task_cache_stage20"),
    "insert": ("tsrouter_core_stage20", "baselines_stage20"),
    "baselines": ("tsfm_results_stage20", "tsrouter_core_stage20", "baselines_stage20", "task_cache_stage20"),
    "summary": ("tsfm_results_stage20", "tsrouter_core_stage20", "baselines_stage20", "tables_figures_stage20"),
}


def build_plan(command: str, action: str, stage: int | None, reuse: str) -> CommandPlan:
    return CommandPlan(
        command=command,
        action=action,
        stage=stage,
        reuse=reuse,
        artifact_groups=COMMAND_ARTIFACT_GROUPS.get(command, ()),
    )
