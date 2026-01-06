"""Custom evaluator for multi-agent debate with dual evaluation modes."""

import asyncio
from collections import defaultdict
from typing import Literal

import tinker

from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.utils import logtree

from .verifiable_env import VerifiableMathProblem, VerifiableMultiAgentEnvGroupBuilder


class MultiAgentDebateEvaluator(SamplingClientEvaluator):
    """Evaluator that runs both direct and debate modes on all problems."""

    def __init__(
        self,
        problems: list[VerifiableMathProblem],
        renderer,
        num_agents: int,
        max_rounds: int,
        history_rounds: int,
        summarize_history: bool,
        summarize_model: str | None,
        log_full_transcript: bool,
        model_name: str,
        grader: Literal["sympy", "math_verify"] = "sympy",
        grade_timeout: float = 2.0,
        max_tokens: int = 8196,
        num_groups_to_log: int = 4,
        max_parallel_evals: int = 0,  # 0 = unlimited parallel, >0 = max concurrent evaluations
    ):
        self.problems = problems
        self.renderer = renderer
        self.num_agents = num_agents
        self.max_rounds = max_rounds
        self.history_rounds = history_rounds
        self.summarize_history = summarize_history
        self.summarize_model = summarize_model
        self.log_full_transcript = log_full_transcript
        self.model_name = model_name
        self.grader = grader
        self.grade_timeout = grade_timeout
        self.max_tokens = max_tokens
        self.num_groups_to_log = num_groups_to_log
        self.max_parallel_evals = max_parallel_evals

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        policy = TinkerTokenCompleter(sampling_client, max_tokens=self.max_tokens)

        # Run both evaluation modes
        # direct_metrics = await self._eval_direct_mode(policy)
        debate_metrics = await self._eval_debate_mode(policy)

        # Combine metrics with mode prefix
        all_metrics = {}
        # all_metrics.update({f"eval/direct/{k}": v for k, v in direct_metrics.items()})
        all_metrics.update({f"eval/debate/{k}": v for k, v in debate_metrics.items()})

        return all_metrics

    async def _run_rollouts_in_batches(self, builders: list, policy, mode: str) -> list:
        """Run rollouts in batches to control concurrency.

        Args:
            builders: List of environment builders
            policy: Policy to use for rollouts
            mode: Evaluation mode name (for logging)

        Returns:
            List of trajectory groups
        """
        if self.max_parallel_evals <= 0:
            print(f"[{mode}] Evaluating {len(builders)} problems in parallel...")
            return await asyncio.gather(
                *[self._run_group_rollout(builder, i, policy) for i, builder in enumerate(builders)]
            )

        # Run in batches
        trajectory_groups = []
        num_batches = (len(builders) + self.max_parallel_evals - 1) // self.max_parallel_evals
        print(f"{len(builders)} problems in {num_batches} batches (size={self.max_parallel_evals})")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.max_parallel_evals
            end_idx = min(start_idx + self.max_parallel_evals, len(builders))
            batch_builders = builders[start_idx:end_idx]
            print(f" Batch {batch_idx + 1}/{num_batches}: problems {start_idx}-{end_idx - 1}")
            batch_trajectory_groups = await asyncio.gather(
                *[
                    self._run_group_rollout(builder, i + start_idx, policy)
                    for i, builder in enumerate(batch_builders)
                ]
            )
            trajectory_groups.extend(batch_trajectory_groups)

        return trajectory_groups

    async def _eval_direct_mode(self, policy) -> dict[str, float]:
        """Evaluate all problems in direct (single-turn) mode."""
        builders = [
            VerifiableMultiAgentEnvGroupBuilder(
                problems=self.problems,
                problem_index=i,
                renderer=self.renderer,
                num_agents=1,  # Single agent for direct eval
                self_play=True,
                history_turns=0,
                summarize_history=False,
                summarize_model=None,
                log_full_transcript=self.log_full_transcript and i < self.num_groups_to_log,
                max_rounds=1,
                model_name=self.model_name,
                grader=self.grader,
                grade_timeout=self.grade_timeout,
                eval_mode="direct",
                is_training=False,
            )
            for i in range(len(self.problems))
        ]

        trajectory_groups = await self._run_rollouts_in_batches(builders, policy, "direct")
        return self._compute_metrics(trajectory_groups, "direct")

    async def _eval_debate_mode(self, policy) -> dict[str, float]:
        """Evaluate all problems in debate (multi-turn) mode."""
        builders = [
            VerifiableMultiAgentEnvGroupBuilder(
                problems=self.problems,
                problem_index=i,
                renderer=self.renderer,
                num_agents=self.num_agents,
                self_play=True,
                history_turns=self.history_rounds,
                summarize_history=self.summarize_history,
                summarize_model=self.summarize_model,
                log_full_transcript=self.log_full_transcript and i < self.num_groups_to_log,
                max_rounds=self.max_rounds,
                model_name=self.model_name,
                grader=self.grader,
                grade_timeout=self.grade_timeout,
                eval_mode="debate",
                is_training=False,
            )
            for i in range(len(self.problems))
        ]

        trajectory_groups = await self._run_rollouts_in_batches(builders, policy, "debate")
        return self._compute_metrics(trajectory_groups, "debate")

    async def _run_group_rollout(self, builder, i, policy):
        """Run rollout with optional logging."""
        enable_logging = i < self.num_groups_to_log
        with logtree.optional_enable_logging(enable=enable_logging):
            return await do_group_rollout(builder, policy)

    def _compute_metrics(self, trajectory_groups, mode: str) -> dict[str, float]:
        """Compute per-dataset metrics and pass@k style metrics."""
        # Group trajectories by dataset
        dataset_to_metrics = defaultdict(list)

        for i, traj_group in enumerate(trajectory_groups):
            problem = self.problems[i]
            dataset_name = problem.dataset_name

            # Extract metrics from each trajectory in the group
            for metrics in traj_group.metrics_G:
                dataset_to_metrics[dataset_name].append(
                    {
                        "problem_idx": i,
                        "format": metrics.get("format", 0.0),
                        "correct": metrics.get("correct", 0.0),
                    }
                )

        # Compute aggregated metrics
        all_metrics = {}

        # Per-dataset metrics
        for dataset_name, metrics_list in dataset_to_metrics.items():
            if metrics_list:
                # Compute mean for each metric
                dataset_metrics = {}
                for key in ["format", "correct"]:
                    values = [m[key] for m in metrics_list]
                    dataset_metrics[key] = sum(values) / len(values) if values else 0.0

                for key, val in dataset_metrics.items():
                    all_metrics[f"{dataset_name}/{key}"] = val

        # Pass@k, avg@k, cons@k metrics for debate mode
        if mode == "debate" and self.num_agents > 1:
            for dataset_name, metrics_list in dataset_to_metrics.items():
                # Group by problem_idx
                problem_to_agents = defaultdict(list)
                for m in metrics_list:
                    problem_to_agents[m["problem_idx"]].append(m["correct"])

                if problem_to_agents:
                    num_problems = len(problem_to_agents)

                    # pass@k: at least one agent got it right
                    pass_at_k = (
                        sum(
                            1
                            for agent_corrects in problem_to_agents.values()
                            if any(c > 0.5 for c in agent_corrects)
                        )
                        / num_problems
                    )
                    all_metrics[f"{dataset_name}/pass@{self.num_agents}"] = pass_at_k

                    # avg@k: average accuracy across all agents
                    # (mean of per-problem average correctness)
                    avg_at_k = (
                        sum(
                            sum(agent_corrects) / len(agent_corrects)
                            for agent_corrects in problem_to_agents.values()
                        )
                        / num_problems
                    )
                    all_metrics[f"{dataset_name}/avg@{self.num_agents}"] = avg_at_k

                    # cons@k: consensus accuracy - majority of agents got it right
                    # (more than half of the agents are correct)
                    cons_at_k = (
                        sum(
                            1
                            for agent_corrects in problem_to_agents.values()
                            if sum(1 for c in agent_corrects if c > 0.5) > len(agent_corrects) / 2
                        )
                        / num_problems
                    )
                    all_metrics[f"{dataset_name}/cons@{self.num_agents}"] = cons_at_k

        return all_metrics
