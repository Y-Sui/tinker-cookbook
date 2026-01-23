"""
Evaluator builders for CANT recipe.

Supports both CANT protocol and baseline evaluation modes using dynamic
task loading from Inspect AI.
"""

import logging

import chz
from inspect_ai import Tasks

from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.eval.inspect_evaluators import InspectEvaluator, InspectEvaluatorBuilder

logger = logging.getLogger(__name__)


@chz.chz
class CANTEvaluatorBuilder(InspectEvaluatorBuilder):
    """
    Evaluator builder that supports CANT protocol or baseline mode.

    This builder dynamically loads any Inspect AI task and optionally wraps
    it with the CANT protocol solver based on the use_cant_protocol flag.

    Config options:
    - task_names: List of Inspect AI task names (e.g., ["inspect_evals/gsm8k", "inspect_evals/math"])
    - use_cant_protocol: If True, wraps tasks with 4-round CANT discussion
    - num_agents: Number of agents for CANT protocol (default: 4)

    Usage:
        # CANT Protocol Mode
        evaluator = CANTEvaluatorBuilder(
            task_names=["inspect_evals/gsm8k", "inspect_evals/math"],
            use_cant_protocol=True,
            num_agents=4,
            model_name="Qwen/Qwen3-8B",
            renderer_name="qwen3",
        )

        # Baseline Mode
        evaluator = CANTEvaluatorBuilder(
            task_names=["inspect_evals/gsm8k", "inspect_evals/math"],
            use_cant_protocol=False,
            model_name="Qwen/Qwen3-8B",
            renderer_name="qwen3",
        )
    """

    # Override parent's tasks field to make it optional (we build it from task_names)
    tasks: Tasks = chz.field(default_factory=list)

    # Benchmark configuration
    task_names: list[str] = chz.field(default_factory=lambda: ["inspect_evals/gsm8k"])

    # CANT protocol vs baseline
    use_cant_protocol: bool = True
    num_agents: int = 4
    use_llm_summarization: bool = True

    def __call__(self) -> SamplingClientEvaluator:
        """Build evaluator with configured tasks."""

        # Always use task names directly - they work with eval_async
        tasks = self.task_names

        # For CANT protocol, we'll pass the solver via eval_async
        # For baseline, we pass no solver (uses task's default)
        config = chz.replace(self, tasks=tasks)
        return CANTInspectEvaluator(
            config,
            use_cant_protocol=self.use_cant_protocol,
            num_agents=self.num_agents,
            use_llm_summarization=self.use_llm_summarization,
        )


class CANTInspectEvaluator(InspectEvaluator):
    """
    Custom evaluator that applies CANT protocol solver to tasks.

    Extends InspectEvaluator to optionally inject the CANT protocol solver
    via the solver parameter in eval_async.
    """

    def __init__(
        self,
        config: InspectEvaluatorBuilder,
        use_cant_protocol: bool = False,
        num_agents: int = 4,
        use_llm_summarization: bool = True,
    ):
        super().__init__(config)
        self.use_cant_protocol = use_cant_protocol
        self.num_agents = num_agents
        self.use_llm_summarization = use_llm_summarization

    async def __call__(self, sampling_client):
        """
        Run evaluation with optional CANT protocol solver.

        If use_cant_protocol is True, injects the CANT protocol solver
        via eval_async's solver parameter.
        """
        import tinker
        from inspect_ai import eval_async
        from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
        from inspect_ai.model import Model as InspectAIModel
        from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

        if self.config.model_name is None:
            raise ValueError("model_name must be set before running evaluation")

        # Create Inspect AI model wrapper
        api = InspectAPIFromTinkerSampling(
            renderer_name=self.config.renderer_name,
            model_name=self.config.model_name,
            sampling_client=sampling_client,
            verbose=self.config.verbose,
        )

        model = InspectAIModel(
            api=api,
            config=InspectAIGenerateConfig(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                num_choices=self.config.num_choices,
                seed=self.config.seed,
            ),
        )

        # Prepare eval_async parameters
        eval_params = {
            "tasks": self.config.tasks,
            "model": [model],
            "limit": self.config.limit,
            "debug_errors": self.config.debug_errors,
            "retry_on_error": 0,
            "fail_on_error": False,
            "log_dir": self.config.log_dir or "~/inspect-logs",
            "max_connections": self.config.max_connections,
            "log_level": self.config.log_level,
            "log_realtime": False,
            "log_buffer": 1000,
        }

        # If CANT protocol mode, inject the solver
        if self.use_cant_protocol:
            from tinker_cookbook.recipes.cant.eval.inspect_tasks import cant_protocol_solver

            logger.info(f"Using CANT protocol solver with {self.num_agents} agents")
            eval_params["solver"] = cant_protocol_solver(
                self.num_agents, use_llm_summarization=self.use_llm_summarization
            )

        # Debug: Log what we're passing to eval_async
        logger.info(f"Passing tasks to eval_async: {eval_params['tasks']}")
        logger.info(f"Tasks type: {type(eval_params['tasks'])}")
        if isinstance(eval_params["tasks"], list) and len(eval_params["tasks"]) > 0:
            logger.info(
                f"First task: {eval_params['tasks'][0]} (type: {type(eval_params['tasks'][0])})"
            )

        # Run evaluation
        results = await eval_async(**eval_params)

        # Extract metrics
        metrics = {}
        for task_result in results:
            if task_result.results is not None and task_result.results.scores is not None:
                for task_name, score in task_result.results.scores[0].metrics.items():
                    if task_result.eval.dataset is not None:
                        dataset_name = task_result.eval.dataset.name
                    else:
                        dataset_name = task_result.eval.task
                    metrics[dataset_name + "/" + task_name] = score.value

        return metrics
