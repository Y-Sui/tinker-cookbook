"""
Coordinator for managing the four-round CANT protocol.
"""

from dataclasses import dataclass, field
from typing import Any

from tinker_cookbook.recipes.cant.parsers import (
    parse_round1_response,
    parse_round2_response,
    parse_round3_response,
    parse_round4_response,
    Round1Response,
    Round2Response,
    Round3Response,
    Round4Response,
)


@dataclass
class CANTCoordinator:
    """
    Coordinates the four-round CANT protocol for a single problem.

    Tracks agent responses across rounds and maintains shared state.

    Rounds:
        0 (Proposal): Agents generate initial solutions
        1 (Critique): Agents provide blind rankings and targeted critiques
        2 (Revision): Agents revise solutions
        3 (Final Verdict): Agents provide final rankings of revised solutions
    """

    question: str
    num_agents: int
    answer: str | None = None  # Optional ground truth (for verifiable tasks)

    # Current round (0, 1, 2, or 3)
    current_round: int = 0

    # Round 0: Initial solutions
    initial_solutions: dict[int, str] = field(default_factory=dict)
    round1_responses: dict[int, Round1Response] = field(default_factory=dict)

    # Round 1: Blind rankings and critiques
    blind_rankings: dict[int, list[tuple[int, str, int]]] = field(default_factory=dict)
    critiques: dict[int, list[int]] = field(default_factory=dict)  # {author: [targets]}
    critique_texts: dict[int, dict[int, str]] = field(default_factory=dict)  # {author: {target: text}}
    round2_responses: dict[int, Round2Response] = field(default_factory=dict)

    # Round 2: Revised solutions
    revised_solutions: dict[int, str] = field(default_factory=dict)
    round3_responses: dict[int, Round3Response] = field(default_factory=dict)
    # Round 3: Final rankings
    final_rankings: dict[int, list[tuple[int, str, int]]] = field(default_factory=dict)
    round4_responses: dict[int, Round4Response] = field(default_factory=dict)

    # Reward computation results (populated after episode ends)
    rewards_computed: bool = False
    bradley_terry_scores_t0: dict[int, float] = field(default_factory=dict)
    bradley_terry_scores_final: dict[int, float] = field(default_factory=dict)
    reward_breakdown: dict[int, dict[str, float]] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Check if all rounds are complete."""
        return self.current_round >= 4

    def get_current_round(self) -> int:
        """Get the current round number (0, 1, 2, or 3)."""
        return self.current_round

    def add_round1_response(self, agent_id: int, response_text: str) -> None:
        """
        Process and store a Round 1 response.

        Args:
            agent_id: ID of the responding agent
            response_text: Raw response text from the agent
        """
        parsed = parse_round1_response(response_text, agent_id)
        self.round1_responses[agent_id] = parsed
        self.initial_solutions[agent_id] = parsed.initial_solution

    def add_round2_response(self, agent_id: int, response_text: str) -> None:
        """
        Process and store a Round 2 response.

        Args:
            agent_id: ID of the responding agent
            response_text: Raw response text from the agent
        """
        parsed = parse_round2_response(response_text, agent_id)
        self.round2_responses[agent_id] = parsed
        self.blind_rankings[agent_id] = parsed.blind_ranking
        self.critiques[agent_id] = parsed.critique_targets
        self.critique_texts[agent_id] = parsed.critique_texts

    def add_round3_response(self, agent_id: int, response_text: str) -> None:
        """
        Process and store a Round 3 response.

        Args:
            agent_id: ID of the responding agent
            response_text: Raw response text from the agent
        """
        parsed = parse_round3_response(response_text, agent_id)
        self.round3_responses[agent_id] = parsed
        self.revised_solutions[agent_id] = parsed.revised_solution

    def add_round4_response(self, agent_id: int, response_text: str) -> None:
        """
        Process and store a Round 4 response.

        Args:
            agent_id: ID of the responding agent
            response_text: Raw response text from the agent
        """
        parsed = parse_round4_response(response_text, agent_id)
        self.round4_responses[agent_id] = parsed
        self.final_rankings[agent_id] = parsed.final_ranking

    def advance_round(self) -> None:
        """Advance to the next round."""
        if self.current_round < 4:
            self.current_round += 1

    def can_advance_round(self) -> bool:
        """
        Check if all agents have responded in the current round.

        Returns:
            True if ready to advance to next round
        """
        if self.current_round == 0:
            return len(self.round1_responses) == self.num_agents
        elif self.current_round == 1:
            return len(self.round2_responses) == self.num_agents
        elif self.current_round == 2:
            return len(self.round3_responses) == self.num_agents
        elif self.current_round == 3:
            return len(self.round4_responses) == self.num_agents
        return False

    def get_initial_solutions(self) -> dict[int, str]:
        """Get all initial solutions (for Round 2 context)."""
        return self.initial_solutions.copy()

    def get_critiques_for_agent(self, agent_id: int) -> dict[int, str]:
        """
        Get all critiques targeting a specific agent.

        Args:
            agent_id: ID of the agent receiving critiques

        Returns:
            Dict mapping {author_id: critique_text} for critiques targeting this agent
        """
        critiques_received = {}
        for author_id, targets in self.critique_texts.items():
            if agent_id in targets:
                critiques_received[author_id] = targets[agent_id]
        return critiques_received

    def compute_rewards(
        self,
        beta_disc: float = 5.0,
        weight_disc: float = 2.0,
        weight_sol: float = 1.0,
        weight_meta: float = 1.0,
        weight_accept: float = 0.5,
    ) -> dict[int, dict[str, float]]:
        """
        Compute all reward components after episode completion.

        Args:
            beta_disc: Scaling factor for persuasion rewards
            weight_disc: Weight for persuasion component
            weight_sol: Weight for solution component
            weight_meta: Weight for consensus component
            weight_accept: Weight for acceptance component

        Returns:
            Dict mapping agent_id to reward breakdown
        """
        from tinker_cookbook.recipes.cant.rewards import compute_all_rewards, combine_rewards

        # Compute all reward components
        results = compute_all_rewards(
            critiques=self.critiques,
            blind_rankings=self.blind_rankings,
            final_rankings=self.final_rankings,
            num_agents=self.num_agents,
            beta_disc=beta_disc,
            bonus_accept=weight_accept,  # Pass bonus value directly
        )

        # Store BT scores for analysis
        for agent_id in range(self.num_agents):
            self.bradley_terry_scores_t0[agent_id] = float(results['v_t0'][agent_id])
            self.bradley_terry_scores_final[agent_id] = float(results['v_final'][agent_id])

        # Combine rewards with weights
        self.reward_breakdown = combine_rewards(
            r_disc=results['r_disc'],
            r_sol=results['r_sol'],
            r_meta=results['r_meta'],
            r_accept=results['r_accept'],
            weight_disc=weight_disc,
            weight_sol=weight_sol,
            weight_meta=weight_meta,
            weight_accept=weight_accept,
        )

        self.rewards_computed = True
        return self.reward_breakdown

    def get_total_reward(self, agent_id: int) -> float:
        """
        Get total reward for an agent (sum of all components).

        Args:
            agent_id: ID of the agent

        Returns:
            Total reward value
        """
        if not self.rewards_computed:
            raise ValueError("Rewards not computed yet. Call compute_rewards() first.")

        breakdown = self.reward_breakdown.get(agent_id, {})
        return sum(breakdown.values())

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of the episode for logging.

        Returns:
            Dict containing episode statistics
        """
        summary = {
            'question': self.question,
            'num_agents': self.num_agents,
            'current_round': self.current_round,
            'is_complete': self.is_complete(),
            'num_initial_solutions': len(self.initial_solutions),
            'num_critiques_total': sum(len(targets) for targets in self.critiques.values()),
            'num_revised_solutions': len(self.revised_solutions),
        }

        if self.rewards_computed:
            summary.update({
                'bt_scores_t0': self.bradley_terry_scores_t0,
                'bt_scores_final': self.bradley_terry_scores_final,
                'total_rewards': {
                    agent_id: self.get_total_reward(agent_id)
                    for agent_id in range(self.num_agents)
                },
            })

        return summary

    def get_debug_info(self) -> str:
        """
        Get detailed debug information about the episode.

        Returns:
            Formatted string with all episode data
        """
        lines = [
            f"CANT Episode Debug Info",
            f"=" * 80,
            f"Question: {self.question}",
            f"Num Agents: {self.num_agents}",
            f"Current Round: {self.current_round}",
            f"",
        ]

        # Round 1
        lines.append("ROUND 1: Initial Solutions")
        lines.append("-" * 80)
        for agent_id in sorted(self.initial_solutions.keys()):
            solution = self.initial_solutions[agent_id]
            lines.append(f"Agent {agent_id}: {solution[:100]}...")
        lines.append("")

        # Round 2
        lines.append("ROUND 2: Blind Rankings and Critiques")
        lines.append("-" * 80)
        for agent_id in sorted(self.blind_rankings.keys()):
            rankings = self.blind_rankings[agent_id]
            critiques = self.critiques.get(agent_id, [])
            lines.append(f"Agent {agent_id}:")
            lines.append(f"  Rankings: {rankings}")
            lines.append(f"  Critiqued: {critiques}")
        lines.append("")

        # Round 3
        lines.append("ROUND 3: Revised Solutions")
        lines.append("-" * 80)
        for agent_id in sorted(self.revised_solutions.keys()):
            solution = self.revised_solutions[agent_id]
            lines.append(f"Agent {agent_id}:")
            lines.append(f"  Revised: {solution[:100]}...")
        lines.append("")

        # Round 4
        lines.append("ROUND 4: Final Rankings")
        lines.append("-" * 80)
        for agent_id in sorted(self.final_rankings.keys()):
            rankings = self.final_rankings[agent_id]
            lines.append(f"Agent {agent_id}:")
            lines.append(f"  Rankings: {rankings}")
        lines.append("")

        # Rewards
        if self.rewards_computed:
            lines.append("REWARDS")
            lines.append("-" * 80)
            for agent_id in range(self.num_agents):
                bt_t0 = self.bradley_terry_scores_t0.get(agent_id, 0.0)
                bt_final = self.bradley_terry_scores_final.get(agent_id, 0.0)
                breakdown = self.reward_breakdown.get(agent_id, {})
                total = self.get_total_reward(agent_id)
                lines.append(f"Agent {agent_id}:")
                lines.append(f"  BT t0: {bt_t0:.3f}, BT final: {bt_final:.3f}")
                lines.append(f"  Breakdown: {breakdown}")
                lines.append(f"  Total: {total:.3f}")
            lines.append("")

        return "\n".join(lines)
