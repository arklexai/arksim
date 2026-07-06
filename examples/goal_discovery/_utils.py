# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from arksim.scenario.goal_discovery.models import GoalDiscoveryResult


def print_results(result: GoalDiscoveryResult) -> None:
    print()
    print("=" * 60)
    print(f"  GOAL DISCOVERY RESULTS  ({result.method})")
    print("=" * 60)
    print(f"  Input conversations : {result.n_input}")
    print(f"  Goals discovered    : {len(result.goals)}")
    meta = result.metadata
    if meta:
        print(f"  Clustering method   : {meta.get('clustering_method', '-')}")
        print(f"  Conversations used  : {meta.get('n_clustered', '-')}")
        print(f"  Noise / unclustered : {meta.get('n_noise', '-')}")
        print(f"  Embedding model     : {meta.get('embedding_model', '-')}")
    print()

    for i, goal in enumerate(result.goals, 1):
        print(f"  Goal {i:02d}  {goal.name}")
        print(f"          {goal.description}")
        neg = goal.negative_emotion_count
        neg_pct = neg / goal.size if goal.size else 0.0
        print(
            f"          prevalence={goal.prevalence:.1%}  size={goal.size}"
            f"  negative={neg} ({neg_pct:.0%})"
        )
        print("          Exemplars:")
        for ex in goal.exemplars:
            short = ex if len(ex) <= 80 else ex[:77] + "..."
            print(f"            · {short}")
        print()

    print("  to_goal_list() output (Stage 2 input):")
    print("  " + "-" * 56)
    for item in result.to_goal_list():
        print(f"  goal       : {item['goal']}")
        print(f"  prevalence : {item['prevalence']:.1%}")
        print()
