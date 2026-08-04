"""Online reinforcement learning from live trade outcomes.

Flow
----
1. A trade closes (API / broker SL-TP) → ``process_closed_trade``.
2. Composite reward/penalty is scored (quality, RR, process — not PnL alone).
3. Episode + lessons are persisted into the RL knowledge store.
4. Accepted knowledge feeds Engine4 via bar-level ``feat_rl_*`` features,
   sample re-weighting, and the training queue (consumed after each train).
5. ``/api/rl/monitor`` exposes the Learning Monitor for the autotrade UI.
"""

from __future__ import annotations

from atis.shared.rl_learning.knowledge_store import (
    KnowledgeStore,
    consume_rl_for_training,
    delete_episodes,
    episodes_pending_for_training,
    get_monitor_snapshot,
    inject_rl_training_features,
    load_episodes,
    load_policy_weights,
    query_episodes,
    repair_episodes,
    rl_training_context,
)
from atis.shared.rl_learning.rewards import (
    RewardBreakdown,
    score_closed_trade,
)
from atis.shared.rl_learning.service import process_closed_trade

__all__ = [
    "KnowledgeStore",
    "RewardBreakdown",
    "consume_rl_for_training",
    "delete_episodes",
    "episodes_pending_for_training",
    "get_monitor_snapshot",
    "inject_rl_training_features",
    "load_episodes",
    "load_policy_weights",
    "process_closed_trade",
    "query_episodes",
    "repair_episodes",
    "rl_training_context",
    "score_closed_trade",
]
