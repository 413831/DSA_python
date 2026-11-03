"""Voice dependency resolution using Graph Traversal (BFS / DFS).

Implements:
- Adjacency list representation of voices (`voice_id -> list[base_voice_id]`).
- DFS with 3-state coloring (WHITE/GRAY/BLACK) for cycle detection and topological
  dependency resolution, plus BFS level-order inspection.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set

from core.exceptions import CyclicVoiceDependencyError, VoiceNotFoundError


@dataclass(frozen=True)
class VoiceProfile:
    """Metadata for a voice model and its parent/base voice dependencies."""

    voice_id: str
    name: str
    dependencies: List[str] = field(default_factory=list)
    stability: float = 0.75


# Default adjacency graph simulating ElevenLabs voice library:
# Base foundational models -> Premade voices -> Instant/Professional cloned voices
DEFAULT_VOICE_GRAPH: Dict[str, VoiceProfile] = {
    "eleven_multilingual_v2": VoiceProfile(
        voice_id="eleven_multilingual_v2",
        name="Eleven Multilingual v2 Base",
        dependencies=[],
    ),
    "rachel_base": VoiceProfile(
        voice_id="rachel_base",
        name="Rachel (Premade)",
        dependencies=["eleven_multilingual_v2"],
    ),
    "adam_base": VoiceProfile(
        voice_id="adam_base",
        name="Adam (Premade)",
        dependencies=["eleven_multilingual_v2"],
    ),
    "cloned_narrator_v1": VoiceProfile(
        voice_id="cloned_narrator_v1",
        name="Custom Narrator Clone v1",
        dependencies=["rachel_base"],
    ),
    "cloned_hybrid_pro": VoiceProfile(
        voice_id="cloned_hybrid_pro",
        name="Hybrid Voice Mix (Rachel + Adam)",
        dependencies=["cloned_narrator_v1", "adam_base"],
    ),
    # Voice intentionally configured to fail synthesis in worker to test retries & DLQ
    "unstable_experimental_voice": VoiceProfile(
        voice_id="unstable_experimental_voice",
        name="Experimental Failing Voice",
        dependencies=["eleven_multilingual_v2"],
    ),
}


class VoiceDependencyGraph:
    """Manages voice profiles and resolves their dependency chains via graph traversal."""

    def __init__(self, voices: Dict[str, VoiceProfile] | None = None) -> None:
        # Hash Map / Adjacency List: voice_id -> VoiceProfile
        self._voices: Dict[str, VoiceProfile] = (
            dict(voices) if voices is not None else dict(DEFAULT_VOICE_GRAPH)
        )

    def register_voice(self, profile: VoiceProfile) -> None:
        """Add or update a voice in the adjacency map."""
        self._voices[profile.voice_id] = profile

    def exists(self, voice_id: str) -> bool:
        """Return True if the voice_id is registered in the graph."""
        return voice_id in self._voices

    def resolve_dependency_chain_dfs(self, voice_id: str) -> List[str]:
        """Resolve the ordered initialization chain for `voice_id` using Post-Order DFS.

        Algorithm: Depth-First Search with 3-color cycle detection (O(V + E) time and space).
        - State 0 (unvisited), State 1 (visiting / on current recursion stack), State 2 (visited).
        - If we encounter a node in State 1, a cycle exists -> raises `CyclicVoiceDependencyError`.
        - Appends nodes in post-order so base dependencies appear BEFORE the derived voices
          that depend on them (Topological Sort order).

        Returns:
            List of `voice_id`s from foundational base voice up to the requested `voice_id`.
        """
        if voice_id not in self._voices:
            raise VoiceNotFoundError(f"Voice '{voice_id}' does not exist.")

        state: Dict[str, int] = {}  # 0 = unvisited, 1 = visiting, 2 = visited
        resolution_order: List[str] = []

        def dfs(current_id: str) -> None:
            if current_id not in self._voices:
                raise VoiceNotFoundError(
                    f"Missing dependency voice '{current_id}' required by graph."
                )

            node_state = state.get(current_id, 0)
            if node_state == 1:
                raise CyclicVoiceDependencyError(
                    f"Cyclic dependency detected involving voice '{current_id}'."
                )
            if node_state == 2:
                return

            state[current_id] = 1  # Mark as currently visiting
            for dep_id in self._voices[current_id].dependencies:
                dfs(dep_id)

            state[current_id] = 2  # Mark as completely visited
            resolution_order.append(current_id)

        dfs(voice_id)
        return resolution_order

    def get_dependencies_bfs(self, voice_id: str) -> List[str]:
        """Traverse voice dependencies level-by-level using Breadth-First Search (BFS).

        Algorithm: Queue-based BFS (O(V + E) time) useful for finding shortest dependency
        depth or inspecting immediate parent voices first.
        """
        if voice_id not in self._voices:
            raise VoiceNotFoundError(f"Voice '{voice_id}' does not exist.")

        visited: Set[str] = {voice_id}
        queue: deque[str] = deque([voice_id])
        bfs_order: List[str] = []

        while queue:
            current_id = queue.popleft()
            if current_id not in self._voices:
                raise VoiceNotFoundError(f"Missing dependency voice '{current_id}'.")

            bfs_order.append(current_id)
            for dep_id in self._voices[current_id].dependencies:
                if dep_id not in visited:
                    visited.add(dep_id)
                    queue.append(dep_id)

        return bfs_order

