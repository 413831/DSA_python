"""Unit tests for the Voice Dependency Graph DFS/BFS algorithms (`core/voice_graph.py`)."""

import pytest
from core.exceptions import CyclicVoiceDependencyError, VoiceNotFoundError
from core.voice_graph import VoiceDependencyGraph, VoiceProfile


def test_dfs_resolves_topological_dependency_order() -> None:
    graph = VoiceDependencyGraph()
    chain = graph.resolve_dependency_chain_dfs("cloned_hybrid_pro")

    # Base model must be initialized first, and target voice last
    assert chain[0] == "eleven_multilingual_v2"
    assert chain[-1] == "cloned_hybrid_pro"
    assert chain.index("rachel_base") < chain.index("cloned_narrator_v1")
    assert chain.index("cloned_narrator_v1") < chain.index("cloned_hybrid_pro")
    assert chain.index("adam_base") < chain.index("cloned_hybrid_pro")


def test_bfs_level_order_traversal() -> None:
    graph = VoiceDependencyGraph()
    bfs_nodes = graph.get_dependencies_bfs("cloned_narrator_v1")
    assert bfs_nodes == [
        "cloned_narrator_v1",
        "rachel_base",
        "eleven_multilingual_v2",
    ]


def test_cycle_detection_raises_cyclic_dependency_error() -> None:
    cyclic_voices = {
        "voice_a": VoiceProfile("voice_a", "Voice A", dependencies=["voice_b"]),
        "voice_b": VoiceProfile("voice_b", "Voice B", dependencies=["voice_c"]),
        "voice_c": VoiceProfile("voice_c", "Voice C", dependencies=["voice_a"]),
    }
    graph = VoiceDependencyGraph(voices=cyclic_voices)

    with pytest.raises(CyclicVoiceDependencyError, match="Cyclic dependency detected"):
        graph.resolve_dependency_chain_dfs("voice_a")


def test_unknown_voice_raises_not_found() -> None:
    graph = VoiceDependencyGraph()
    with pytest.raises(VoiceNotFoundError):
        graph.resolve_dependency_chain_dfs("non_existent_voice_999")

