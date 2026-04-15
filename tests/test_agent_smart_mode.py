"""Phase 15a — smart_mode 프리셋 테스트.

``smart_mode=True``이면 P0 Phase 기능 전체가 자동 활성화되는지 검증.
"""

from __future__ import annotations

import pytest

from slm_factory.config import AgentRagConfig


class TestSmartModeCascade:
    def test_기본값은_모두_False(self):
        cfg = AgentRagConfig()
        assert cfg.smart_mode is False
        assert cfg.intent_classifier_enabled is False
        assert cfg.clarifier_enabled is False
        assert cfg.personas_enabled is False
        assert cfg.review_work_enabled is False
        assert cfg.planner_enabled is False
        assert cfg.reflector_enabled is False

    def test_smart_mode_True이면_P0_전체_활성(self):
        cfg = AgentRagConfig(smart_mode=True)
        assert cfg.intent_classifier_enabled is True
        assert cfg.clarifier_enabled is True
        assert cfg.personas_enabled is True
        assert cfg.review_work_enabled is True
        assert cfg.planner_enabled is True
        assert cfg.verifier_enabled is True
        assert cfg.reflector_enabled is True
        assert cfg.legacy_fallback_enabled is True

    def test_smart_mode_False이고_개별_True는_존중(self):
        cfg = AgentRagConfig(
            smart_mode=False,
            planner_enabled=True,
            reflector_enabled=True,
        )
        assert cfg.planner_enabled is True
        assert cfg.reflector_enabled is True
        assert cfg.clarifier_enabled is False
        assert cfg.personas_enabled is False

    def test_smart_mode_True는_추가_플래그도_켜짐(self):
        """smart_mode=True + 개별 False → smart_mode가 승리(OR 합산)."""
        cfg = AgentRagConfig(
            smart_mode=True,
            # 개별 False 명시했지만 smart_mode가 True이면 override
            clarifier_enabled=False,
        )
        # smart_mode가 cascade하므로 결과는 True
        assert cfg.clarifier_enabled is True

    def test_P1_P2_플래그는_영향_없음(self):
        """smart_mode는 P0만 활성화 — P1(parallel_steps, session_source_reuse 등) 영향 없음."""
        cfg = AgentRagConfig(smart_mode=True)
        # P1/P2 플래그는 기본값 유지
        # parallel_steps는 기본 False (위험성 있어 opt-in)
        assert cfg.parallel_steps is False
        assert cfg.persist_sessions is False


class TestUltraModeCascade:
    """Phase 15b — ultra_mode는 smart_mode + P1/P2 전체."""

    def test_ultra_mode_True이면_P0_P1_P2_전부_활성(self):
        cfg = AgentRagConfig(ultra_mode=True)
        # P0
        assert cfg.intent_classifier_enabled is True
        assert cfg.clarifier_enabled is True
        assert cfg.personas_enabled is True
        assert cfg.review_work_enabled is True
        assert cfg.planner_enabled is True
        assert cfg.verifier_enabled is True
        assert cfg.reflector_enabled is True
        assert cfg.legacy_fallback_enabled is True
        # P1/P2
        assert cfg.hooks_enabled is True
        assert cfg.memory_compression_enabled is True
        assert cfg.self_improvement_enabled is True
        assert cfg.review_work_retry is True
        assert cfg.session_source_reuse is True

    def test_ultra_mode는_파일기반_기능은_건드리지_않음(self):
        """skills_enabled, custom_personas_dir는 디렉터리 지정 필수이므로 자동 활성화 안 함."""
        cfg = AgentRagConfig(ultra_mode=True)
        assert cfg.skills_enabled is False
        assert cfg.custom_personas_dir == ""
        assert cfg.persist_sessions is False
        assert cfg.parallel_steps is False

    def test_ultra_mode_False에서_개별_True_존중(self):
        cfg = AgentRagConfig(
            ultra_mode=False,
            smart_mode=False,
            hooks_enabled=True,
            self_improvement_enabled=True,
        )
        assert cfg.hooks_enabled is True
        assert cfg.self_improvement_enabled is True
        # P0는 비활성 유지
        assert cfg.planner_enabled is False
