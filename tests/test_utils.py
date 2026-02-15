"""로깅 유틸리티(utils.py)의 단위 테스트입니다."""

import logging

import slm_factory.utils as utils_module
from slm_factory.utils import get_logger, setup_logging


class TestSetupLogging:
    """setup_logging 함수의 테스트입니다."""

    def test_로거_반환(self):
        """setup_logging이 logging.Logger 인스턴스를 반환하는지 확인합니다."""
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "slm_factory"

    def test_중복_호출_시_basicConfig_한번만_호출(self, mocker):
        """setup_logging을 여러 번 호출해도 basicConfig가 한 번만 호출되는지 확인합니다."""
        # _configured를 False로 리셋하여 테스트 시작
        utils_module._configured = False
        mock_basic = mocker.patch("logging.basicConfig")

        setup_logging()
        setup_logging()
        setup_logging()

        # basicConfig는 최초 한 번만 호출되어야 합니다
        mock_basic.assert_called_once()

        # 테스트 후 상태 복원
        utils_module._configured = True


class TestGetLogger:
    """get_logger 함수의 테스트입니다."""

    def test_올바른_이름의_로거_반환(self):
        """get_logger가 'slm_factory.{name}' 형식의 로거를 반환하는지 확인합니다."""
        logger = get_logger("parsers.text")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "slm_factory.parsers.text"

    def test_다른_이름으로_다른_로거_반환(self):
        """서로 다른 이름으로 호출하면 서로 다른 로거를 반환하는지 확인합니다."""
        logger_a = get_logger("module_a")
        logger_b = get_logger("module_b")
        assert logger_a.name != logger_b.name
        assert logger_a.name == "slm_factory.module_a"
        assert logger_b.name == "slm_factory.module_b"
