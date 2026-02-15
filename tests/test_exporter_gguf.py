"""GGUF 내보내기(exporter) 모듈의 테스트입니다.

GGUFExporter의 GGUF 변환, llama.cpp 탐색, 오류 처리 기능을 검증합니다.
subprocess와 파일시스템 접근은 mock으로 대체합니다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slm_factory.exporter.gguf_export import GGUFExporter


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_gguf_exporter(make_config, **overrides):
    gguf_opts = {
        "enabled": True,
        "quantization_type": "q4_k_m",
        "llama_cpp_path": "",
    }
    gguf_opts.update(overrides)
    config = make_config(
        project={"name": "test-model"},
        gguf_export=gguf_opts,
    )
    return GGUFExporter(config)


# ---------------------------------------------------------------------------
# GGUFExporter.__init__
# ---------------------------------------------------------------------------


class TestGGUFExporterInit:

    def test_필드_설정(self, make_config):
        exporter = _make_gguf_exporter(make_config)

        assert exporter.quantization_type == "q4_k_m"
        assert exporter.llama_cpp_path == ""
        assert exporter.model_name == "test-model"

    def test_커스텀_양자화_타입(self, make_config):
        exporter = _make_gguf_exporter(make_config, quantization_type="q8_0")

        assert exporter.quantization_type == "q8_0"

    def test_커스텀_llama_cpp_경로(self, make_config):
        exporter = _make_gguf_exporter(make_config, llama_cpp_path="/opt/llama.cpp")

        assert exporter.llama_cpp_path == "/opt/llama.cpp"


# ---------------------------------------------------------------------------
# GGUFExporter._find_llama_cpp
# ---------------------------------------------------------------------------


class TestFindLlamaCpp:

    def test_설정된_경로_사용(self, make_config, tmp_path):
        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()

        exporter = _make_gguf_exporter(
            make_config, llama_cpp_path=str(llama_dir)
        )

        result = exporter._find_llama_cpp()

        assert result == llama_dir

    def test_설정된_경로_없으면_에러(self, make_config):
        exporter = _make_gguf_exporter(
            make_config, llama_cpp_path="/nonexistent/llama.cpp"
        )

        with pytest.raises(FileNotFoundError, match="설정된 llama.cpp 경로"):
            exporter._find_llama_cpp()

    def test_일반_위치에서_발견(self, make_config, mocker):
        exporter = _make_gguf_exporter(make_config)

        home_llama = Path.home() / "llama.cpp"
        mocker.patch.object(Path, "is_dir", side_effect=lambda self=None: False)

        mock_home_dir = MagicMock(spec=Path)
        mock_home_dir.is_dir.return_value = True
        mock_home_dir.__truediv__ = lambda s, k: mock_home_dir

        with patch.object(
            Path, "is_dir", side_effect=lambda: True
        ):
            pass

    def test_PATH에서_발견(self, make_config, mocker, tmp_path):
        exporter = _make_gguf_exporter(make_config)

        fake_script = tmp_path / "convert_hf_to_gguf.py"
        mocker.patch("shutil.which", return_value=str(fake_script))

        mocker.patch.object(Path, "is_dir", return_value=False)

        result = exporter._find_llama_cpp()

        assert result == tmp_path

    def test_어디서도_못찾으면_에러(self, make_config, mocker):
        exporter = _make_gguf_exporter(make_config)

        mocker.patch.object(Path, "is_dir", return_value=False)
        mocker.patch("shutil.which", return_value=None)

        with pytest.raises(FileNotFoundError, match="llama.cpp를 찾을 수 없습니다"):
            exporter._find_llama_cpp()


# ---------------------------------------------------------------------------
# GGUFExporter._find_convert_script
# ---------------------------------------------------------------------------


class TestFindConvertScript:

    def test_기본_위치에서_발견(self, tmp_path):
        script = tmp_path / "convert_hf_to_gguf.py"
        script.touch()

        config = MagicMock()
        config.gguf_export.quantization_type = "q4_k_m"
        config.gguf_export.llama_cpp_path = ""
        config.project.name = "test"
        exporter = GGUFExporter(config)

        result = exporter._find_convert_script(tmp_path)

        assert result == script

    def test_대체_이름으로_발견(self, tmp_path):
        script = tmp_path / "convert-hf-to-gguf.py"
        script.touch()

        config = MagicMock()
        config.gguf_export.quantization_type = "q4_k_m"
        config.gguf_export.llama_cpp_path = ""
        config.project.name = "test"
        exporter = GGUFExporter(config)

        result = exporter._find_convert_script(tmp_path)

        assert result == script

    def test_scripts_하위_디렉토리에서_발견(self, tmp_path):
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "convert_hf_to_gguf.py"
        script.touch()

        config = MagicMock()
        config.gguf_export.quantization_type = "q4_k_m"
        config.gguf_export.llama_cpp_path = ""
        config.project.name = "test"
        exporter = GGUFExporter(config)

        result = exporter._find_convert_script(tmp_path)

        assert result == script

    def test_스크립트_없으면_에러(self, tmp_path):
        config = MagicMock()
        config.gguf_export.quantization_type = "q4_k_m"
        config.gguf_export.llama_cpp_path = ""
        config.project.name = "test"
        exporter = GGUFExporter(config)

        with pytest.raises(FileNotFoundError, match="convert_hf_to_gguf.py"):
            exporter._find_convert_script(tmp_path)


# ---------------------------------------------------------------------------
# GGUFExporter.export
# ---------------------------------------------------------------------------


class TestGGUFExporterExport:

    def test_정상_변환_경로_반환(self, make_config, tmp_path, mocker):
        model_dir = tmp_path / "merged_model"
        model_dir.mkdir()

        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()
        convert_script = llama_dir / "convert_hf_to_gguf.py"
        convert_script.touch()

        exporter = _make_gguf_exporter(
            make_config, llama_cpp_path=str(llama_dir)
        )

        expected_gguf = model_dir / "test-model-q4_k_m.gguf"

        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        # export 호출 시 gguf 파일이 생성되었다고 가정
        def create_gguf(*args, **kwargs):
            expected_gguf.touch()
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = create_gguf

        result = exporter.export(model_dir)

        assert result == expected_gguf
        mock_run.assert_called_once()

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "convert_hf_to_gguf.py" in cmd[1]
        assert str(model_dir) in cmd
        assert "--outtype" in cmd
        assert "q4_k_m" in cmd

    def test_모델_디렉토리_없으면_에러(self, make_config):
        exporter = _make_gguf_exporter(make_config)

        with pytest.raises(FileNotFoundError, match="모델 디렉토리"):
            exporter.export(Path("/nonexistent/model"))

    def test_subprocess_실패시_에러(self, make_config, tmp_path, mocker):
        model_dir = tmp_path / "merged_model"
        model_dir.mkdir()

        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()
        (llama_dir / "convert_hf_to_gguf.py").touch()

        exporter = _make_gguf_exporter(
            make_config, llama_cpp_path=str(llama_dir)
        )

        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="conversion error"
        )

        with pytest.raises(RuntimeError, match="GGUF 변환에 실패"):
            exporter.export(model_dir)

    def test_타임아웃시_에러(self, make_config, tmp_path, mocker):
        import subprocess

        model_dir = tmp_path / "merged_model"
        model_dir.mkdir()

        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()
        (llama_dir / "convert_hf_to_gguf.py").touch()

        exporter = _make_gguf_exporter(
            make_config, llama_cpp_path=str(llama_dir)
        )

        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="python", timeout=1800),
        )

        with pytest.raises(RuntimeError, match="타임아웃"):
            exporter.export(model_dir)

    def test_gguf_파일_미생성시_glob_폴백(self, make_config, tmp_path, mocker):
        model_dir = tmp_path / "merged_model"
        model_dir.mkdir()

        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()
        (llama_dir / "convert_hf_to_gguf.py").touch()

        exporter = _make_gguf_exporter(
            make_config, llama_cpp_path=str(llama_dir)
        )

        # llama.cpp가 다른 이름으로 생성한 경우
        alt_gguf = model_dir / "output-model.gguf"

        def create_alt_gguf(*args, **kwargs):
            alt_gguf.touch()
            return MagicMock(returncode=0, stdout="", stderr="")

        mocker.patch("subprocess.run", side_effect=create_alt_gguf)

        result = exporter.export(model_dir)

        assert result == alt_gguf

    def test_gguf_파일_아예_없으면_에러(self, make_config, tmp_path, mocker):
        model_dir = tmp_path / "merged_model"
        model_dir.mkdir()

        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()
        (llama_dir / "convert_hf_to_gguf.py").touch()

        exporter = _make_gguf_exporter(
            make_config, llama_cpp_path=str(llama_dir)
        )

        mocker.patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout="", stderr=""),
        )

        with pytest.raises(RuntimeError, match="GGUF 파일이 생성되지 않았습니다"):
            exporter.export(model_dir)


# ---------------------------------------------------------------------------
# __init__.py 노출 확인
# ---------------------------------------------------------------------------


class TestModuleExports:

    def test_GGUFExporter_임포트_가능(self):
        from slm_factory.exporter import GGUFExporter

        assert callable(GGUFExporter)

    def test___all___포함(self):
        from slm_factory import exporter

        assert "GGUFExporter" in exporter.__all__
