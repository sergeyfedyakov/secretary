"""Тесты OV-бэкенда: разбор ссылок и ленивую загрузку (без openvino)."""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secretary.engine_ov import OV_ALIASES, OpenVINOEngine, default_ov_device, resolve_ov_model


def test_alias_resolves_to_ov_repo():
    assert resolve_ov_model("large-v3-turbo") == OV_ALIASES["large-v3-turbo"]


def test_repo_and_path_pass_through(tmp_path):
    assert resolve_ov_model("OpenVINO/whisper-small-int8-ov") == "OpenVINO/whisper-small-int8-ov"
    assert resolve_ov_model(str(tmp_path)) == str(tmp_path)


def test_default_ov_device_env(monkeypatch):
    monkeypatch.delenv("SECRETARY_OV_DEVICE", raising=False)
    assert default_ov_device() == "GPU"
    monkeypatch.setenv("SECRETARY_OV_DEVICE", "NPU")
    assert default_ov_device() == "NPU"


def test_engine_takes_device_from_env(monkeypatch):
    monkeypatch.setenv("SECRETARY_OV_DEVICE", "NPU")
    engine = OpenVINOEngine(model="large-v3-turbo")
    assert engine.device == "NPU"
    assert engine.model_ref == OV_ALIASES["large-v3-turbo"]


def test_missing_openvino_raises_hint(monkeypatch):
    if importlib.util.find_spec("openvino_genai") is not None:
        return  # окружение с установленным openvino — проверка неактуальна
    engine = OpenVINOEngine(model="large-v3-turbo")
    try:
        engine._load_model()
    except RuntimeError as exc:
        assert "requirements-openvino" in str(exc)
    else:
        raise AssertionError("ожидалась понятная ошибка без openvino-genai")
