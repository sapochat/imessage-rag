"""Tests for the persistent settings store."""

import json

import pytest

from src import settings


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Point the settings module at a temp path and reset cache."""
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "_SETTINGS_PATH", settings_path)
    # Reset module-level cache
    settings._cache = None
    settings._cache_mtime = 0.0
    yield settings_path


class TestSettings:
    def test_defaults_to_env_values(self):
        """With no settings file, getters return env/config defaults."""
        backend = settings.get_generation_backend()
        assert backend  # should be non-empty (from config)

    def test_save_and_read(self, isolated_settings):
        settings.save({"generation_backend": "openai", "generation_model": "gpt-4"})

        assert settings.get_generation_backend() == "openai"
        assert settings.get_generation_model() == "gpt-4"

    def test_save_merges(self, isolated_settings):
        settings.save({"generation_backend": "openai"})
        settings.save({"generation_model": "gpt-4o"})

        assert settings.get_generation_backend() == "openai"
        assert settings.get_generation_model() == "gpt-4o"

    def test_file_permissions(self, isolated_settings):
        settings.save({"generation_backend": "ollama"})
        mode = isolated_settings.stat().st_mode & 0o777
        assert mode == 0o600

    def test_rejects_non_localhost_url(self):
        with pytest.raises(ValueError, match="localhost"):
            settings.save({"generation_api_url": "http://api.openai.com/v1"})

    def test_accepts_localhost_url(self, isolated_settings):
        settings.save({"generation_api_url": "http://localhost:8080/v1"})
        assert settings.get_generation_api_url() == "http://localhost:8080/v1"

    def test_get_all(self, isolated_settings):
        settings.save({"generation_backend": "openai", "generation_model": "test-model"})
        all_settings = settings.get_all()
        assert all_settings["generation_backend"] == "openai"
        assert all_settings["generation_model"] == "test-model"
        assert "generation_api_url" in all_settings
        assert "generation_api_key" in all_settings
