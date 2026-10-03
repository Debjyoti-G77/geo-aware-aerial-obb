"""Project configuration and path helpers.

All relative paths are resolved from the repository root, not from the shell's
current working directory. Directory creation is kept as an explicit operation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("configs/project.yaml")


class ConfigError(RuntimeError):
    """Raised when project configuration cannot be loaded or validated."""


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root by walking upward with ``pathlib``."""
    current = Path(start) if start is not None else Path(__file__)
    current = current.expanduser().resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        has_config = (candidate / DEFAULT_CONFIG_PATH).is_file()
        has_repository_layout = (candidate / "src").is_dir() and (
            candidate / ".git"
        ).exists()
        if has_config or has_repository_layout:
            return candidate

    raise ConfigError(f"Could not locate repository root from: {current}")


def resolve_project_path(
    value: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve a path relative to the repository root unless already absolute."""
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else find_project_root()
    )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def load_project_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load a YAML configuration file without creating files or directories."""
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else find_project_root()
    )
    resolved_config = resolve_project_path(config_path, project_root=root)
    if not resolved_config.is_file():
        raise ConfigError(f"Configuration file not found: {resolved_config}")

    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "PyYAML is required to read the project configuration. "
            "Install the declared project dependencies before running inference."
        ) from exc

    try:
        with resolved_config.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as exc:
        raise ConfigError(f"Could not read configuration: {resolved_config}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML configuration: {resolved_config}") from exc

    if not isinstance(config, dict):
        raise ConfigError(
            f"Configuration root must be a mapping: {resolved_config}"
        )
    return config


def config_value(config: dict[str, Any], *keys: str) -> Any:
    """Return a required nested configuration value."""
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            dotted_key = ".".join(keys)
            raise ConfigError(f"Missing required configuration value: {dotted_key}")
        value = value[key]
    return value


def ensure_directory(path: str | Path) -> Path:
    """Create a directory only when a caller explicitly requests it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
