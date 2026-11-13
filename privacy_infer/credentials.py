"""Read one local credential without evaluating shell syntax or changing the environment."""
import os
from pathlib import Path
import stat

from .errors import PrivacyError

DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def get_api_key():
    if "DEEPSEEK_API_KEY" in os.environ:
        return os.environ["DEEPSEEK_API_KEY"].strip()
    try:
        fd = os.open(DEFAULT_ENV_PATH, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return ""
    except OSError:
        raise PrivacyError("Cannot read local credential file") from None
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
                raise PrivacyError("Local credential file must be owned by you with permissions 600")
            if info.st_size > 16384:
                raise PrivacyError("Local credential file is too large")
            lines = handle.read(16385).splitlines()
        values = []
        for line in lines:
            name, sep, value = line.strip().partition("=")
            if sep and name.strip() == "DEEPSEEK_API_KEY":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                values.append(value)
        if len(values) > 1:
            raise PrivacyError("Duplicate API key entries in local credential file")
        return values[0] if values else ""
    except (OSError, UnicodeError):
        raise PrivacyError("Cannot read local credential file") from None
