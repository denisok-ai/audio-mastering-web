"""JWT authentication utilities for Magic Master.

Tokens are signed HS256 JWTs stored in the browser (localStorage).
Secret key is read from env var MAGIC_MASTER_JWT_SECRET (default: dev key, change in production!).
Хеширование паролей — через bcrypt напрямую (совместимость с bcrypt 4.1+, без passlib).
"""
import os
import time
from typing import Optional

try:
    from jose import JWTError, jwt
    _JOSE_AVAILABLE = True
except ImportError:
    _JOSE_AVAILABLE = False
    jwt = None  # type: ignore[assignment]
    JWTError = Exception  # type: ignore[assignment,misc]

try:
    import bcrypt
    _BCRYPT_AVAILABLE = True
except ImportError:
    bcrypt = None  # type: ignore[assignment]
    _BCRYPT_AVAILABLE = False

AUTH_AVAILABLE = _JOSE_AVAILABLE and _BCRYPT_AVAILABLE

# ─── Config ──────────────────────────────────────────────────────────────────
JWT_SECRET: str = os.environ.get(
    "MAGIC_MASTER_JWT_SECRET",
    "change-me-in-production-use-a-long-random-string-32chars",
)
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = 60 * 60 * 24 * 30  # 30 дней

# bcrypt ограничивает пароль 72 байтами
_BCRYPT_MAX_PASSWORD_BYTES = 72


def _password_bytes(password: str) -> bytes:
    """Пароль в байтах, не длиннее 72 байт (лимит bcrypt)."""
    raw = password.encode("utf-8")
    return raw[:_BCRYPT_MAX_PASSWORD_BYTES] if len(raw) > _BCRYPT_MAX_PASSWORD_BYTES else raw


def get_password_hash(password: str) -> str:
    if not _BCRYPT_AVAILABLE or bcrypt is None:
        raise RuntimeError("bcrypt не установлен — авторизация недоступна")
    pw = _password_bytes(password)
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pw, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not _BCRYPT_AVAILABLE or bcrypt is None:
        return False
    if not hashed_password:
        return False
    pw = _password_bytes(plain_password)
    try:
        stored = hashed_password.encode("utf-8") if isinstance(hashed_password, str) else hashed_password
        return bcrypt.checkpw(pw, stored)
    except Exception:
        return False


# ─── JWT ─────────────────────────────────────────────────────────────────────
def create_access_token(user_id: int, email: str, tier: str, is_admin: bool = False) -> str:
    """Создать JWT токен для пользователя."""
    if not AUTH_AVAILABLE or jwt is None:
        raise RuntimeError("python-jose не установлен — авторизация недоступна")
    payload = {
        "sub": str(user_id),
        "email": email,
        "tier": tier,
        "is_admin": is_admin,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRE_SECONDS,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Декодировать JWT. Возвращает payload или None если невалидный/истёкший."""
    if not AUTH_AVAILABLE or jwt is None:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Извлечь токен из заголовка Authorization: Bearer <token>."""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None
