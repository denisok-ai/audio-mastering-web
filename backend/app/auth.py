"""JWT authentication utilities for Magic Master.

Tokens are signed HS256 JWTs stored in the browser (localStorage).
Secret key is read from env var MAGIC_MASTER_JWT_SECRET (default: dev key, change in production!).
"""
import os
import time
from typing import Optional

try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    jwt = None  # type: ignore[assignment]
    JWTError = Exception  # type: ignore[assignment,misc]
    CryptContext = None  # type: ignore[assignment]

# ─── Config ──────────────────────────────────────────────────────────────────
JWT_SECRET: str = os.environ.get(
    "MAGIC_MASTER_JWT_SECRET",
    "change-me-in-production-use-a-long-random-string-32chars",
)
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = 60 * 60 * 24 * 30  # 30 дней

# ─── Password hashing ────────────────────────────────────────────────────────
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto") if AUTH_AVAILABLE else None


def get_password_hash(password: str) -> str:
    if not AUTH_AVAILABLE or _pwd_ctx is None:
        raise RuntimeError("passlib не установлен — авторизация недоступна")
    return _pwd_ctx.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not AUTH_AVAILABLE or _pwd_ctx is None:
        return False
    return _pwd_ctx.verify(plain_password, hashed_password)


# ─── JWT ─────────────────────────────────────────────────────────────────────
def create_access_token(user_id: int, email: str, tier: str) -> str:
    """Создать JWT токен для пользователя."""
    if not AUTH_AVAILABLE or jwt is None:
        raise RuntimeError("python-jose не установлен — авторизация недоступна")
    payload = {
        "sub": str(user_id),
        "email": email,
        "tier": tier,
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
