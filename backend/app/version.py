# Версия приложения и дата сборки.
# Полные правила: doc/VERSIONING.md (SemVer, когда менять MINOR/PATCH, CHANGELOG).
# Кратко: MINOR (0.X.0) — новые функции; PATCH (0.0.X) — багфиксы и мелкие правки.
from datetime import date

__version__ = "0.5.6"
__build_date__ = date.today().isoformat()
