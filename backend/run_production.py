# @file run_production.py
# @description Запуск сервера для production (без --reload, host/port из env)
# @dependencies app.main
# @created 2026-02-26

import os
import uvicorn

if __name__ == "__main__":
    host = os.environ.get("MAGIC_MASTER_HOST", "0.0.0.0")
    port = int(os.environ.get("MAGIC_MASTER_PORT", "8000"))
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
    )
