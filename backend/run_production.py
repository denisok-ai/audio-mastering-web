# @file run_production.py
# @description Запуск сервера для production (без --reload, host/port из env)
# @dependencies app.main
# @created 2026-02-26

import os
import uvicorn

if __name__ == "__main__":
    host = os.environ.get("MAGIC_MASTER_HOST", "0.0.0.0")
    port = int(os.environ.get("MAGIC_MASTER_PORT", "8000"))
    # workers=1 обязательно: задачи мастеринга хранятся в памяти процесса (_jobs)
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
        timeout_keep_alive=75,
        timeout_graceful_shutdown=30,
    )
