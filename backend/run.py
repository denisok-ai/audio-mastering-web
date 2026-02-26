# @file run.py
# @description Точка входа для запуска сервера (uvicorn)
# @dependencies app.main
# @created 2026-02-26

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
