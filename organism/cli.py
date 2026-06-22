from __future__ import annotations

import argparse
import uvicorn


def serve() -> None:
    parser = argparse.ArgumentParser(description="Start the Organism FastAPI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("organism.api.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    serve()
