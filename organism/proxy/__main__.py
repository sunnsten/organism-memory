import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Organism Memory Proxy")
    parser.add_argument("--config", help="Path to organism_proxy.yaml")
    parser.add_argument("--port", type=int, help="Override port from config")
    args, _ = parser.parse_known_args()

    if args.config:
        os.environ["ORGANISM_PROXY_CONFIG"] = args.config

    # Import after env is set
    from .server import create_app
    from .config import ProxyConfig
    from pathlib import Path

    cfg_path = os.environ.get("ORGANISM_PROXY_CONFIG")
    if cfg_path:
        cfg = ProxyConfig.from_yaml(cfg_path)
    else:
        cfg = ProxyConfig()

    port = args.port or cfg.port

    import uvicorn
    uvicorn.run(
        "organism.proxy.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    # Handle CLI sub-commands: create-key, list-keys, revoke-key
    if len(sys.argv) > 1 and sys.argv[1] in ("create-key", "list-keys", "revoke-key"):
        from .cli import main as cli_main
        cli_main()
    else:
        main()
