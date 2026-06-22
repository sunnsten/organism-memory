from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _get_store():
    from organism.core.config import CoreConfig
    from .api_key_store import ApiKeyStore
    return ApiKeyStore(Path(CoreConfig().db_path))


def cmd_create_key(args):
    store = _get_store()
    raw_key = store.create_key(
        user_id=args.user,
        expires_days=args.expires_days,
    )
    print(f"Created key for user '{args.user}':")
    print(f"  {raw_key}")
    print("(shown once — store it securely)")


def cmd_list_keys(args):
    store = _get_store()
    keys = store.list_keys(args.user)
    if not keys:
        print(f"No keys found for user '{args.user}'")
        return
    import time
    now = int(time.time())
    for k in keys:
        status = "active" if k["active"] else "revoked"
        exp = k["expires_at"]
        if exp and exp < now:
            status = "expired"
        exp_str = f"expires {exp}" if exp else "never expires"
        print(f"  id={k['id']}  {status}  {exp_str}")


def cmd_revoke_key(args):
    store = _get_store()
    found = store.revoke_key(args.key)
    if found:
        print(f"Revoked key: {args.key[:20]}...")
    else:
        print(f"Key not found: {args.key[:20]}...")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="python -m organism.proxy")
    sub = parser.add_subparsers(dest="command")

    p_create = sub.add_parser("create-key", help="Create a new API key")
    p_create.add_argument("--user", required=True, help="User ID")
    p_create.add_argument("--expires-days", type=int, default=None, help="Expiry in days")

    p_list = sub.add_parser("list-keys", help="List keys for a user")
    p_list.add_argument("--user", required=True, help="User ID")

    p_revoke = sub.add_parser("revoke-key", help="Revoke a key")
    p_revoke.add_argument("key", help="Raw key value (sk-organism-...)")

    args = parser.parse_args()

    if args.command == "create-key":
        cmd_create_key(args)
    elif args.command == "list-keys":
        cmd_list_keys(args)
    elif args.command == "revoke-key":
        cmd_revoke_key(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
