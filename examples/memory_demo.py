from __future__ import annotations

import time
from pathlib import Path
from organism.config import OrganismConfig
from organism.core.organism import Organism


def build_organism(db_path: str) -> Organism:
    cfg = OrganismConfig.from_yaml("organism_config.yaml")
    # Point all users to the same DB so we can inspect it directly.
    cfg.base_model.db_path = db_path  # type: ignore[attr-defined]
    return Organism.from_config(cfg)


def main() -> None:
    db = "demo.db"
    user = "demo_user"

    print("=== Session 1: teaching Organism about the user ===\n")
    org = build_organism(db)

    turns = [
        "Hi! My name is Alice and I'm a software engineer at Acme Corp.",
        "I mostly work with Python and Rust. I prefer tabs over spaces.",
        "My favourite editor is Neovim and I use tmux for terminals.",
    ]
    for msg in turns:
        reply = org.chat(user_id=user, user_message=msg, session_id="session1")
        print(f"User : {msg}")
        print(f"Agent: {reply.reply}\n")

    print("Waiting for FactExtractor to flush (up to 5 s)...")
    time.sleep(5)

    memories = org.list_memories(user_id=user, limit=10)
    print(f"Facts stored after session 1: {len(memories)}")
    for m in memories:
        print(f"  - {m['content']}")

    print("\n=== Session 2: new Organism instance, same DB ===\n")
    org2 = build_organism(db)

    query = "What do you know about me?"
    reply2 = org2.chat(user_id=user, user_message=query, session_id="session2")
    print(f"User : {query}")
    print(f"Agent: {reply2.reply}")

    memories2 = org2.list_memories(user_id=user, limit=10)
    print(f"\nFacts visible in session 2: {len(memories2)}")

    # Clean up demo DB
    Path(db).unlink(missing_ok=True)
    print("\nDemo complete. DB cleaned up.")


if __name__ == "__main__":
    main()
