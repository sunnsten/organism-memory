from __future__ import annotations

import logging


def run(
    conn,
    model_path: str,
    n_gpu_layers: int,
    n_ctx: int,
    verbose: bool,
) -> None:
    """Entry point for the worker subprocess. Blocks until connection closes."""
    logging.basicConfig(level=logging.WARNING)

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        from llama_cpp import Llama

        model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
        )
        metadata = dict(model.metadata or {})
    except Exception as exc:
        try:
            conn.send({"status": "error", "error": str(exc)})
        except Exception:
            pass
        conn.close()
        return

    conn.send({"status": "ready", "metadata": metadata})

    # ── Request loop ──────────────────────────────────────────────────────────
    while True:
        try:
            req = conn.recv()
        except (EOFError, OSError):
            break  # parent closed the connection

        if req is None or req.get("type") == "shutdown":
            break

        req_type = req.get("type")
        try:
            if req_type == "generate":
                result = model.create_chat_completion(
                    messages=req["messages"],
                    max_tokens=req["max_tokens"],
                    temperature=req["temperature"],
                    top_p=req["top_p"],
                    stream=False,
                )
                text = (result["choices"][0]["message"]["content"] or "").strip()  # type: ignore[index]
                conn.send({"status": "ok", "text": text})

            elif req_type == "tokenize":
                try:
                    tokens = model.tokenize(
                        req["text"].encode("utf-8"), add_bos=False, special=False
                    )
                except TypeError:
                    tokens = model.tokenize(req["text"].encode("utf-8"), add_bos=False)
                conn.send({"status": "ok", "count": len(tokens)})

            else:
                conn.send({"status": "error", "error": f"unknown request type: {req_type!r}"})

        except Exception as exc:
            try:
                conn.send({"status": "error", "error": str(exc)})
            except Exception:
                break  # pipe broken — exit cleanly

    conn.close()
