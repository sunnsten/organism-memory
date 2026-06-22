from __future__ import annotations

import asyncio
import json


def _text(result: object) -> str:
    from mcp.types import TextContent
    item = getattr(result, "content", [None])[0]
    return item.text if isinstance(item, TextContent) else str(item)


async def main() -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command="python",
        args=["-m", "organism.mcp_server", "--config", "configs/mcp_server.yaml"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  {tool.name} — {tool.description}")
            print()

            user = "demo_user"

            # 1. Store an explicit memory
            result = await session.call_tool(
                "organism_remember",
                arguments={"user_id": user, "text": "Alice is a Rust developer who loves functional programming."},
            )
            print("remember →", _text(result))

            # 2. Chat turn
            result = await session.call_tool(
                "organism_chat",
                arguments={
                    "user_id": user,
                    "message": "What programming style do I prefer?",
                    "session_id": "demo_session",
                },
            )
            print("chat →", _text(result))

            # 3. List stored memories
            result = await session.call_tool(
                "organism_list_memories",
                arguments={"user_id": user, "limit": 5},
            )
            memories = json.loads(_text(result))
            print(f"\nStored memories ({len(memories)}):")
            for m in memories:
                print(f"  [{m.get('category', '?')}] {m.get('content', '')}")


if __name__ == "__main__":
    asyncio.run(main())
