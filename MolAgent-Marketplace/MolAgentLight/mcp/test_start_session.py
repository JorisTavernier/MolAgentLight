"""Quick manual test: call start_training_session via in-process FastMCP client."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

CSV_FILE = "C:/Users/JTaverni/Projects/Data/Caco2_wang.csv"


async def main():
    from fastmcp import Client
    from server import mcp

    async with Client(mcp) as client:
        result = await client.call_tool("start_training_session", {"csv_file": CSV_FILE})
        print(json.dumps(result.data, indent=2))


asyncio.run(main())
