"""MCP server ttg-ops: flight status, rebooking options, flight and hotel bookings (synthetic, stateless).

  python -m ttg.mcp.ops_server                                         # stdio (the agent spawns it)
  python -m ttg.mcp.ops_server --transport streamable-http --port 8101  # http://localhost:8101/mcp (Kubernetes, bench)
Each tool is a thin typed wrapper over ttg.world: names, argument types and descriptions match world.TOOL_SPECS.
Value checks (ranges, id formats, voucher kinds) live in ttg.world only, so every transport returns the same
{"error": ...} text the agent can act on.
"""
import os

from mcp.server.fastmcp import FastMCP

from ttg import world
from ttg.mcp._common import run

D = {t["name"]: t["description"] for t in world.TOOL_SPECS}
mcp = FastMCP("ttg-ops", instructions="Tern Air operations on the storm day: flights, rebooking, hotel inventory.",
              log_level=os.environ.get("TTG_MCP_LOG_LEVEL", "WARNING"), stateless_http=True)


@mcp.tool(description=D["flight_status"])
def flight_status(flight_no: str) -> dict:
    return world.execute_tool("flight_status", {"flight_no": flight_no})


@mcp.tool(description=D["rebooking_options"])
def rebooking_options(flight_no: str, party_size: int) -> dict:
    return world.execute_tool("rebooking_options", {"flight_no": flight_no, "party_size": party_size})


@mcp.tool(description=D["book_flight"])
def book_flight(traveler_id: str, flight_no: str, party_size: int) -> dict:
    return world.execute_tool("book_flight", {"traveler_id": traveler_id, "flight_no": flight_no,
                                              "party_size": party_size})


@mcp.tool(description=D["hotel_availability"])
def hotel_availability(airport: str, nights: int, rooms: int) -> dict:
    return world.execute_tool("hotel_availability", {"airport": airport, "nights": nights, "rooms": rooms})


@mcp.tool(description=D["book_hotel"])
def book_hotel(traveler_id: str, hotel_id: str, nights: int, rooms: int) -> dict:
    return world.execute_tool("book_hotel", {"traveler_id": traveler_id, "hotel_id": hotel_id, "nights": nights,
                                             "rooms": rooms})


if __name__ == "__main__":
    run(mcp, 8101)
