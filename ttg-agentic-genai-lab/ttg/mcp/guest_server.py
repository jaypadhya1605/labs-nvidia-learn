"""MCP server ttg-guest: loyalty profiles, policy search, vouchers and receipt reading (synthetic, stateless).

  python -m ttg.mcp.guest_server                                         # stdio
  python -m ttg.mcp.guest_server --transport streamable-http --port 8102

read_receipt answers from the world's ground truth by default (M3-M7). In M8 set TTG_VLM_BASE_URL (+ TTG_VLM_MODEL,
TTG_VLM_API_KEY) and it reads data/receipts/<ref>.png with a real VLM through vlm/extract_receipt.py instead.
"""
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from ttg import world
from ttg.mcp._common import run

D = {t["name"]: t["description"] for t in world.TOOL_SPECS}
mcp = FastMCP("ttg-guest", instructions="Tern guest services: loyalty, policy, vouchers, receipts.",
              log_level=os.environ.get("TTG_MCP_LOG_LEVEL", "WARNING"), stateless_http=True)


@mcp.tool(description=D["loyalty_profile"])
def loyalty_profile(traveler_id: str) -> dict:
    return world.execute_tool("loyalty_profile", {"traveler_id": traveler_id})


@mcp.tool(description=D["policy_search"])
def policy_search(query: str, k: int = 3) -> dict:
    return world.execute_tool("policy_search", {"query": query, "k": k})


@mcp.tool(description=D["issue_voucher"])
def issue_voucher(traveler_id: str, kind: str, amount_usd: float) -> dict:
    return world.execute_tool("issue_voucher", {"traveler_id": traveler_id, "kind": kind, "amount_usd": amount_usd})


def read_receipt_vlm(receipt_ref: str) -> dict:
    """M8: the same tool contract, answered by a vision-language model reading the uploaded image."""
    ref = str(receipt_ref or "").strip()
    if not world.RECEIPT_RE.match(ref):
        return {"error": "receipt_ref must look like R-S0003"}
    path = Path(os.environ.get("TTG_RECEIPT_DIR", str(world.REPO / "data" / "receipts"))) / f"{ref}.png"
    if not path.exists():
        return {"error": f"no uploaded image for {ref} (run: python -m vlm.make_receipts)"}
    from vlm.extract_receipt import extract_fields   # M8 module, imported only when the VLM path is on
    try:
        fields = extract_fields(str(path), base_url=os.environ["TTG_VLM_BASE_URL"],
                                model=os.environ.get("TTG_VLM_MODEL", "ttg-vlm"),
                                api_key=os.environ.get("TTG_VLM_API_KEY", os.environ.get("TTG_API_KEY", "not-used")))
    except Exception as e:      # the agent sees a tool error and can decide what to do
        return {"error": f"receipt could not be read: {type(e).__name__}: {e}"}
    return {"receipt_ref": ref, **fields, "source": "vlm"}


@mcp.tool(description=D["read_receipt"])
def read_receipt(receipt_ref: str) -> dict:
    if os.environ.get("TTG_VLM_BASE_URL"):
        return read_receipt_vlm(receipt_ref)
    return world.execute_tool("read_receipt", {"receipt_ref": receipt_ref})


if __name__ == "__main__":
    run(mcp, 8102)
