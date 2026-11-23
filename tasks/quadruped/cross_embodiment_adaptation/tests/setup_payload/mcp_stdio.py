"""Minimal MCP stdio bridge to the robot service; no shell, paths or code execution."""
import json
import sys

from client import Client

TOOLS = [
    {"name": "robot_observe", "description": "Current observation (UAV down-camera on a survey site, Go1 readings on the pad); no motion.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "robot_act", "description": "Survey site: translate the UAV by [dx, dy] metres (norm <= 0.60). Pad: "
                                         "{\"target_rad\": [12 joint angles], \"ticks\": n} holds a Go1 pose.",
     "inputSchema": {"type": "object", "properties": {"action": {}}, "required": ["action"],
                     "additionalProperties": False}},
    {"name": "robot_look", "description": "Pad only: one Go1 mast-camera exposure at pan detent 0..7.",
     "inputSchema": {"type": "object", "properties": {"detent": {"type": "integer", "minimum": 0, "maximum": 7}},
                     "required": ["detent"], "additionalProperties": False}},
    {"name": "robot_new_site", "description": "Commission another survey site (UAV only).",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "robot_pad", "description": "Start a commissioning-pad episode: a loaded Go1 in a bare bay of the hall among bollards.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
]


def main():
    client = Client()
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        if not isinstance(req, dict) or "id" not in req:
            continue
        try:
            method = req["method"]
            if method == "initialize":
                result = {"protocolVersion": req.get("params", {}).get("protocolVersion", "2024-11-05"),
                          "capabilities": {"tools": {}},
                          "serverInfo": {"name": "cross_embodiment_robot", "version": "9.0"}}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                name = req["params"]["name"]
                if name not in {t["name"] for t in TOOLS}:
                    raise ValueError("unknown tool")
                obs = client.request(name.removeprefix("robot_"), **req["params"].get("arguments", {}))
                images = obs.pop("images", [])
                result = {"content": [{"type": "text", "text": json.dumps(obs)}] +
                          [{"type": "image", "data": x, "mimeType": "image/png"} for x in images]}
            elif method == "ping":
                result = {}
            else:
                raise ValueError("unsupported method")
            reply = {"jsonrpc": "2.0", "id": req["id"], "result": result}
        except (ValueError, KeyError, TypeError, OSError) as exc:
            reply = {"jsonrpc": "2.0", "id": req["id"], "error": {"code": -32602, "message": str(exc)}}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
