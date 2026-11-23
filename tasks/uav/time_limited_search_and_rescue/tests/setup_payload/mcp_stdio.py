"""Minimal MCP stdio bridge; no shell, arbitrary paths, code execution, or reset."""
import json
import sys
from client import Client

TOOLS = [
    {"name": "robot_observe", "description": "The current answer again (two onboard RGB images, sensors, counters); no time passes.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "robot_act", "description": "Move the stabilized UAV by [right,forward,up,yaw_left,grip] for 0.2 seconds.",
     "inputSchema": {"type": "object", "properties": {"action": {"type": "array", "minItems": 5, "maxItems": 5,
                                                                  "items": {"type": "number", "minimum": -1, "maximum": 1}}},
                     "required": ["action"], "additionalProperties": False}},
    {"name": "robot_new_site", "description": "Commissioning only: abandon this site and start at the launch point of a new one.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
]


def main():
    client = Client()
    for line in sys.stdin:
        req = json.loads(line)
        if "id" not in req:
            continue
        try:
            method = req["method"]
            if method == "initialize":
                result = {"protocolVersion": req.get("params", {}).get("protocolVersion", "2024-11-05"),
                          "capabilities": {"tools": {}},
                          "serverInfo": {"name": "visual_robotics", "version": "1.0"}}
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
        except (ValueError, KeyError, OSError) as exc:
            reply = {"jsonrpc": "2.0", "id": req["id"], "error": {"code": -32602, "message": str(exc)}}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
