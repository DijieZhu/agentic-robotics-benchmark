"""Public client of the aircraft service. The socket accepts only the documented calls."""
import json
import os
import socket

SOCKET = os.environ.get("ROBOT_SOCKET", "/run/visual_robotics/world.sock")


class Client:
    def __init__(self, path=None):
        self.path = path or SOCKET

    def request(self, method, **params):
        with socket.socket(socket.AF_UNIX) as sock:
            sock.settimeout(60)
            sock.connect(self.path)
            sock.sendall((json.dumps({"method": method, "params": params})+"\n").encode())
            with sock.makefile("rb") as stream:
                result = json.loads(stream.readline(4_000_000))
        if "error" in result:
            raise ValueError(result["error"])
        return result

    def observe(self):
        """The current answer again; no time passes."""
        return self.request("observe")

    def act(self, action):
        """[right, forward, up, yaw_left, grip], each in [-1, 1]; 0.2 s of flight."""
        return self.request("act", action=action)

    def new_site(self):
        """Commissioning only: abandon this site and start at the launch point of a new one."""
        return self.request("new_site")
