"""Public client of the robot service. The socket accepts only the documented calls.

    from client import Client, camera_image
    robot = Client()                 # ROBOT_SOCKET, default /run/visual_robotics/world.sock
    obs = robot.observe()            # does not move the robot
    obs = robot.act([.5, 0.])        # UAV translation in metres; returns down-camera frame
    image = camera_image(obs)
    obs = robot.new_site()           # commissioning only: another survey site (UAV only)
    obs = robot.pad()                # commissioning only: a loaded Go1 on the commissioning pad
    obs = robot.advance()            # grading only, one switch to the Go1 after >=4 scout moves
    obs = robot.act({"target_rad": [0., .9, -1.8]*4, "ticks": 12})  # Go1 pose (pad or graded site)
    image = camera_image(robot.look(3))                             # Go1 mast camera (pad or graded site)
"""
import base64
import io
import json
import os
import socket

SOCKET = os.environ.get("ROBOT_SOCKET", "/run/visual_robotics/world.sock")


class Client:
    def __init__(self, path=None, timeout=60):
        self.path = path or SOCKET
        self.timeout = timeout

    def request(self, method, **params):
        with socket.socket(socket.AF_UNIX) as sock:
            sock.settimeout(self.timeout)
            sock.connect(self.path)
            sock.sendall((json.dumps({"method": method, "params": params})+"\n").encode())
            with sock.makefile("rb") as stream:
                result = json.loads(stream.readline(4_000_000))
        if "error" in result:
            raise ValueError(result["error"])
        return result

    def observe(self):
        return self.request("observe")

    def act(self, action):
        return self.request("act", action=action)

    def advance(self):
        return self.request("advance")

    def look(self, detent):
        """One ground RGB exposure; pan CCW in 45-degree detents relative to the trunk."""
        return self.request("look", detent=detent)

    def new_site(self):
        return self.request("new_site")

    def pad(self):
        """Commissioning only: start a pad episode (a loaded Go1 in a bare bay of the hall among bollards)."""
        return self.request("pad")


def camera_image(obs):
    """Decode one public UAV or ground RGB exposure as a (256, 256, 3) array."""
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(obs["images"][0]))).convert("RGB"))
