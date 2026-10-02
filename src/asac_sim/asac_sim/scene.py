"""Actual rasterized RGB + z-buffer depth via Bullet TinyRenderer; no model masks."""

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def robot_description(asset_directory, for_rviz=False):
    asset_directory = Path(asset_directory).resolve()
    text = (asset_directory / "piper_description.urdf").read_text()
    root = ET.fromstring(text)
    child_links = {j.find("child").attrib["link"] for j in root.findall("joint")}
    roots = [
        link.attrib["name"]
        for link in root.findall("link")
        if link.attrib["name"] not in child_links
    ]
    if len(roots) != 1:
        raise ValueError("PiPER URDF must have one root link")
    # Resolve package paths to installed, vendored mesh files, for Bullet and RViz.
    for mesh in root.iter("mesh"):
        uri = mesh.attrib["filename"]
        prefix = "package://piper_description/"
        if not uri.startswith(prefix):
            raise ValueError(f"Unexpected mesh URI: {uri}")
        mesh.attrib["filename"] = str(asset_directory / uri[len(prefix):])
        if not Path(mesh.attrib["filename"]).is_file():
            raise ValueError(f"Missing mesh: {uri}")
        if for_rviz:
            mesh.attrib["filename"] = Path(mesh.attrib["filename"]).as_uri()
    return ET.tostring(root, encoding="unicode"), roots[0]


def apple_mesh(path, radius, lobes=0.04):
    """Procedural apple with shallow top/bottom dimples. Authored here, Apache-2.0."""
    rings, segments = 40, 64
    vertices, faces = [], []
    for j in range(rings + 1):
        theta = np.pi * j / rings
        for i in range(segments + 1):
            phi = 2 * np.pi * i / segments
            rr = radius * np.sin(theta) * (1 + lobes * np.cos(5 * phi) * np.sin(theta) ** 2)
            z = radius * (
                0.94 * np.cos(theta)
                - 0.16 * np.exp(-((theta / 0.28) ** 2))
                + 0.09 * np.exp(-(((theta - np.pi) / 0.28) ** 2))
            )
            vertices.append((rr * np.cos(phi), rr * np.sin(phi), z))
    for j in range(rings):
        for i in range(segments):
            a = j * (segments + 1) + i
            b = a + segments + 1
            faces.extend(((a, b, a + 1), (a + 1, b, b + 1)))
    with Path(path).open("w") as f:
        for x, y, z in vertices:
            f.write(f"v {x} {y} {z}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


class Scene:
    def __init__(self, config, asset_directory):
        global p
        import pybullet as p

        camera = config["camera"]
        q = np.asarray(camera["optical_quaternion_xyzw"], float)
        if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-6:
            raise ValueError("Camera optical quaternion must be finite and unit length")
        if not (
            config["rate_hz"] > 0
            and camera["width"] > 0
            and camera["height"] > 0
            and 0 < camera["vertical_fov_deg"] < 179
            and 0 < camera["near_m"] < camera["far_m"]
        ):
            raise ValueError("Invalid camera geometry or simulator rate")

        self.config = config
        self.client = p.connect(p.DIRECT)
        self.temporary = tempfile.TemporaryDirectory(prefix="asac-scene-")
        self.description, self.base_frame = robot_description(asset_directory)
        urdf = Path(self.temporary.name) / "piper.urdf"
        urdf.write_text(self.description)
        self.robot = p.loadURDF(
            str(urdf),
            useFixedBase=True,
            basePosition=config["robot_position"],
            physicsClientId=self.client,
        )
        self.joints = []
        for i in range(p.getNumJoints(self.robot, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.client)
            if info[2] != p.JOINT_FIXED:
                name = info[1].decode()
                position = float(config.get("joint_positions", {}).get(name, 0))
                p.resetJointState(self.robot, i, position, physicsClientId=self.client)
                self.joints.append((name, position))
        self.objects = []
        self._box(
            "table", config["table"]["position"], config["table"]["size"], [0.6, 0.5, 0.38, 1]
        )
        self._box("tray", config["tray"]["position"], config["tray"]["size"], [0.8, 0.8, 0.75, 1])
        tray = config["tray"]
        x, y, z = tray["position"]
        sx, sy, sz = tray["size"]
        for dy in (-sy / 2, sy / 2):
            self._box("tray_wall", [x, y + dy, z + 0.012], [sx, 0.007, 0.025], [0.7, 0.7, 0.65, 1])
        for dx in (-sx / 2, sx / 2):
            self._box("tray_wall", [x + dx, y, z + 0.012], [0.007, sy, 0.025], [0.7, 0.7, 0.65, 1])
        self.apples = {}
        for i, apple in enumerate(config["apples"]):
            self.add_apple(apple, i)
        for box in config.get("occluders", []):
            self._box(
                "occluder", box["position"], box["size"], box.get("color", [0.3, 0.3, 0.3, 1])
            )
        camera = config["camera"]
        self.width, self.height = camera["width"], camera["height"]
        self.near, self.far = camera["near_m"], camera["far_m"]
        self.rotation = Rotation.from_quat(camera["optical_quaternion_xyzw"])
        pos = np.asarray(camera["position"])
        forward = self.rotation.apply([0, 0, 1])
        up = self.rotation.apply([0, -1, 0])
        self.view = p.computeViewMatrix(pos, pos + forward, up)
        self.projection = p.computeProjectionMatrixFOV(
            camera["vertical_fov_deg"], self.width / self.height, self.near, self.far
        )
        focal = self.height / (2 * np.tan(np.deg2rad(camera["vertical_fov_deg"]) / 2))
        self.k = list(map(float, [focal, 0, self.width / 2, 0, focal, self.height / 2, 0, 0, 1]))
        self.events_done = set()

    def _box(self, name, position, size, color):
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=np.asarray(size) / 2,
            rgbaColor=color,
            physicsClientId=self.client,
        )
        body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=self.client,
        )
        self.objects.append(dict(name=name, position=position, size=size, color=color, body=body))

    def add_apple(self, apple, index):
        name = apple.get("name", f"apple_{index}")
        if name in self.apples:
            raise ValueError(f"Duplicate apple name: {name}")
        radius = apple["radius_m"]
        color = apple.get("color", [0.85, 0.06, 0.025, 1.0])
        shape = apple.get("shape", "apple")
        if shape == "sphere":
            visual = p.createVisualShape(
                p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=self.client
            )
        elif shape == "apple":
            mesh = Path(self.temporary.name) / f"{name}.obj"
            apple_mesh(mesh, radius, apple.get("lobes", 0.04))
            visual = p.createVisualShape(
                p.GEOM_MESH, fileName=str(mesh), rgbaColor=color, physicsClientId=self.client
            )
        else:
            raise ValueError("Apple shape must be sphere or apple")
        body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=apple["position"],
            physicsClientId=self.client,
        )
        stems = []
        if shape == "apple":
            stem = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius * 0.075,
                length=radius * 0.38,
                rgbaColor=[0.25, 0.12, 0.025, 1],
                physicsClientId=self.client,
            )
            pos = np.asarray(apple["position"]) + [0, 0, radius * 0.9]
            stems.append(
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=stem,
                    basePosition=pos,
                    physicsClientId=self.client,
                )
            )
        self.apples[name] = dict(apple, name=name, body=body, stems=stems, active=True, shape=shape)

    def apply_events(self, stamp):
        for i, event in enumerate(self.config.get("events", [])):
            if i in self.events_done or stamp < event["time_sec"]:
                continue
            self.events_done.add(i)
            if event["action"] == "add":
                self.add_apple(event["apple"], len(self.apples))
            elif event["action"] in ("remove", "move"):
                apple = self.apples[event["name"]]
                position = event.get("position", [0, 0, -10])
                delta = np.asarray(position) - apple["position"]
                for body in [apple["body"], *apple["stems"]]:
                    pos, q = p.getBasePositionAndOrientation(body, physicsClientId=self.client)
                    p.resetBasePositionAndOrientation(
                        body, np.asarray(pos) + delta, q, physicsClientId=self.client
                    )
                apple["position"] = position
                apple["active"] = event["action"] != "remove"
            else:
                raise ValueError(f"Unknown scene event: {event}")

    def render(self):
        light = self.config.get("light", {})
        image = p.getCameraImage(
            self.width,
            self.height,
            viewMatrix=self.view,
            projectionMatrix=self.projection,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK,
            lightDirection=light.get("direction", [-1, -1, 3]),
            lightColor=light.get("color", [1, 1, 1]),
            lightAmbientCoeff=light.get("ambient", 0.5),
            lightDiffuseCoeff=light.get("diffuse", 0.6),
            lightSpecularCoeff=light.get("specular", 0.25),
            physicsClientId=self.client,
        )
        rgb = np.asarray(image[2], np.uint8).reshape(self.height, self.width, 4)[..., :3]
        buffer = np.asarray(image[3]).reshape(self.height, self.width)
        depth = self.far * self.near / (self.far - (self.far - self.near) * buffer)
        depth[buffer >= 1 - 1e-7] = np.nan
        return rgb.copy(), depth.astype(np.float32)

    def truth(self):
        origin = np.asarray(self.config["robot_position"])
        return [
            dict(
                name=a["name"],
                center_base=(np.asarray(a["position"]) - origin).tolist(),
                radius_m=a["radius_m"],
                shape=a["shape"],
            )
            for a in self.apples.values()
            if a["active"]
        ]

    def close(self):
        p.disconnect(physicsClientId=self.client)
        self.temporary.cleanup()
