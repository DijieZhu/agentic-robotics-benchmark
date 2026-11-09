"""The built-in warm-up scene of render_twin.py --warmup.

It uses every kind of thing the exporter writes (planes, boxes, spheres, ellipsoids, cylinders, capsules and a
mesh; plain, textured, layered (colour + normal + ORM), metallic, emissive and transparent materials; spot, point and
directional lights; the dome; the OpenCV pinhole camera), so the first render compiles the shaders and the
materials that later renders reuse from the cache. Small image, few samples: once compiled it renders in seconds.
"""

SCENE_XML = """
<mujoco model="render_twin_warmup">
  <asset>
    <texture name="checker" type="2d" builtin="checker" rgb1="0.80 0.80 0.78" rgb2="0.25 0.30 0.35"
             width="256" height="256"/>
    <texture name="gradient" type="2d" builtin="gradient" rgb1="0.90 0.55 0.20" rgb2="0.15 0.25 0.60"
             width="128" height="128"/>
    <texture name="flat_normal" type="2d" builtin="flat" rgb1="0.5 0.5 1.0" width="16" height="16"/>
    <texture name="flat_orm" type="2d" builtin="flat" rgb1="1.0 0.6 0.0" width="16" height="16"/>
    <material name="floor" texture="checker" texrepeat="4 4" texuniform="true" roughness="0.8"/>
    <material name="painted" texture="gradient"/>
    <material name="layered">
      <layer role="rgb" texture="gradient"/>
      <layer role="normal" texture="flat_normal"/>
      <layer role="orm" texture="flat_orm"/>
    </material>
    <material name="steel" rgba="0.70 0.70 0.72 1" metallic="1" roughness="0.3"/>
    <material name="lamp" rgba="1.0 0.9 0.7 1" emission="1"/>
    <material name="glass" rgba="0.6 0.8 1.0 0.4"/>
    <mesh name="wedge" vertex="0 0 0  0.12 0 0  0 0.12 0  0 0 0.08"
          face="0 2 1  0 1 3  0 3 2  1 2 3" texcoord="0 0  1 0  0 1  0.5 0.5"/>
  </asset>
  <worldbody>
    <light name="spot" type="spot" pos="0.6 -0.6 1.6" dir="-0.35 0.35 -1" cutoff="35" exponent="4" castshadow="true"/>
    <light name="bulb" type="point" pos="-0.5 0.4 0.9" diffuse="0.4 0.4 0.4"/>
    <light name="sun" type="directional" pos="0 0 3" dir="0.3 0.2 -1" diffuse="0.3 0.3 0.3"/>
    <camera name="photo_cam" pos="0 -1.1 0.75" xyaxes="1 0 0 0 0.56 0.83" resolution="480 270"
            focalpixel="420 418" principalpixel="6 -4" sensorsize="0.0048 0.0027"/>
    <geom name="floor" type="plane" size="1 1 0.01" material="floor"/>
    <geom name="box" type="box" size="0.08 0.08 0.08" pos="0 0.05 0.08" euler="0 0 20" material="painted"/>
    <geom name="layered_box" type="box" size="0.05 0.07 0.03" pos="-0.12 -0.18 0.03" material="layered"/>
    <geom name="can" type="cylinder" size="0.04 0.07" pos="0.22 0.12 0.07" material="steel"/>
    <geom name="ball" type="sphere" size="0.05" pos="-0.24 0.08 0.05" rgba="0.75 0.20 0.18 1"/>
    <geom name="tube" type="capsule" size="0.025 0.06" pos="0.05 -0.28 0.025" euler="0 90 0" material="lamp"/>
    <geom name="egg" type="ellipsoid" size="0.07 0.045 0.035" pos="0.25 -0.15 0.035" material="glass"/>
    <geom name="wedge" type="mesh" mesh="wedge" pos="-0.35 -0.15 0" material="painted"/>
  </worldbody>
</mujoco>
"""
