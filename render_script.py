
import bpy, os, sys, math, addon_utils
from mathutils import Vector

addon_utils.enable("cycles")
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 32
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.get_devices()
for d in prefs.devices: d.use = True

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setup_camera_and_lights():
    cam_data = bpy.data.cameras.new('Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    light_data = bpy.data.lights.new(name='Light', type='SUN')
    light_data.energy = 5.0
    light_obj = bpy.data.objects.new(name='Light', object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(60), math.radians(0), math.radians(45))
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = (1, 1, 1, 1)
        bg_node.inputs[1].default_value = 1.0

def normalize_scene():
    bbox_corners = []
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            for corner in obj.bound_box: bbox_corners.append(obj.matrix_world @ Vector(corner))
    if not bbox_corners: return
    min_b = Vector((min([v.x for v in bbox_corners]), min([v.y for v in bbox_corners]), min([v.z for v in bbox_corners])))
    max_b = Vector((max([v.x for v in bbox_corners]), max([v.y for v in bbox_corners]), max([v.z for v in bbox_corners])))
    center = (min_b + max_b) / 2
    scale = 1.0 / max(max_b - min_b)
    empty = bpy.data.objects.new("Empty", None)
    bpy.context.collection.objects.link(empty)
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.parent is None: obj.parent = empty
    empty.location = -center
    empty_master = bpy.data.objects.new("Master", None)
    bpy.context.collection.objects.link(empty_master)
    empty.parent = empty_master
    empty_master.scale = (scale*1.8, scale*1.8, scale*1.8)

def render_view(filepath, az, el, radius=2.2):
    cam = bpy.context.scene.camera
    cam.location = (
        radius * math.cos(math.radians(el)) * math.cos(math.radians(az)),
        radius * math.cos(math.radians(el)) * math.sin(math.radians(az)),
        radius * math.sin(math.radians(el))
    )
    direction = -cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    bpy.context.scene.render.filepath = filepath
    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256
    bpy.ops.render.render(write_still=True)

argv = sys.argv
argv = argv[argv.index("--") + 1:]
glb_path, uid_dir = argv[0], argv[1]

clear_scene()
try:
    bpy.ops.import_scene.gltf(filepath=glb_path)
    normalize_scene()
    setup_camera_and_lights()
    render_view(os.path.join(uid_dir, "input.png"), az=45, el=20)
    render_view(os.path.join(uid_dir, "view_0.png"), az=30, el=20)
    render_view(os.path.join(uid_dir, "view_1.png"), az=90, el=20)
    render_view(os.path.join(uid_dir, "view_2.png"), az=150, el=20)
    render_view(os.path.join(uid_dir, "view_3.png"), az=210, el=-20)
    render_view(os.path.join(uid_dir, "view_4.png"), az=270, el=-20)
    render_view(os.path.join(uid_dir, "view_5.png"), az=330, el=-20)
except: pass
