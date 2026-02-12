# dynamic_pkl_to_fbx.py
# Exports a subset of 56 bones from a 441-bone skeleton, renamed for Unity Humanoid avatar.
# Computes world-space transforms from the full 441 hierarchy, then re-derives correct
# local transforms for the new (sparser) Unity hierarchy.

import argparse, json, os, pickle, sys, importlib
import numpy as np

FBX_TICK = 46186158000  # ticks/sec in FBX

# ---------------------------------------------------------------------------
# Unity bone mapping: (unity_name, source_441_name, unity_parent_name)
# "unity_parent_name" is None for the root bone.
# ---------------------------------------------------------------------------
UNITY_BONE_MAP = [
    # Root & torso
    ("Root",            "root",       None),
    ("Hips",            "pelvis",     "Root"),
    ("Spine",           "spine_01",   "Hips"),
    ("Chest",           "spine_03",   "Spine"),
    ("UpperChest",      "spine_05",   "Chest"),
    ("Neck",            "neck_01",    "UpperChest"),
    ("Head",            "head",       "Neck"),
    # Eyes & jaw
    ("LeftEye",         "FACIAL_L_Eye",  "Head"),
    ("RightEye",        "FACIAL_R_Eye",  "Head"),
    ("Jaw",             "FACIAL_C_Jaw",  "Head"),
    # Left arm
    ("LeftShoulder",    "clavicle_l",  "UpperChest"),
    ("LeftUpperArm",    "upperarm_l",  "LeftShoulder"),
    ("LeftLowerArm",    "lowerarm_l",  "LeftUpperArm"),
    ("LeftHand",        "hand_l",      "LeftLowerArm"),
    # Right arm
    ("RightShoulder",   "clavicle_r",  "UpperChest"),
    ("RightUpperArm",   "upperarm_r",  "RightShoulder"),
    ("RightLowerArm",   "lowerarm_r",  "RightUpperArm"),
    ("RightHand",       "hand_r",      "RightLowerArm"),
    # Left leg
    ("LeftUpperLeg",    "thigh_l",     "Hips"),
    ("LeftLowerLeg",    "calf_l",      "LeftUpperLeg"),
    ("LeftFoot",        "foot_l",      "LeftLowerLeg"),
    ("LeftToes",        "ball_l",      "LeftFoot"),
    # Right leg
    ("RightUpperLeg",   "thigh_r",     "Hips"),
    ("RightLowerLeg",   "calf_r",      "RightUpperLeg"),
    ("RightFoot",       "foot_r",      "RightLowerLeg"),
    ("RightToes",       "ball_r",      "RightFoot"),
    # Left hand fingers
    ("LeftThumbProximal",       "thumb_01_l",   "LeftHand"),
    ("LeftThumbIntermediate",   "thumb_02_l",   "LeftThumbProximal"),
    ("LeftThumbDistal",         "thumb_03_l",   "LeftThumbIntermediate"),
    ("LeftIndexProximal",       "index_01_l",   "LeftHand"),
    ("LeftIndexIntermediate",   "index_02_l",   "LeftIndexProximal"),
    ("LeftIndexDistal",         "index_03_l",   "LeftIndexIntermediate"),
    ("LeftMiddleProximal",      "middle_01_l",  "LeftHand"),
    ("LeftMiddleIntermediate",  "middle_02_l",  "LeftMiddleProximal"),
    ("LeftMiddleDistal",        "middle_03_l",  "LeftMiddleIntermediate"),
    ("LeftRingProximal",        "ring_01_l",    "LeftHand"),
    ("LeftRingIntermediate",    "ring_02_l",    "LeftRingProximal"),
    ("LeftRingDistal",          "ring_03_l",    "LeftRingIntermediate"),
    ("LeftLittleProximal",      "pinky_01_l",   "LeftHand"),
    ("LeftLittleIntermediate",  "pinky_02_l",   "LeftLittleProximal"),
    ("LeftLittleDistal",        "pinky_03_l",   "LeftLittleIntermediate"),
    # Right hand fingers
    ("RightThumbProximal",      "thumb_01_r",   "RightHand"),
    ("RightThumbIntermediate",  "thumb_02_r",   "RightThumbProximal"),
    ("RightThumbDistal",        "thumb_03_r",   "RightThumbIntermediate"),
    ("RightIndexProximal",      "index_01_r",   "RightHand"),
    ("RightIndexIntermediate",  "index_02_r",   "RightIndexProximal"),
    ("RightIndexDistal",        "index_03_r",   "RightIndexIntermediate"),
    ("RightMiddleProximal",     "middle_01_r",  "RightHand"),
    ("RightMiddleIntermediate", "middle_02_r",  "RightMiddleProximal"),
    ("RightMiddleDistal",       "middle_03_r",  "RightMiddleIntermediate"),
    ("RightRingProximal",       "ring_01_r",    "RightHand"),
    ("RightRingIntermediate",   "ring_02_r",    "RightRingProximal"),
    ("RightRingDistal",         "ring_03_r",    "RightRingIntermediate"),
    ("RightLittleProximal",     "pinky_01_r",   "RightHand"),
    ("RightLittleIntermediate", "pinky_02_r",   "RightLittleProximal"),
    ("RightLittleDistal",       "pinky_03_r",   "RightLittleIntermediate"),
]

# ---------------------------------------------------------------------------
# Quaternion helpers  (xyzw layout, batched over leading dims)
# ---------------------------------------------------------------------------

def quat_mul(q1, q2):
    """Hamilton product of two quaternion arrays in xyzw layout."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], axis=-1)


def quat_inv(q):
    """Inverse (conjugate) of unit quaternion in xyzw layout."""
    return q * np.array([-1, -1, -1, 1], dtype=q.dtype)


def quat_rotate(q, v):
    """Rotate vector v by quaternion q (both batched, xyzw layout)."""
    # v_quat = (vx, vy, vz, 0)
    vq = np.concatenate([v, np.zeros_like(v[..., :1])], axis=-1)
    return quat_mul(quat_mul(q, vq), quat_inv(q))[..., :3]


def quat_xyzw_to_euler_xyz_deg(q):
    """Vectorised quaternion (xyzw) → intrinsic XYZ Euler angles in degrees.
    Input shape: (..., 4).  Output shape: (..., 3).
    """
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(x*x + y*y + z*z + w*w)
    n = np.where(n == 0, 1.0, n)
    x, y, z, w = x/n, y/n, z/n, w/n

    t0 = 2.0*(w*x + y*z)
    t1 = 1.0 - 2.0*(x*x + y*y)
    roll = np.arctan2(t0, t1)

    t2 = np.clip(2.0*(w*y - z*x), -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0*(w*z + x*y)
    t4 = 1.0 - 2.0*(y*y + z*z)
    yaw = np.arctan2(t3, t4)

    return np.stack([roll, pitch, yaw], axis=-1) * (180.0 / np.pi)


def quat_xyzw_to_mat3(q):
    """Convert a single xyzw quaternion to a 3x3 rotation matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = x*x + y*y + z*z + w*w
    if n == 0:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x*s*x, y*s*y, z*s*z
    xy, xz, yz = x*s*y, x*s*z, y*s*z
    wx, wy, wz = w*s*x, w*s*y, w*s*z
    return np.array([
        [1-yy-zz, xy-wz,   xz+wy],
        [xy+wz,   1-xx-zz, yz-wx],
        [xz-wy,   yz+wx,   1-xx-yy],
    ])


def build_world_mat44(world_rot_b, world_trans_b):
    """Build a row-major 4x4 matrix from world-space quaternion and translation."""
    R = quat_xyzw_to_mat3(world_rot_b)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = world_trans_b
    return M


# ---------------------------------------------------------------------------
# FBX 7.4 ASCII writer
# ---------------------------------------------------------------------------

def write_fbx_ascii(filepath, bone_names, parents, trans, rot_quat_xyzw,
                    world_rot_frame0, world_trans_frame0, fps=30):
    """Write an FBX 7.4 ASCII file with skeleton animation."""
    F, B, _ = trans.shape
    assert len(bone_names) == B and len(parents) == B

    next_id = [100000]
    def gid():
        next_id[0] += 1
        return next_id[0]

    rig_model_id = gid()
    rig_na_id = gid()
    model_ids = [gid() for _ in range(B)]
    na_ids = [gid() for _ in range(B)]
    pose_id = gid()
    astack_id = gid()
    alayer_id = gid()
    cn_t_ids = [gid() for _ in range(B)]
    cn_r_ids = [gid() for _ in range(B)]
    curve_ids = {}
    for i in range(B):
        for t in ("T", "R"):
            for a in ("X", "Y", "Z"):
                curve_ids[(i, t, a)] = gid()

    dt = int(round(FBX_TICK / fps))
    times = [k * dt for k in range(F)]

    eulers = quat_xyzw_to_euler_xyz_deg(rot_quat_xyzw)  # (F, B, 3)
    bind_eulers = eulers[0]  # (B, 3)

    L = []  # output lines
    def w(line=""):
        L.append(line)

    def fmt_d(v):
        return f"{float(v)}"

    def comma_d(arr):
        return ",".join(fmt_d(v) for v in arr)

    # --- Header ---
    w("; FBX 7.4.0 project file")
    w("FBXHeaderExtension:  {")
    w("    FBXHeaderVersion: 1003")
    w("    FBXVersion: 7400")
    w("    Creator: \"dynamic_pkl_to_fbx\"")
    w("}")
    w("")

    # --- GlobalSettings ---
    w("GlobalSettings:  {")
    w("    Version: 1000")
    w("    Properties70:  {")
    w("        P: \"UpAxis\", \"int\", \"Integer\", \"\",1")
    w("        P: \"UpAxisSign\", \"int\", \"Integer\", \"\",1")
    w("        P: \"FrontAxis\", \"int\", \"Integer\", \"\",2")
    w("        P: \"FrontAxisSign\", \"int\", \"Integer\", \"\",1")
    w("        P: \"CoordAxis\", \"int\", \"Integer\", \"\",0")
    w("        P: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1")
    w("        P: \"TimeMode\", \"enum\", \"\", \"\",6")
    w("        P: \"UnitScaleFactor\", \"double\", \"Number\", \"\",1")
    w("        P: \"OriginalUnitScaleFactor\", \"double\", \"Number\", \"\",1")
    w("    }")
    w("}")
    w("")

    # --- Documents ---
    w("Documents:  {")
    w("    Count: 1")
    w("    Document: 999999999, \"Scene\", \"\" {")
    w("        Properties70:  {")
    w("            P: \"SourceObject\", \"object\", \"\", \"\"")
    w("            P: \"ActiveAnimStackName\", \"KString\", \"\", \"\", \"Take 001\"")
    w("        }")
    w("        RootNode: 0")
    w("    }")
    w("}")
    w("")

    # --- References ---
    w("References:  {")
    w("}")
    w("")

    # --- Definitions ---
    w("Definitions:  {")
    w("    Version: 100")
    w(f"    Count: {B + 1 + B + 1 + 1 + 1 + 1 + 2*B + 6*B + 1}")
    w("    ObjectType: \"GlobalSettings\" {")
    w("        Count: 1")
    w("    }")
    w(f"    ObjectType: \"Model\" {{")
    w(f"        Count: {B + 1}")
    w("    }")
    w(f"    ObjectType: \"NodeAttribute\" {{")
    w(f"        Count: {B + 1}")
    w("    }")
    w("    ObjectType: \"Pose\" {")
    w("        Count: 1")
    w("    }")
    w("    ObjectType: \"AnimationStack\" {")
    w("        Count: 1")
    w("    }")
    w("    ObjectType: \"AnimationLayer\" {")
    w("        Count: 1")
    w("    }")
    w(f"    ObjectType: \"AnimationCurveNode\" {{")
    w(f"        Count: {2 * B}")
    w("    }")
    w(f"    ObjectType: \"AnimationCurve\" {{")
    w(f"        Count: {6 * B}")
    w("    }")
    w("}")
    w("")

    # --- Objects ---
    w("Objects:  {")

    # Rig NodeAttribute (Null)
    w(f"    NodeAttribute: {rig_na_id}, \"NodeAttribute::Rig\", \"Null\" {{")
    w("        TypeFlags: \"Null\"")
    w("    }")

    # Bone NodeAttributes (LimbNode + Skeleton)
    for b, name in enumerate(bone_names):
        w(f"    NodeAttribute: {na_ids[b]}, \"NodeAttribute::{name}\", \"LimbNode\" {{")
        w("        TypeFlags: \"Skeleton\"")
        w("    }")

    # Rig Model (Null)
    w(f"    Model: {rig_model_id}, \"Model::Rig\", \"Null\" {{")
    w("        Version: 232")
    w("        Properties70:  {")
    w("            P: \"ScalingMax\", \"Vector3D\", \"Vector\", \"\",0,0,0")
    w("            P: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",0")
    w("        }")
    w("        Shading: T")
    w("        Culling: \"CullingOff\"")
    w("    }")

    # Bone Models (LimbNode)
    for b, name in enumerate(bone_names):
        tx, ty, tz = float(trans[0, b, 0]), float(trans[0, b, 1]), float(trans[0, b, 2])
        rx, ry, rz = float(bind_eulers[b, 0]), float(bind_eulers[b, 1]), float(bind_eulers[b, 2])
        w(f"    Model: {model_ids[b]}, \"Model::{name}\", \"LimbNode\" {{")
        w("        Version: 232")
        w("        Properties70:  {")
        w("            P: \"RotationActive\", \"bool\", \"\", \"\",1")
        w("            P: \"InheritType\", \"enum\", \"\", \"\",1")
        w("            P: \"ScalingMax\", \"Vector3D\", \"Vector\", \"\",0,0,0")
        w("            P: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",0")
        w(f"            P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",{fmt_d(tx)},{fmt_d(ty)},{fmt_d(tz)}")
        w(f"            P: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A\",{fmt_d(rx)},{fmt_d(ry)},{fmt_d(rz)}")
        w("            P: \"Lcl Scaling\", \"Lcl Scaling\", \"\", \"A\",1,1,1")
        w("        }")
        w("        Shading: T")
        w("        Culling: \"CullingOff\"")
        w("    }")

    # BindPose
    w(f"    Pose: {pose_id}, \"Pose::BindPose\", \"BindPose\" {{")
    w("        Type: \"BindPose\"")
    w("        Version: 100")
    w(f"        NbPoseNodes: {B + 1}")
    # Rig identity
    w("        PoseNode:  {")
    w(f"            Node: {rig_model_id}")
    w("            Matrix: *16 {")
    w("                a: 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1")
    w("            }")
    w("        }")
    for b in range(B):
        M = build_world_mat44(world_rot_frame0[b], world_trans_frame0[b])
        mvals = comma_d(M.flatten())
        w("        PoseNode:  {")
        w(f"            Node: {model_ids[b]}")
        w("            Matrix: *16 {")
        w(f"                a: {mvals}")
        w("            }")
        w("        }")
    w("    }")

    # AnimationStack
    last_time = times[-1]
    w(f"    AnimationStack: {astack_id}, \"AnimStack::Take 001\", \"\" {{")
    w("        Properties70:  {")
    w(f"            P: \"LocalStart\", \"KTime\", \"Time\", \"\",0")
    w(f"            P: \"LocalStop\", \"KTime\", \"Time\", \"\",{last_time}")
    w(f"            P: \"ReferenceStart\", \"KTime\", \"Time\", \"\",0")
    w(f"            P: \"ReferenceStop\", \"KTime\", \"Time\", \"\",{last_time}")
    w("        }")
    w("    }")

    # AnimationLayer
    w(f"    AnimationLayer: {alayer_id}, \"AnimLayer::BaseLayer\", \"\" {{")
    w("    }")

    # AnimationCurveNodes
    for b in range(B):
        for tag, cid in [("T", cn_t_ids[b]), ("R", cn_r_ids[b])]:
            w(f"    AnimationCurveNode: {cid}, \"AnimCurveNode::{tag}\", \"\" {{")
            w("        Properties70:  {")
            w("            P: \"d|X\", \"Number\", \"\", \"A\",0")
            w("            P: \"d|Y\", \"Number\", \"\", \"A\",0")
            w("            P: \"d|Z\", \"Number\", \"\", \"A\",0")
            w("        }")
            w("    }")

    # AnimationCurves
    for b in range(B):
        for ai, axis in enumerate("XYZ"):
            for typ, cn_ids, data in [("T", cn_t_ids, trans), ("R", cn_r_ids, eulers)]:
                cid = curve_ids[(b, typ, axis)]
                vals = data[:, b, ai]
                times_str = ",".join(str(t) for t in times)
                vals_str = ",".join(fmt_d(v) for v in vals)
                w(f"    AnimationCurve: {cid}, \"AnimCurve::\", \"\" {{")
                w("        Default: 0")
                w("        KeyVer: 4008")
                w(f"        KeyTime: *{F} {{")
                w(f"            a: {times_str}")
                w("        }")
                w(f"        KeyValueFloat: *{F} {{")
                w(f"            a: {vals_str}")
                w("        }")
                w("        KeyAttrFlags: *1 {")
                w("            a: 24840")
                w("        }")
                w("        KeyAttrDataFloat: *4 {")
                w("            a: 0,0,0,0")
                w("        }")
                w(f"        KeyAttrRefCount: *1 {{")
                w(f"            a: {F}")
                w("        }")
                w("    }")

    w("}")  # end Objects
    w("")

    # --- Connections ---
    w("Connections:  {")
    # Rig
    w(f"    C: \"OO\",{rig_na_id},{rig_model_id}")
    w(f"    C: \"OO\",{rig_model_id},0")
    # NodeAttr → Model
    for b in range(B):
        w(f"    C: \"OO\",{na_ids[b]},{model_ids[b]}")
    # Bone hierarchy
    for b in range(B):
        p = int(parents[b])
        pid = rig_model_id if p == -1 else model_ids[p]
        w(f"    C: \"OO\",{model_ids[b]},{pid}")
    # Animation
    w(f"    C: \"OO\",{alayer_id},{astack_id}")
    for b in range(B):
        w(f"    C: \"OO\",{cn_t_ids[b]},{alayer_id}")
        w(f"    C: \"OO\",{cn_r_ids[b]},{alayer_id}")
        w(f"    C: \"OP\",{cn_t_ids[b]},{model_ids[b]}, \"Lcl Translation\"")
        w(f"    C: \"OP\",{cn_r_ids[b]},{model_ids[b]}, \"Lcl Rotation\"")
        for ai, axis in enumerate("XYZ"):
            w(f"    C: \"OP\",{curve_ids[(b, 'T', axis)]},{cn_t_ids[b]}, \"d|{axis}\"")
            w(f"    C: \"OP\",{curve_ids[(b, 'R', axis)]},{cn_r_ids[b]}, \"d|{axis}\"")
    w("}")
    w("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ---------------------------------------------------------------------------
# World-space forward kinematics for all 441 bones
# ---------------------------------------------------------------------------

def compute_world_transforms(local_rot, local_trans, parents_441):
    """Compute world-space rotation and translation for every bone every frame.

    local_rot:    (F, 441, 4)  xyzw quaternions
    local_trans:  (F, 441, 3)
    parents_441:  (441,) int array, -1 for root

    Returns: world_rot (F, 441, 4), world_trans (F, 441, 3)
    """
    F, B, _ = local_rot.shape
    world_rot = np.zeros_like(local_rot)
    world_trans = np.zeros_like(local_trans)

    # Topological order: process bones in index order (parents always have lower index)
    for b in range(B):
        p = int(parents_441[b])
        if p == -1:
            world_rot[:, b] = local_rot[:, b]
            world_trans[:, b] = local_trans[:, b]
        else:
            # world_rot[b] = world_rot[parent] * local_rot[b]
            world_rot[:, b] = quat_mul(world_rot[:, p], local_rot[:, b])
            # world_trans[b] = world_trans[parent] + rotate(world_rot[parent], local_trans[b])
            world_trans[:, b] = world_trans[:, p] + quat_rotate(world_rot[:, p], local_trans[:, b])

    return world_rot, world_trans


def rederive_local_transforms(world_rot, world_trans, unity_parent_indices):
    """Re-derive local transforms for the Unity hierarchy from world-space data.

    world_rot:   (F, num_unity_bones, 4)  xyzw
    world_trans: (F, num_unity_bones, 3)
    unity_parent_indices: list[int], -1 for root

    Returns: local_rot (F, num_unity_bones, 4), local_trans (F, num_unity_bones, 3)
    """
    F, B, _ = world_rot.shape
    local_rot = np.zeros_like(world_rot)
    local_trans = np.zeros_like(world_trans)

    for b in range(B):
        p = unity_parent_indices[b]
        if p == -1:
            local_rot[:, b] = world_rot[:, b]
            local_trans[:, b] = world_trans[:, b]
        else:
            # local_rot[b] = inv(world_rot[parent]) * world_rot[b]
            local_rot[:, b] = quat_mul(quat_inv(world_rot[:, p]), world_rot[:, b])
            # local_trans[b] = inv_rotate(world_rot[parent], world_trans[b] - world_trans[parent])
            local_trans[:, b] = quat_rotate(
                quat_inv(world_rot[:, p]),
                world_trans[:, b] - world_trans[:, p]
            )

    return local_rot, local_trans


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def flip_y_world_transforms(world_rot, world_trans):
    """Apply 180° rotation around X axis to convert Y-down → Y-up coordinate system.

    This is a proper rotation (not a reflection), equivalent to diag(1, -1, -1):
      translation: (tx, ty, tz) → (tx, -ty, -tz)
      quaternion xyzw: (x, y, z, w) → (x, -y, -z, w)
    """
    world_trans = world_trans.copy()
    world_trans[..., 1] *= -1
    world_trans[..., 2] *= -1

    world_rot = world_rot.copy()
    world_rot[..., 1] *= -1  # negate y
    world_rot[..., 2] *= -1  # negate z

    return world_rot, world_trans


def main():
    parser = argparse.ArgumentParser(
        description="Export 56 Unity bones from 441-bone skeleton as FBX ASCII."
    )
    parser.add_argument("pkl", help="Path to dynamic.pkl")
    parser.add_argument("names1185", help="Path to bone_names_1185.json")
    parser.add_argument("names441", help="Path to bone_names_441.json")
    parser.add_argument("out_fbx", help="Output FBX file path")
    parser.add_argument("parents441", help="Path to parents_441.json or .npy")
    parser.add_argument("--reverse-y", action="store_true",
                        help="Flip Y axis (negate Y translations, reflect rotations) "
                             "to convert from Y-down to Y-up coordinate system")
    args = parser.parse_args()

    pkl_path = args.pkl
    names1185_path = args.names1185
    names441_path = args.names441
    out_fbx = args.out_fbx
    parents_path = args.parents441

    # Compatibility for pickles made with old numpy module paths
    sys.modules['numpy._core'] = importlib.import_module('numpy.core')
    sys.modules['numpy._core.multiarray'] = importlib.import_module('numpy.core.multiarray')
    sys.modules['numpy._core._multiarray_umath'] = importlib.import_module('numpy.core._multiarray_umath')

    # --- Load data ---
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    rot_all = d["poses"]["rotations"]       # (F, 1185, 4) xyzw
    trans_all = d["poses"]["translations"]   # (F, 1185, 3)

    with open(names1185_path, "r", encoding="utf-8") as f:
        names1185 = json.load(f)
    with open(names441_path, "r", encoding="utf-8") as f:
        names441 = json.load(f)

    if parents_path.endswith(".npy"):
        parents441 = np.load(parents_path).astype(np.int32)
    else:
        with open(parents_path, "r", encoding="utf-8") as f:
            parents441 = np.array(json.load(f), dtype=np.int32)

    # --- Map 441 names → 1185 indices, extract 441 bone data ---
    name1185_to_i = {n: i for i, n in enumerate(names1185)}
    idx_441_in_1185 = []
    missing = []
    for n in names441:
        if n in name1185_to_i:
            idx_441_in_1185.append(name1185_to_i[n])
        else:
            idx_441_in_1185.append(-1)
            missing.append(n)

    if missing:
        print(f"ERROR: {len(missing)} bones from 441 list not found in 1185 list:")
        for n in missing[:20]:
            print(f"  - {n}")
        sys.exit(2)

    idx_441_in_1185 = np.array(idx_441_in_1185, dtype=np.int32)
    rot_441 = rot_all[:, idx_441_in_1185, :]     # (F, 441, 4)
    trans_441 = trans_all[:, idx_441_in_1185, :]  # (F, 441, 3)

    # --- Compute world-space transforms for all 441 bones ---
    print("Computing world-space transforms for 441 bones...")
    world_rot_441, world_trans_441 = compute_world_transforms(rot_441, trans_441, parents441)

    if args.reverse_y:
        print("Applying Y-axis flip to world-space transforms...")
        world_rot_441, world_trans_441 = flip_y_world_transforms(world_rot_441, world_trans_441)

    # --- Select Unity bones from 441 by name lookup ---
    name441_to_i = {n: i for i, n in enumerate(names441)}
    unity_names = []
    unity_src_indices = []  # index into 441 array
    unity_parent_map = {}   # unity_name -> index in unity_names list

    for unity_name, src_name, parent_unity_name in UNITY_BONE_MAP:
        if src_name not in name441_to_i:
            print(f"WARNING: source bone '{src_name}' for Unity bone '{unity_name}' not found in 441 skeleton, skipping")
            continue
        unity_names.append(unity_name)
        unity_src_indices.append(name441_to_i[src_name])
        unity_parent_map[unity_name] = len(unity_names) - 1

    # Build parent index array for the Unity hierarchy
    unity_parent_indices = []
    for unity_name, src_name, parent_unity_name in UNITY_BONE_MAP:
        if src_name not in name441_to_i:
            continue
        if parent_unity_name is None:
            unity_parent_indices.append(-1)
        else:
            unity_parent_indices.append(unity_parent_map[parent_unity_name])

    num_unity = len(unity_names)
    print(f"Selected {num_unity} Unity bones")

    # Gather world-space data for the selected Unity bones: (F, num_unity, ...)
    unity_src_indices = np.array(unity_src_indices, dtype=np.int32)
    world_rot_unity = world_rot_441[:, unity_src_indices, :]
    world_trans_unity = world_trans_441[:, unity_src_indices, :]

    # --- Re-derive local transforms for the Unity hierarchy ---
    print("Re-deriving local transforms for Unity hierarchy...")
    local_rot_unity, local_trans_unity = rederive_local_transforms(
        world_rot_unity, world_trans_unity, unity_parent_indices
    )

    # --- Resolve output path (avoid overwriting existing files) ---
    if os.path.exists(out_fbx):
        base, ext = os.path.splitext(out_fbx)
        n = 2
        while os.path.exists(f"{base}_{n}{ext}"):
            n += 1
        out_fbx = f"{base}_{n}{ext}"
        print(f"File already exists, writing to: {out_fbx}")

    # --- Write FBX ---
    world_rot_f0 = world_rot_unity[0]      # (num_unity, 4)
    world_trans_f0 = world_trans_unity[0]   # (num_unity, 3)
    print(f"Writing FBX with {num_unity} bones, {local_rot_unity.shape[0]} frames...")
    write_fbx_ascii(out_fbx, unity_names, unity_parent_indices,
                     local_trans_unity, local_rot_unity,
                     world_rot_f0, world_trans_f0, fps=30)
    print(f"Wrote: {out_fbx}")


if __name__ == "__main__":
    main()
