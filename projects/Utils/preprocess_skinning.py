# preprocess_skinning.py
# Collapse 441-bone skinning weights to 56 Unity bones, sparsify to top-4,
# compute bind-pose inverse matrices, and export binary files for Unity runtime.

import json, pickle, sys, os, importlib
import numpy as np

# Reuse the bone mapping from dynamic_pkl_to_fbx.py
UNITY_BONE_MAP = [
    ("Root",            "root",       None),
    ("Hips",            "pelvis",     "Root"),
    ("Spine",           "spine_01",   "Hips"),
    ("Chest",           "spine_03",   "Spine"),
    ("UpperChest",      "spine_05",   "Chest"),
    ("Neck",            "neck_01",    "UpperChest"),
    ("Head",            "head",       "Neck"),
    ("LeftEye",         "FACIAL_L_Eye",  "Head"),
    ("RightEye",        "FACIAL_R_Eye",  "Head"),
    ("Jaw",             "FACIAL_C_Jaw",  "Head"),
    ("LeftShoulder",    "clavicle_l",  "UpperChest"),
    ("LeftUpperArm",    "upperarm_l",  "LeftShoulder"),
    ("LeftLowerArm",    "lowerarm_l",  "LeftUpperArm"),
    ("LeftHand",        "hand_l",      "LeftLowerArm"),
    ("RightShoulder",   "clavicle_r",  "UpperChest"),
    ("RightUpperArm",   "upperarm_r",  "RightShoulder"),
    ("RightLowerArm",   "lowerarm_r",  "RightUpperArm"),
    ("RightHand",       "hand_r",      "RightLowerArm"),
    ("LeftUpperLeg",    "thigh_l",     "Hips"),
    ("LeftLowerLeg",    "calf_l",      "LeftUpperLeg"),
    ("LeftFoot",        "foot_l",      "LeftLowerLeg"),
    ("LeftToes",        "ball_l",      "LeftFoot"),
    ("RightUpperLeg",   "thigh_r",     "Hips"),
    ("RightLowerLeg",   "calf_r",      "RightUpperLeg"),
    ("RightFoot",       "foot_r",      "RightLowerLeg"),
    ("RightToes",       "ball_r",      "RightFoot"),
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


def quat_xyzw_to_mat3(q):
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


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], axis=-1)


def quat_inv(q):
    return q * np.array([-1, -1, -1, 1], dtype=q.dtype)


def quat_rotate(q, v):
    vq = np.concatenate([v, np.zeros_like(v[..., :1])], axis=-1)
    return quat_mul(quat_mul(q, vq), quat_inv(q))[..., :3]


def compute_world_transforms(local_rot, local_trans, parents):
    F, B, _ = local_rot.shape
    world_rot = np.zeros_like(local_rot)
    world_trans = np.zeros_like(local_trans)
    for b in range(B):
        p = int(parents[b])
        if p == -1:
            world_rot[:, b] = local_rot[:, b]
            world_trans[:, b] = local_trans[:, b]
        else:
            world_rot[:, b] = quat_mul(world_rot[:, p], local_rot[:, b])
            world_trans[:, b] = world_trans[:, p] + quat_rotate(world_rot[:, p], local_trans[:, b])
    return world_rot, world_trans


def build_world_mat44(rot_xyzw, trans):
    R = quat_xyzw_to_mat3(rot_xyzw)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = trans
    return M


def main():
    if len(sys.argv) < 6:
        print("Usage: python preprocess_skinning.py gaussian_static.pkl bone_names_441.json parents_441.json dynamic.pkl bone_names_1185.json [output_dir]")
        sys.exit(1)

    static_pkl_path = sys.argv[1]
    names441_path = sys.argv[2]
    parents441_path = sys.argv[3]
    dynamic_pkl_path = sys.argv[4]
    names1185_path = sys.argv[5]
    output_dir = sys.argv[6] if len(sys.argv) > 6 else "."

    os.makedirs(output_dir, exist_ok=True)

    # Compatibility shims for old numpy pickles
    sys.modules['numpy._core'] = importlib.import_module('numpy.core')
    sys.modules['numpy._core.multiarray'] = importlib.import_module('numpy.core.multiarray')
    sys.modules['numpy._core._multiarray_umath'] = importlib.import_module('numpy.core._multiarray_umath')

    # --- Load static pkl (has W weights matrix) ---
    print("Loading gaussian_static.pkl...")
    with open(static_pkl_path, "rb") as f:
        static_data = pickle.load(f)

    W = static_data["W"]  # (N_splats, 441) skinning weights
    N_splats = W.shape[0]
    print(f"  Splat count: {N_splats}, Weight matrix shape: {W.shape}")

    # --- Load bone names and parents ---
    with open(names441_path, "r") as f:
        names441 = json.load(f)
    with open(parents441_path, "r") as f:
        parents441 = np.array(json.load(f), dtype=np.int32)

    assert len(names441) == 441
    assert len(parents441) == 441

    # --- Load dynamic pkl for frame-0 transforms ---
    print("Loading dynamic pkl for bind-pose transforms...")
    with open(dynamic_pkl_path, "rb") as f:
        dyn_data = pickle.load(f)

    with open(names1185_path, "r") as f:
        names1185 = json.load(f)

    rot_all = dyn_data["poses"]["rotations"]      # (F, 1185, 4) xyzw
    trans_all = dyn_data["poses"]["translations"]  # (F, 1185, 3)

    # Map 441 → 1185 indices
    name1185_to_i = {n: i for i, n in enumerate(names1185)}
    idx_441_in_1185 = np.array([name1185_to_i[n] for n in names441], dtype=np.int32)

    # Extract frame-0 only for 441 bones (just need 1 frame for bind pose)
    rot_441_f0 = rot_all[0:1, idx_441_in_1185, :]    # (1, 441, 4)
    trans_441_f0 = trans_all[0:1, idx_441_in_1185, :] # (1, 441, 3)

    # Compute world-space transforms for all 441 bones at frame 0
    print("Computing frame-0 world transforms for 441 bones...")
    world_rot_441, world_trans_441 = compute_world_transforms(rot_441_f0, trans_441_f0, parents441)
    # Squeeze frame dim: (441, 4), (441, 3)
    world_rot_441 = world_rot_441[0]
    world_trans_441 = world_trans_441[0]

    # --- Build Unity bone mapping ---
    name441_to_i = {n: i for i, n in enumerate(names441)}
    unity_names = []
    unity_src_441_indices = []

    for unity_name, src_name, _ in UNITY_BONE_MAP:
        if src_name not in name441_to_i:
            print(f"WARNING: '{src_name}' not in 441, skipping Unity bone '{unity_name}'")
            continue
        unity_names.append(unity_name)
        unity_src_441_indices.append(name441_to_i[src_name])

    NUM_UNITY = len(unity_names)
    print(f"Unity bones: {NUM_UNITY}")

    # Build a set of 441-index → Unity-index mapping
    src441_to_unity = {}
    for ui, idx441 in enumerate(unity_src_441_indices):
        src441_to_unity[idx441] = ui

    # --- For each of 441 bones, find nearest Unity ancestor ---
    bone441_to_unity = np.full(441, -1, dtype=np.int32)
    for b441 in range(441):
        cur = b441
        while cur != -1:
            if cur in src441_to_unity:
                bone441_to_unity[b441] = src441_to_unity[cur]
                break
            cur = int(parents441[cur])

    unmapped = np.sum(bone441_to_unity == -1)
    if unmapped > 0:
        print(f"WARNING: {unmapped} of 441 bones have no Unity ancestor (weights will be dropped)")

    # --- Collapse weights: sum W[:, 441] by their mapped Unity bone ---
    print("Collapsing weights from 441 → Unity bones...")
    W_unity = np.zeros((N_splats, NUM_UNITY), dtype=np.float32)
    for b441 in range(441):
        ui = bone441_to_unity[b441]
        if ui >= 0:
            W_unity[:, ui] += W[:, b441].astype(np.float32)

    # --- Sparsify to top-4 per splat ---
    print("Sparsifying to top-4 bones per splat...")
    top4_indices = np.zeros((N_splats, 4), dtype=np.int32)
    top4_weights = np.zeros((N_splats, 4), dtype=np.float32)

    for i in range(N_splats):
        w = W_unity[i]
        if np.sum(w) < 1e-12:
            # No weights at all - assign to bone 0 (Root)
            top4_indices[i, 0] = 0
            top4_weights[i, 0] = 1.0
            continue

        # Get top-4 bone indices by weight magnitude
        if NUM_UNITY <= 4:
            idx = np.arange(NUM_UNITY)
        else:
            idx = np.argpartition(w, -4)[-4:]

        vals = w[idx]
        # Sort descending
        order = np.argsort(-vals)
        idx = idx[order]
        vals = vals[order]

        # Keep only positive weights
        mask = vals > 0
        n_valid = np.sum(mask)
        if n_valid == 0:
            top4_indices[i, 0] = 0
            top4_weights[i, 0] = 1.0
            continue

        top4_indices[i, :n_valid] = idx[:n_valid]
        top4_weights[i, :n_valid] = vals[:n_valid]

        # Normalize
        total = np.sum(top4_weights[i])
        if total > 0:
            top4_weights[i] /= total

    # --- Compute bind-pose inverse matrices for Unity bones ---
    print("Computing bind-pose inverse matrices...")
    bind_inv_matrices = np.zeros((NUM_UNITY, 4, 4), dtype=np.float32)
    for ui in range(NUM_UNITY):
        idx441 = unity_src_441_indices[ui]
        M = build_world_mat44(world_rot_441[idx441], world_trans_441[idx441])
        bind_inv_matrices[ui] = np.linalg.inv(M).astype(np.float32)

    # --- Export binary files ---
    # skinning_indices.bytes: (N_splats * 4) int32
    indices_path = os.path.join(output_dir, "skinning_indices.bytes")
    top4_indices.tofile(indices_path)
    print(f"Wrote {indices_path}: {os.path.getsize(indices_path)} bytes")

    # skinning_weights.bytes: (N_splats * 4) float32
    weights_path = os.path.join(output_dir, "skinning_weights.bytes")
    top4_weights.tofile(weights_path)
    print(f"Wrote {weights_path}: {os.path.getsize(weights_path)} bytes")

    # bind_inv_matrices.bytes: (NUM_UNITY * 16) float32, row-major 4x4
    matrices_path = os.path.join(output_dir, "bind_inv_matrices.bytes")
    bind_inv_matrices.reshape(-1).tofile(matrices_path)
    print(f"Wrote {matrices_path}: {os.path.getsize(matrices_path)} bytes")

    # Also export the bone name order for verification
    names_path = os.path.join(output_dir, "unity_bone_names.json")
    with open(names_path, "w") as f:
        json.dump(unity_names, f, indent=2)
    print(f"Wrote {names_path}")

    # Summary
    print(f"\nDone! {NUM_UNITY} Unity bones, {N_splats} splats")
    print(f"Expected sizes:")
    print(f"  skinning_indices.bytes: {N_splats * 4 * 4} bytes ({N_splats} x 4 x int32)")
    print(f"  skinning_weights.bytes: {N_splats * 4 * 4} bytes ({N_splats} x 4 x float32)")
    print(f"  bind_inv_matrices.bytes: {NUM_UNITY * 16 * 4} bytes ({NUM_UNITY} x 4x4 x float32)")


if __name__ == "__main__":
    main()
