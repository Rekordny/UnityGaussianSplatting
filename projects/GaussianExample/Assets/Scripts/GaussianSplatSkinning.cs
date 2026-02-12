// SPDX-License-Identifier: MIT
// Runtime component that drives Gaussian splat bone skinning (position-only).
// Attach to the same GameObject as GaussianSplatRenderer.
// Following vsupersplat approach: position-only LBS, no rotation blending.

using System;
using UnityEngine;
using GaussianSplatting.Runtime;

[RequireComponent(typeof(GaussianSplatRenderer))]
public class GaussianSplatSkinning : MonoBehaviour
{
    [Header("Skinning Data (exported from preprocess_skinning.py)")]
    [Tooltip("skinning_indices.bytes — (N x 4) int32, top-4 bone indices per splat")]
    public TextAsset skinningIndices;

    [Tooltip("skinning_weights.bytes — (N x 4) float32, top-4 bone weights per splat")]
    public TextAsset skinningWeights;

    [Header("References")]
    [Tooltip("The Animator driving the skeleton (on the FBX)")]
    public Animator skeletonAnimator;

    [Tooltip("Compute shader for GPU skinning")]
    public ComputeShader skinningCompute;

    [Header("Debug")]
    [Tooltip("Force all skinning matrices to identity (character should look static/normal)")]
    public bool debugForceIdentity = false;

    [Tooltip("Log detailed debug info on first few frames")]
    public bool debugVerbose = true;

    [Tooltip("GPU passthrough: compute shader copies bind-pose directly (skips matrix multiply on GPU)")]
    public bool debugGpuPassthrough = false;

    [Tooltip("Transpose matrices before upload (debug: tests structured buffer layout)")]
    public bool debugTransposeMatrices = false;

    // Unity Humanoid bone names in the exact order exported by preprocess_skinning.py.
    static readonly string[] kBoneNames =
    {
        "Root",
        "Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
        "LeftEye", "RightEye", "Jaw",
        "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
        "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
        "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
        "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
        "LeftThumbProximal", "LeftThumbIntermediate", "LeftThumbDistal",
        "LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal",
        "LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
        "LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal",
        "LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal",
        "RightThumbProximal", "RightThumbIntermediate", "RightThumbDistal",
        "RightIndexProximal", "RightIndexIntermediate", "RightIndexDistal",
        "RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal",
        "RightRingProximal", "RightRingIntermediate", "RightRingDistal",
        "RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal",
    };

    // Map from our bone name to HumanBodyBones enum (Root is special)
    static readonly HumanBodyBones?[] kHumanBones =
    {
        null, // Root — use animator.transform
        HumanBodyBones.Hips,
        HumanBodyBones.Spine,
        HumanBodyBones.Chest,
        HumanBodyBones.UpperChest,
        HumanBodyBones.Neck,
        HumanBodyBones.Head,
        HumanBodyBones.LeftEye,
        HumanBodyBones.RightEye,
        HumanBodyBones.Jaw,
        HumanBodyBones.LeftShoulder,
        HumanBodyBones.LeftUpperArm,
        HumanBodyBones.LeftLowerArm,
        HumanBodyBones.LeftHand,
        HumanBodyBones.RightShoulder,
        HumanBodyBones.RightUpperArm,
        HumanBodyBones.RightLowerArm,
        HumanBodyBones.RightHand,
        HumanBodyBones.LeftUpperLeg,
        HumanBodyBones.LeftLowerLeg,
        HumanBodyBones.LeftFoot,
        HumanBodyBones.LeftToes,
        HumanBodyBones.RightUpperLeg,
        HumanBodyBones.RightLowerLeg,
        HumanBodyBones.RightFoot,
        HumanBodyBones.RightToes,
        HumanBodyBones.LeftThumbProximal,
        HumanBodyBones.LeftThumbIntermediate,
        HumanBodyBones.LeftThumbDistal,
        HumanBodyBones.LeftIndexProximal,
        HumanBodyBones.LeftIndexIntermediate,
        HumanBodyBones.LeftIndexDistal,
        HumanBodyBones.LeftMiddleProximal,
        HumanBodyBones.LeftMiddleIntermediate,
        HumanBodyBones.LeftMiddleDistal,
        HumanBodyBones.LeftRingProximal,
        HumanBodyBones.LeftRingIntermediate,
        HumanBodyBones.LeftRingDistal,
        HumanBodyBones.LeftLittleProximal,
        HumanBodyBones.LeftLittleIntermediate,
        HumanBodyBones.LeftLittleDistal,
        HumanBodyBones.RightThumbProximal,
        HumanBodyBones.RightThumbIntermediate,
        HumanBodyBones.RightThumbDistal,
        HumanBodyBones.RightIndexProximal,
        HumanBodyBones.RightIndexIntermediate,
        HumanBodyBones.RightIndexDistal,
        HumanBodyBones.RightMiddleProximal,
        HumanBodyBones.RightMiddleIntermediate,
        HumanBodyBones.RightMiddleDistal,
        HumanBodyBones.RightRingProximal,
        HumanBodyBones.RightRingIntermediate,
        HumanBodyBones.RightRingDistal,
        HumanBodyBones.RightLittleProximal,
        HumanBodyBones.RightLittleIntermediate,
        HumanBodyBones.RightLittleDistal,
    };

    GaussianSplatRenderer m_Renderer;
    int m_SplatCount;
    int m_BoneCount;

    // GPU buffers
    GraphicsBuffer m_BufBoneIndices;
    GraphicsBuffer m_BufBoneWeights;
    GraphicsBuffer m_BufBindPosePos;
    GraphicsBuffer m_BufBoneMatrices;
    GraphicsBuffer m_BufSkinPos;

    // CPU arrays
    Matrix4x4[] m_BindInvMatrices;
    Matrix4x4[] m_BoneMatricesCPU;
    Transform[] m_BoneTransforms;

    // Kernel indices
    int m_KernelCopyBindPose;
    int m_KernelBoneSkinning;

    bool m_Initialized;
    bool m_BindPoseCaptured;
    bool m_SkinningActivated;
    int m_FrameCount;

    static readonly int PropBindPosePos = Shader.PropertyToID("_BindPosePos");
    static readonly int PropBindPosePosIn = Shader.PropertyToID("_BindPosePosIn");
    static readonly int PropBoneIndices = Shader.PropertyToID("_BoneIndices");
    static readonly int PropBoneWeights = Shader.PropertyToID("_BoneWeights");
    static readonly int PropBoneMatrices = Shader.PropertyToID("_BoneMatrices");
    static readonly int PropSkinOutPos = Shader.PropertyToID("_SkinOutPos");
    static readonly int PropSplatCount = Shader.PropertyToID("_SplatCount");
    static readonly int PropSplatSkinPos = Shader.PropertyToID("_SplatSkinPos");
    static readonly int PropSplatSkinningActive = Shader.PropertyToID("_SplatSkinningActive");

    void OnEnable()
    {
        m_Renderer = GetComponent<GaussianSplatRenderer>();
        TryInitialize();
    }

    bool TryInitialize()
    {
        if (m_Initialized)
            return true;

        if (m_Renderer == null || !m_Renderer.HasValidAsset || !m_Renderer.HasValidRenderSetup)
            return false;

        if (skinningIndices == null || skinningWeights == null)
        {
            Debug.LogError("GaussianSplatSkinning: Skinning data TextAssets not assigned");
            enabled = false;
            return false;
        }

        if (skeletonAnimator == null)
        {
            Debug.LogError("GaussianSplatSkinning: Animator not assigned");
            enabled = false;
            return false;
        }

        if (skinningCompute == null)
        {
            Debug.LogError("GaussianSplatSkinning: Compute shader not assigned");
            enabled = false;
            return false;
        }

        m_SplatCount = m_Renderer.splatCount;
        m_BoneCount = kBoneNames.Length;
        Debug.Log($"GaussianSplatSkinning: {m_SplatCount} splats, {m_BoneCount} bones");

        // Resolve bone transforms from Animator
        m_BoneTransforms = new Transform[m_BoneCount];
        int foundBones = 0;
        for (int b = 0; b < m_BoneCount && b < kHumanBones.Length; b++)
        {
            if (kHumanBones[b] == null)
            {
                m_BoneTransforms[b] = skeletonAnimator.transform;
                foundBones++;
            }
            else
            {
                m_BoneTransforms[b] = skeletonAnimator.GetBoneTransform(kHumanBones[b].Value);
                if (m_BoneTransforms[b] != null)
                    foundBones++;
                else if (debugVerbose)
                    Debug.LogWarning($"GaussianSplatSkinning: Bone '{kBoneNames[b]}' ({kHumanBones[b].Value}) not found on Animator");
            }
        }
        Debug.Log($"GaussianSplatSkinning: Found {foundBones}/{m_BoneCount} bones on Animator");

        // Create GPU buffers
        m_BufBoneIndices = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount, 16); // uint4
        m_BufBoneWeights = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount, 16); // float4
        m_BufBindPosePos = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount, 12); // float3
        m_BufBoneMatrices = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_BoneCount, 64); // float4x4
        m_BufSkinPos = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount, 12);     // float3

        // Upload bone indices: int32 → reinterpret bits as float → Vector4 → GPU reads as uint4
        {
            byte[] raw = skinningIndices.bytes;
            int expectedSize = m_SplatCount * 4 * 4;
            if (raw.Length != expectedSize)
                Debug.LogError($"GaussianSplatSkinning: skinningIndices size mismatch: got {raw.Length}, expected {expectedSize}");

            float[] asFloat = new float[m_SplatCount * 4];
            Buffer.BlockCopy(raw, 0, asFloat, 0, Math.Min(raw.Length, asFloat.Length * 4));
            var packed = new Vector4[m_SplatCount];

            // Debug: check first few bone indices
            int[] debugIndices = new int[Math.Min(16, m_SplatCount * 4)];
            Buffer.BlockCopy(raw, 0, debugIndices, 0, debugIndices.Length * 4);
            if (debugVerbose)
            {
                string idxStr = "";
                for (int d = 0; d < Math.Min(4, m_SplatCount); d++)
                    idxStr += $"  splat[{d}]: [{debugIndices[d*4]},{debugIndices[d*4+1]},{debugIndices[d*4+2]},{debugIndices[d*4+3]}]\n";
                Debug.Log($"GaussianSplatSkinning: First bone indices:\n{idxStr}");
            }

            for (int i = 0; i < m_SplatCount; i++)
            {
                int o = i * 4;
                packed[i] = new Vector4(asFloat[o], asFloat[o + 1], asFloat[o + 2], asFloat[o + 3]);
            }
            m_BufBoneIndices.SetData(packed);
        }

        // Upload bone weights: float32 → Vector4
        {
            byte[] raw = skinningWeights.bytes;
            var packed = new Vector4[m_SplatCount];
            float[] floatData = new float[m_SplatCount * 4];
            Buffer.BlockCopy(raw, 0, floatData, 0, Math.Min(raw.Length, floatData.Length * 4));

            if (debugVerbose)
            {
                string wtStr = "";
                for (int d = 0; d < Math.Min(4, m_SplatCount); d++)
                    wtStr += $"  splat[{d}]: [{floatData[d*4]:F4},{floatData[d*4+1]:F4},{floatData[d*4+2]:F4},{floatData[d*4+3]:F4}]\n";
                Debug.Log($"GaussianSplatSkinning: First bone weights:\n{wtStr}");
            }

            for (int i = 0; i < m_SplatCount; i++)
            {
                int o = i * 4;
                packed[i] = new Vector4(floatData[o], floatData[o + 1], floatData[o + 2], floatData[o + 3]);
            }
            m_BufBoneWeights.SetData(packed);
        }

        // Get kernel indices
        m_KernelCopyBindPose = skinningCompute.FindKernel("CSCopyBindPose");
        m_KernelBoneSkinning = skinningCompute.FindKernel("CSBoneSkinning");

        // Dispatch CSCopyBindPose to decompress bind-pose positions
        DispatchCopyBindPose();

        // Allocate CPU arrays
        m_BindInvMatrices = new Matrix4x4[m_BoneCount];
        m_BoneMatricesCPU = new Matrix4x4[m_BoneCount];

        // DO NOT activate skinning here — wait until after first CSBoneSkinning dispatch
        // to ensure m_BufSkinPos has valid data before the renderer reads it.
        m_SkinningActivated = false;
        m_FrameCount = 0;

        m_Initialized = true;
        return true;
    }

    void DispatchCopyBindPose()
    {
        skinningCompute.SetBuffer(m_KernelCopyBindPose, "_SplatPos", m_Renderer.gpuPosData);
        skinningCompute.SetBuffer(m_KernelCopyBindPose, "_SplatOther", m_Renderer.gpuOtherData);
        skinningCompute.SetBuffer(m_KernelCopyBindPose, "_SplatChunks", m_Renderer.gpuChunkData);
        skinningCompute.SetInt("_SplatFormat", m_Renderer.gpuSplatFormat);
        skinningCompute.SetInt("_SplatChunkCount", m_Renderer.gpuChunkCount);
        skinningCompute.SetInt(PropSplatCount, (int)m_SplatCount);

        // Skinning must be inactive during bind pose copy (read from compressed data)
        skinningCompute.SetInt(PropSplatSkinningActive, 0);

        // Output buffer
        skinningCompute.SetBuffer(m_KernelCopyBindPose, PropBindPosePos, m_BufBindPosePos);

        int threadGroups = (m_SplatCount + 255) / 256;
        skinningCompute.Dispatch(m_KernelCopyBindPose, threadGroups, 1, 1);

        if (debugVerbose)
        {
            Debug.Log($"GaussianSplatSkinning: CSCopyBindPose dispatched ({threadGroups} groups, {m_SplatCount} splats)");

            // GPU readback: verify bind-pose positions
            var readback = new Vector3[Mathf.Min(8, m_SplatCount)];
            m_BufBindPosePos.GetData(readback, 0, 0, readback.Length);
            string posStr = "";
            for (int i = 0; i < readback.Length; i++)
                posStr += $"  bindPos[{i}]: ({readback[i].x:F4}, {readback[i].y:F4}, {readback[i].z:F4})\n";
            Debug.Log($"GaussianSplatSkinning: GPU readback of bind-pose positions:\n{posStr}");
        }
    }

    void LateUpdate()
    {
        if (!m_Initialized && !TryInitialize())
            return;

        m_FrameCount++;
        Matrix4x4 splatWorldToLocal = m_Renderer.transform.worldToLocalMatrix;

        // On the first frame, capture bind-inv from the actual runtime skeleton pose.
        if (!m_BindPoseCaptured)
        {
            for (int b = 0; b < m_BoneCount; b++)
            {
                if (m_BoneTransforms[b] != null)
                {
                    Matrix4x4 bindPose = splatWorldToLocal * m_BoneTransforms[b].localToWorldMatrix;
                    m_BindInvMatrices[b] = bindPose.inverse;
                }
                else
                {
                    m_BindInvMatrices[b] = Matrix4x4.identity;
                }
            }
            m_BindPoseCaptured = true;

            if (debugVerbose)
            {
                Debug.Log($"GaussianSplatSkinning: Captured bind-pose. Splat object pos={m_Renderer.transform.position}, rot={m_Renderer.transform.rotation.eulerAngles}");
                Debug.Log($"GaussianSplatSkinning: Animator pos={skeletonAnimator.transform.position}, rot={skeletonAnimator.transform.rotation.eulerAngles}");
                for (int b = 0; b < Mathf.Min(m_BoneCount, 8); b++)
                {
                    if (m_BoneTransforms[b] != null)
                    {
                        Vector3 bonePos = m_BoneTransforms[b].position;
                        Matrix4x4 bInv = m_BindInvMatrices[b];
                        Debug.Log($"  bone[{b}] ({kBoneNames[b]}): worldPos=({bonePos.x:F3},{bonePos.y:F3},{bonePos.z:F3}), " +
                            $"bindInv diag=({bInv.m00:F3},{bInv.m11:F3},{bInv.m22:F3},{bInv.m33:F3}), " +
                            $"bindInv trans=({bInv.m03:F3},{bInv.m13:F3},{bInv.m23:F3})");
                    }
                }
            }
        }

        // Compute skinning matrices: M_skin = boneInSplatSpace * bindInv
        for (int b = 0; b < m_BoneCount; b++)
        {
            if (debugForceIdentity)
            {
                m_BoneMatricesCPU[b] = Matrix4x4.identity;
                continue;
            }

            Matrix4x4 boneInSplatSpace;
            if (m_BoneTransforms[b] != null)
                boneInSplatSpace = splatWorldToLocal * m_BoneTransforms[b].localToWorldMatrix;
            else
                boneInSplatSpace = Matrix4x4.identity;

            m_BoneMatricesCPU[b] = boneInSplatSpace * m_BindInvMatrices[b];
        }

        // Debug: log first few matrices on first frame
        if (debugVerbose && m_FrameCount <= 2)
        {
            for (int b = 0; b < Mathf.Min(m_BoneCount, 4); b++)
            {
                Matrix4x4 M = m_BoneMatricesCPU[b];
                Debug.Log($"GaussianSplatSkinning frame {m_FrameCount}: skinMatrix[{b}] ({kBoneNames[b]}): " +
                    $"diag=({M.m00:F4},{M.m11:F4},{M.m22:F4},{M.m33:F4}), " +
                    $"trans=({M.m03:F4},{M.m13:F4},{M.m23:F4})");
            }
        }

        // Optionally transpose matrices before upload (debug test)
        if (debugTransposeMatrices)
        {
            for (int b = 0; b < m_BoneCount; b++)
                m_BoneMatricesCPU[b] = Matrix4x4.Transpose(m_BoneMatricesCPU[b]);
        }

        // Upload bone matrices to GPU
        m_BufBoneMatrices.SetData(m_BoneMatricesCPU);

        // Dispatch skinning kernel
        skinningCompute.SetBuffer(m_KernelBoneSkinning, PropBindPosePosIn, m_BufBindPosePos);
        skinningCompute.SetBuffer(m_KernelBoneSkinning, PropBoneIndices, m_BufBoneIndices);
        skinningCompute.SetBuffer(m_KernelBoneSkinning, PropBoneWeights, m_BufBoneWeights);
        skinningCompute.SetBuffer(m_KernelBoneSkinning, PropBoneMatrices, m_BufBoneMatrices);
        skinningCompute.SetBuffer(m_KernelBoneSkinning, PropSkinOutPos, m_BufSkinPos);
        skinningCompute.SetInt(PropSplatCount, m_SplatCount);
        skinningCompute.SetInt("_DebugPassthrough", debugGpuPassthrough ? 1 : 0);

        int threadGroups = (m_SplatCount + 255) / 256;
        skinningCompute.Dispatch(m_KernelBoneSkinning, threadGroups, 1, 1);

        // GPU readback: verify skinned positions on first few frames
        if (debugVerbose && m_FrameCount <= 2)
        {
            // Read back bind-pose positions
            var bindReadback = new Vector3[Mathf.Min(4, m_SplatCount)];
            m_BufBindPosePos.GetData(bindReadback, 0, 0, bindReadback.Length);

            // Read back skinned positions
            var skinReadback = new Vector3[Mathf.Min(4, m_SplatCount)];
            m_BufSkinPos.GetData(skinReadback, 0, 0, skinReadback.Length);

            // Read back bone indices and weights for these splats
            var idxReadback = new Vector4[Mathf.Min(4, m_SplatCount)];
            m_BufBoneIndices.GetData(idxReadback, 0, 0, idxReadback.Length);
            var wtReadback = new Vector4[Mathf.Min(4, m_SplatCount)];
            m_BufBoneWeights.GetData(wtReadback, 0, 0, wtReadback.Length);

            // CPU-side verification: compute expected position for first few splats
            string compareStr = "";
            for (int i = 0; i < bindReadback.Length; i++)
            {
                Vector3 bp = bindReadback[i];

                // Decode bone indices from bit pattern (int32 reinterpreted as float32)
                byte[] idxBytes = new byte[16];
                float[] idxFloats = { idxReadback[i].x, idxReadback[i].y, idxReadback[i].z, idxReadback[i].w };
                System.Buffer.BlockCopy(idxFloats, 0, idxBytes, 0, 16);
                int[] boneIdx = new int[4];
                System.Buffer.BlockCopy(idxBytes, 0, boneIdx, 0, 16);

                float[] wt = { wtReadback[i].x, wtReadback[i].y, wtReadback[i].z, wtReadback[i].w };

                // Compute blended matrix on CPU (same logic as compute shader)
                Matrix4x4 blended = Matrix4x4.zero;
                float totalWeight = 0;
                for (int j = 0; j < 4; j++)
                {
                    if (wt[j] > 0.001f && boneIdx[j] >= 0 && boneIdx[j] < m_BoneCount)
                    {
                        Matrix4x4 boneMat = m_BoneMatricesCPU[boneIdx[j]];
                        // Note: m_BoneMatricesCPU may have been transposed if debugTransposeMatrices is on
                        // Use the pre-transpose values for CPU comparison
                        for (int r = 0; r < 4; r++)
                            for (int c = 0; c < 4; c++)
                                blended[r, c] += wt[j] * boneMat[r, c];
                        totalWeight += wt[j];
                    }
                }

                Vector3 cpuExpected;
                if (totalWeight > 0.001f)
                {
                    for (int r = 0; r < 4; r++)
                        for (int c = 0; c < 4; c++)
                            blended[r, c] /= totalWeight;
                    // GPU does manual column multiply: pos.x*col0 + pos.y*col1 + pos.z*col2 + col3
                    // which equals standard M * pos
                    cpuExpected = blended.MultiplyPoint3x4(bp);
                }
                else
                {
                    cpuExpected = bp;
                }

                compareStr += $"  splat[{i}]: bone[{boneIdx[0]}] w={wt[0]:F3}\n" +
                    $"    bindPos=({bp.x:F4},{bp.y:F4},{bp.z:F4})\n" +
                    $"    cpuExp =({cpuExpected.x:F4},{cpuExpected.y:F4},{cpuExpected.z:F4})\n" +
                    $"    gpuOut =({skinReadback[i].x:F4},{skinReadback[i].y:F4},{skinReadback[i].z:F4})\n";
            }
            Debug.Log($"GaussianSplatSkinning frame {m_FrameCount}: CPU vs GPU position comparison:\n{compareStr}");
        }

        // Activate skinning ONLY AFTER m_BufSkinPos has valid data
        if (!m_SkinningActivated)
        {
            Shader.SetGlobalBuffer(PropSplatSkinPos, m_BufSkinPos);
            Shader.SetGlobalInt(PropSplatSkinningActive, 1);
            GaussianSplatRenderer.s_SkinPosOverride = m_BufSkinPos;
            GaussianSplatRenderer.s_SkinningActive = 1;
            m_SkinningActivated = true;
            Debug.Log("GaussianSplatSkinning: Skinning activated (after first valid dispatch)");
        }
        else
        {
            // Ensure globals stay set
            Shader.SetGlobalBuffer(PropSplatSkinPos, m_BufSkinPos);
            Shader.SetGlobalInt(PropSplatSkinningActive, 1);
            GaussianSplatRenderer.s_SkinPosOverride = m_BufSkinPos;
            GaussianSplatRenderer.s_SkinningActive = 1;
        }
    }

    void OnDisable()
    {
        Shader.SetGlobalInt(PropSplatSkinningActive, 0);
        GaussianSplatRenderer.s_SkinPosOverride = null;
        GaussianSplatRenderer.s_SkinRotOverride = null;
        GaussianSplatRenderer.s_SkinningActive = 0;
        m_SkinningActivated = false;

        m_BufBoneIndices?.Dispose();
        m_BufBoneWeights?.Dispose();
        m_BufBindPosePos?.Dispose();
        m_BufBoneMatrices?.Dispose();
        m_BufSkinPos?.Dispose();

        m_BufBoneIndices = null;
        m_BufBoneWeights = null;
        m_BufBindPosePos = null;
        m_BufBoneMatrices = null;
        m_BufSkinPos = null;

        m_Initialized = false;
        m_BindPoseCaptured = false;
    }
}
