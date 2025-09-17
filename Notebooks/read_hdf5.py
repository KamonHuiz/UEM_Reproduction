import h5py
#ACTITVITYNET
VISUAL_FEATURE_ACT = r"D:\Workspace\UEM\Data\activitynet\FeatureData\new_clip_vit_32_activitynet_vid_features.hdf5"
TEXT_FEATURE_ACT = r"D:\Workspace\UEM\Data\activitynet\TextData\clip_ViT_B_32_activitynet_query_feat.hdf5"

#TVR
VISUAL_FEATURE_TVR = r"D:\Workspace\UEM\Data\tvr\FeatureData\new_clip_vit_32_tvr_vid_features.hdf5"
TEXT_FEATURE_TVR = r"D:\Workspace\UEM\Data\tvr\TextData\clip_ViT_B_32_tvr_query_feat.hdf5"

# mở file hdf5 (read-only)
with h5py.File(TEXT_FEATURE_TVR, "r") as f:
    for i in list(f.keys())[11:20]:
        print(f"This is id: {i}, Shape: {f[str(i)].shape}")

    
