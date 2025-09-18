import h5py
#ACTITVITYNET
VISUAL_FEATURE_ACT = r"D:\Workspace\UEM\Data\activitynet\FeatureData\new_clip_vit_32_activitynet_vid_features.hdf5"
TEXT_FEATURE_ACT = r"D:\Workspace\UEM\Data\activitynet\TextData\clip_ViT_B_32_activitynet_query_feat.hdf5"

#TVR
VISUAL_FEATURE_TVR = r"D:\Workspace\UEM\Data\tvr\FeatureData\new_clip_vit_32_tvr_vid_features.hdf5"
TEXT_FEATURE_TVR = r"D:\Workspace\UEM\Data\tvr\TextData\clip_ViT_B_32_tvr_query_feat.hdf5"

#WORD_FEARTURE
TVR_W_F = r"D:\Workspace\UEM\Data\tvr_clip-B32_text_word_feats.hdf5"
ACT_W_F_VAL = r"D:\Workspace\UEM\Data\clip_ViT_B_32_activitynet_query_feat_val.hdf5"
ACT_W_F_TRAIN = r"D:\Workspace\UEM\Data\clip_ViT_B_32_activitynet_query_feat_train.hdf5"

# mở file hdf5 (read-only)
with h5py.File(ACT_W_F_VAL, "r") as f:
    for i in list(f.keys())[11:20]:
        print(f"This is id: {i}, Shape: {f[str(i)].shape}")

    
