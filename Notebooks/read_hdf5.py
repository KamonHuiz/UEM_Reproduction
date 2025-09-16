import h5py

# mở file hdf5 (read-only)
with h5py.File(r"D:\Download\activitynet\FeatureData\new_clip_vit_32_activitynet_vid_features.hdf5", "r") as f:
    for i in list(f.keys())[11:20]:
        print(f"This is id: {i}, Shape: {f[str(i)].shape}")

    
