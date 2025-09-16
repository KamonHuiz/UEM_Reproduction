import h5py

# mở file hdf5 (read-only)
with h5py.File("data.h5", "r") as f:
    # liệt kê các "keys" ở cấp cao nhất
    print("Keys:", list(f.keys()))

    # giả sử có dataset tên "images"
    data = f["images"]

    # xem shape và dtype
    print("Shape:", data.shape)
    print("Dtype:", data.dtype)

    # đọc vài phần tử đầu tiên (giống như đọc "dòng đầu")
    print(data[:5])
