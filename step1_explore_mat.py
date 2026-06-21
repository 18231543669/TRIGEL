"""
Step 1: 探测 .mat 文件结构
运行方式: python step1_explore_mat.py your_data.mat
把输出结果贴给我，我帮你写转换脚本
"""

import sys
import os

def explore_mat(mat_path):
    print(f"{'='*60}")
    print(f"探测文件: {mat_path}")
    print(f"文件大小: {os.path.getsize(mat_path) / 1024 / 1024:.1f} MB")
    print(f"{'='*60}\n")

    # 尝试用 scipy 加载 (v5 及以下的 .mat)
    try:
        import scipy.io as sio
        mat = sio.loadmat(mat_path)
        print("✓ 使用 scipy.io.loadmat 成功加载 (MATLAB v5 格式)\n")

        print(f"所有 key: {list(mat.keys())}\n")

        for key in sorted(mat.keys()):
            if key.startswith('__'):
                continue
            val = mat[key]
            print(f"--- 变量: '{key}' ---")
            print(f"  类型: {type(val).__name__}")

            if hasattr(val, 'shape'):
                print(f"  shape: {val.shape}")
                print(f"  dtype: {val.dtype}")

                # 如果是 object 类型 (cell array)
                if val.dtype == object:
                    print(f"  [cell array] 共 {val.size} 个元素:")
                    for idx in range(min(val.size, 10)):
                        elem = val.flat[idx]
                        if hasattr(elem, 'shape'):
                            print(f"    [{idx}] shape={elem.shape}, dtype={elem.dtype}")
                        else:
                            print(f"    [{idx}] type={type(elem).__name__}, value={elem}")
                else:
                    # 普通数组，打印前几个值
                    flat = val.flatten()
                    if flat.size <= 20:
                        print(f"  值: {flat}")
                    else:
                        print(f"  前10个值: {flat[:10]}")
                        print(f"  唯一值数量: {len(set(flat))}")
                        if len(set(flat)) <= 30:
                            unique_sorted = sorted(set(flat))
                            print(f"  唯一值: {unique_sorted}")
            print()
        return

    except NotImplementedError:
        print("⚠ scipy.io.loadmat 不支持, 尝试 HDF5 格式...\n")
    except Exception as e:
        print(f"⚠ scipy.io.loadmat 失败: {e}\n尝试 HDF5 格式...\n")

    # 尝试用 h5py 加载 (v7.3 的 .mat)
    try:
        import h5py
        f = h5py.File(mat_path, 'r')
        print("✓ 使用 h5py 成功加载 (MATLAB v7.3 / HDF5 格式)\n")

        def explore_hdf5(group, prefix=""):
            for key in sorted(group.keys()):
                item = group[key]
                full_key = f"{prefix}/{key}" if prefix else key
                if isinstance(item, h5py.Group):
                    print(f"--- 组(Group): '{full_key}' ---")
                    print(f"  子项: {list(item.keys())}")
                    explore_hdf5(item, full_key)
                elif isinstance(item, h5py.Dataset):
                    print(f"--- 变量: '{full_key}' ---")
                    print(f"  shape: {item.shape}")
                    print(f"  dtype: {item.dtype}")

                    # 如果是 object reference 类型 (cell array)
                    if item.dtype == h5py.ref_dtype:
                        print(f"  [cell array / object references] 共 {item.size} 个引用")
                        refs = item[()]
                        flat_refs = refs.flatten()
                        for idx in range(min(len(flat_refs), 10)):
                            try:
                                deref = f[flat_refs[idx]]
                                if isinstance(deref, h5py.Dataset):
                                    print(f"    [{idx}] shape={deref.shape}, dtype={deref.dtype}")
                                else:
                                    print(f"    [{idx}] type={type(deref).__name__}")
                            except Exception as ex:
                                print(f"    [{idx}] 解引用失败: {ex}")
                    else:
                        # 普通数组
                        if item.size <= 20:
                            data = item[()].flatten()
                            print(f"  值: {data}")
                        else:
                            data = item[()].flatten()
                            print(f"  前10个值: {data[:10]}")
                            unique_count = len(set(data))
                            print(f"  唯一值数量: {unique_count}")
                            if unique_count <= 30:
                                print(f"  唯一值: {sorted(set(data))}")
                    print()

        print(f"顶层 keys: {list(f.keys())}\n")
        explore_hdf5(f)
        f.close()
        return

    except ImportError:
        print("✗ h5py 未安装, 请运行: pip install h5py")
    except Exception as e:
        print(f"✗ h5py 加载也失败: {e}")

    print("\n无法加载该 .mat 文件，请检查文件是否完整。")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python step1_explore_mat.py <your_file.mat>")
        sys.exit(1)
    explore_mat(sys.argv[1])
