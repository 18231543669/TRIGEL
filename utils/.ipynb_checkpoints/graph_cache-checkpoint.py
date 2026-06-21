# 创建新文件: utils/graph_cache.py
"""
KNN图缓存管理器 - 全局缓存，避免重复计算
"""
import hashlib
import numpy as np
from typing import Dict, List, Optional

class GraphCacheManager:
    """全局KNN图缓存管理器（单例模式）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}  # 缓存字典
            cls._instance._hits = 0    # 命中次数
            cls._instance._misses = 0  # 未命中次数
        return cls._instance
    
    def _generate_key(self, data: np.ndarray, k: int) -> str:
        """生成缓存键：使用数据的形状和少量统计信息"""
        # 使用数据的形状、均值和前几个值生成哈希
        shape_str = f"{data.shape}"
        
        # 采样部分数据生成哈希（避免计算整个数据的哈希）
        sample_size = min(100, len(data))
        if len(data) > 0:
            sample_data = data[:sample_size].flatten()
            # 使用统计信息
            stats = np.array([
                np.mean(sample_data),
                np.std(sample_data),
                np.min(sample_data),
                np.max(sample_data)
            ])
            stats_str = f"{stats.tobytes().hex()[:20]}"
        else:
            stats_str = "empty"
        
        key_str = f"knn_{shape_str}_k{k}_{stats_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, data: np.ndarray, k: int) -> Optional[object]:
        """从缓存获取KNN图"""
        cache_key = self._generate_key(data, k)
        if cache_key in self._cache:
            self._hits += 1
            return self._cache[cache_key]
        self._misses += 1
        return None
    
    def set(self, data: np.ndarray, k: int, graph_data: object) -> str:
        """保存KNN图到缓存"""
        cache_key = self._generate_key(data, k)
        self._cache[cache_key] = graph_data
        return cache_key
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def stats(self) -> Dict:
        """获取缓存统计信息"""
        total = self._hits + self._misses
        return {
            'cache_size': len(self._cache),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total > 0 else 0,
            'total_queries': total
        }
    
    def print_stats(self):
        """打印缓存统计"""
        stats = self.stats()
        print(f"\n📊 KNN图缓存统计:")
        print(f"  缓存图数量: {stats['cache_size']}")
        print(f"  命中次数: {stats['hits']}")
        print(f"  未命中次数: {stats['misses']}")
        print(f"  命中率: {stats['hit_rate']:.1%}")