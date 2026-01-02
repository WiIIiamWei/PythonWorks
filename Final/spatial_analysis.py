import os
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csr_matrix
from glob import glob
from tqdm import tqdm
import logging

plt.rcParams['font.sans-serif'] = ['HarmonyOS Sans SC']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizedSpatialAnalyzer:  
    def __init__(self, pred_dir="./Output", window_size=32, distance_threshold=64):
        self.pred_dir = pred_dir
        self.window_size = window_size
        self.distance_threshold = distance_threshold
        self.all_densities = None
        self.all_coords = None
        self.image_ids = None
        
    def load_all_predictions(self):
        """
        加载所有预测文件
        """
        pred_files = sorted(glob(os.path.join(self.pred_dir, "*.tif")))
        
        if not pred_files:
            logger.error(f"未找到预测文件在 {self.pred_dir}")
            return None, None, None
        
        logger.info(f"找到 {len(pred_files)} 张预测图像")
        
        all_densities = []
        all_coords = []
        image_info = []
        
        for idx, pred_file in enumerate(tqdm(pred_files, desc="加载预测图像")):
            try:
                with rasterio.open(pred_file) as src:
                    pred_mask = src.read(1)
                
                density, coords = self.calculate_density(pred_mask)
                
                all_densities.append(density.flatten())
                all_coords.append(coords)
                
                image_info.append({
                    'image_id': idx,
                    'filename': os.path.basename(pred_file),
                    'n_cells': len(coords),
                    'mean_density': density.mean(),
                    'max_density': density.max(),
                    'min_density': density.min()
                })
                
            except Exception as e:
                logger.warning(f"处理 {pred_file} 失败: {str(e)}")
                continue
        
        if not all_densities:
            logger.error("未成功加载任何预测图像")
            return None, None, None
        
        # 合并所有数据
        merged_densities = np.concatenate(all_densities)
        merged_coords = np.vstack(all_coords)
        image_ids = np.repeat(
            np.arange(len(all_densities)), 
            [len(d) for d in all_densities]
        )
        
        self.all_densities = merged_densities
        self.all_coords = merged_coords
        self.image_ids = image_ids
        
        info_df = pd.DataFrame(image_info)
        
        logger.info(f"✓ 总样本单元数: {len(merged_densities)}")
        logger.info(f"✓ 平均单元密度: {merged_densities.mean():.4f}")
        logger.info(f"✓ 密度范围: [{merged_densities.min():.4f}, {merged_densities.max():.4f}]")
        
        return merged_densities, merged_coords, info_df
    
    def calculate_density(self, mask_array, window_size=None):
        """
        计算建筑物密度
        """
        if window_size is None:
            window_size = self.window_size
        
        H, W = mask_array.shape
        
        n_rows = (H - window_size) // window_size + 1
        n_cols = (W - window_size) // window_size + 1
        
        density = np.zeros((n_rows, n_cols))
        coords = []
        
        for i in range(n_rows):
            for j in range(n_cols):
                y0 = i * window_size
                x0 = j * window_size
                y1 = min(y0 + window_size, H)
                x1 = min(x0 + window_size, W)
                
                window = mask_array[y0:y1, x0:x1]
                density[i, j] = np.sum(window > 0) / window.size
                coords.append((x0 + window_size // 2, y0 + window_size // 2))
        
        return density, np.array(coords)
    
    def compute_morans_i(self, values, coords, distance_threshold=None):
        """
        最优化的Moran's I计算 - 使用KDTree（内存<1GB）
        
        关键改进：
        1. 不计算完整距离矩阵（原方案需要398GB内存）
        2. 使用KDTree进行邻接查询（仅保存邻接关系）
        3. 使用稀疏矩阵存储权重（内存占用<1GB）
        """
        if distance_threshold is None:
            distance_threshold = self.distance_threshold
        
        n = len(values)
        logger.info(f"【计算全局Moran's I】")
        logger.info(f"样本单元数: {n}")
        logger.info(f"距离阈值: {distance_threshold}")
        
        # 标准化
        z = (values - np.mean(values)) / np.std(values)
        
        # 使用KDTree构建邻接关系（内存高效）
        logger.info("步骤1：构建KDTree索引...")
        tree = cKDTree(coords)
        
        logger.info(f"步骤2：查询邻接关系...")
        neighbor_indices = tree.query_ball_tree(tree, r=distance_threshold)
        
        logger.info(f"步骤3：构建稀疏权重矩阵...")
        W = lil_matrix((n, n), dtype=np.float32)  # 使用float32节省内存
        
        edge_count = 0
        for i, neighbors in enumerate(neighbor_indices):
            for j in neighbors:
                if i != j:  # 排除自己
                    W[i, j] = 1.0
                    edge_count += 1
        
        W = W.tocsr()  # 转换为压缩稀疏格式
        
        logger.info(f"✓ 权重矩阵: {edge_count:,} 条边")
        logger.info(f"✓ 矩阵稀疏度: {100*edge_count/(n*n):.4f}%")
        logger.info(f"✓ 内存占用: ~{edge_count * 12 / 1024 / 1024:.1f} MB (包括索引)")
        
        # 行标准化
        logger.info("步骤4：行标准化权重矩阵...")
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums = np.maximum(row_sums, 1e-10)
        W_norm = W.multiply(1.0 / row_sums[:, np.newaxis])
        
        # 计算Moran's I
        logger.info("步骤5：计算Moran's I指数...")
        Wz = W_norm @ z
        numerator = np.dot(z, Wz)
        denominator = np.dot(z, z)
        
        morans_i = (n / W.sum()) * (numerator / denominator)
        
        # 期望值和方差
        expected_i = -1.0 / (n - 1)
        
        logger.info("步骤6：计算方差和显著性...")
        z2 = np.sum(z ** 2)
        z4 = np.sum(z ** 4)
        b2 = (z4 / n) / ((z2 / n) ** 2)
        
        S1 = 0.5 * np.sum((W + W.T) ** 2)
        S2 = np.sum(np.array(W.sum(axis=1)).flatten() ** 2) + \
             np.sum(np.array(W.sum(axis=0)).flatten() ** 2)
        
        S0 = W.sum()
        
        variance = ((n * S1 - b2 * S2 + 3 * S0 ** 2) / 
                   ((n - 1) * S0 ** 2)) - expected_i ** 2
        
        variance = max(variance, 1e-10)
        std = np.sqrt(variance)
        z_score = (morans_i - expected_i) / std
        
        # p值计算（双尾）
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        logger.info(f"【Moran's I 计算结果】")
        logger.info(f"✓ Moran's I  = {morans_i:.6f}")
        logger.info(f"✓ E(I)       = {expected_i:.6f}")
        logger.info(f"✓ Var(I)     = {variance:.6f}")
        logger.info(f"✓ Z-score    = {z_score:.4f}")
        logger.info(f"✓ P-value    = {p_value:.6f}")
        
        return {
            'morans_i': morans_i,
            'expected_i': expected_i,
            'variance': variance,
            'z_score': z_score,
            'p_value': p_value,
            'n_units': n
        }
    
    def identify_lisa_clusters(self, values, coords):
        """
        LISA局部聚类分析
        """
        logger.info(f"【LISA 局部聚类分析】")
        
        n = len(values)
        z = (values - np.mean(values)) / np.std(values)
        
        # 使用KDTree构建权重矩阵
        tree = cKDTree(coords)
        neighbor_indices = tree.query_ball_tree(tree, r=self.distance_threshold)
        
        W = lil_matrix((n, n), dtype=np.float32)
        for i, neighbors in enumerate(neighbor_indices):
            for j in neighbors:
                if i != j:
                    W[i, j] = 1.0
        
        W = W.tocsr()
        
        # 行标准化
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums = np.maximum(row_sums, 1e-10)
        W_norm = W.multiply(1.0 / row_sums[:, np.newaxis])
        
        # 局部Moran's I
        lisa_i = z * (W_norm @ z)
        
        # 聚类分类
        mean_z = np.mean(z)
        mean_lisa = np.mean(lisa_i)
        
        cluster_types = np.zeros(n, dtype=int)
        for i in range(n):
            if lisa_i[i] > mean_lisa:
                cluster_types[i] = 1 if z[i] > mean_z else 2  # HH or LL
            else:
                cluster_types[i] = 4 if z[i] > mean_z else 3  # LH or HL
        
        cluster_counts = {
            'HH': np.sum(cluster_types == 1),
            'LL': np.sum(cluster_types == 2),
            'HL': np.sum(cluster_types == 3),
            'LH': np.sum(cluster_types == 4)
        }
        
        logger.info("LISA聚类结果:")
        for ctype, count in cluster_counts.items():
            percentage = 100 * count / n
            logger.info(f"  {ctype}: {count:6d} ({percentage:5.1f}%)")
        
        return cluster_types, cluster_counts
    
    def visualize_results(self, values, coords, cluster_types, output_dir="./output"):
        """
        可视化结果
        """
        logger.info(f"【生成可视化结果】")
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. 密度散点图
        ax = axes[0, 0]
        scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                           c=values, cmap='YlOrRd', s=20, alpha=0.6)
        ax.set_title(f'全局建筑物密度分布\n(所有图像聚合，样本数={len(values)})', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('X坐标（像素）')
        ax.set_ylabel('Y坐标（像素）')
        plt.colorbar(scatter, ax=ax, label='建筑物密度')
        
        # 2. 密度直方图
        ax = axes[0, 1]
        ax.hist(values, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, 
                  label=f'均值={np.mean(values):.3f}')
        ax.set_xlabel('建筑物密度')
        ax.set_ylabel('单元数')
        ax.set_title('建筑物密度分布直方图', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. LISA聚类分布
        ax = axes[1, 0]
        colors = ['gray', 'red', 'blue', 'orange', 'green']
        labels = ['不显著', 'HH(高聚)', 'LL(低聚)', 'HL(异常)', 'LH(异常)']
        
        for cluster_type in range(1, 5):
            mask = cluster_types == cluster_type
            if np.any(mask):
                ax.scatter(coords[mask, 0], coords[mask, 1], 
                         c=colors[cluster_type], label=labels[cluster_type],
                         s=20, alpha=0.7)
        
        ax.set_title('LISA聚类分析结果\n(空间聚类分布)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X坐标（像素）')
        ax.set_ylabel('Y坐标（像素）')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
        
        # 4. Moran散点图
        ax = axes[1, 1]
        z = (values - np.mean(values)) / np.std(values)
        
        tree = cKDTree(coords)
        neighbor_indices = tree.query_ball_tree(tree, r=self.distance_threshold)
        
        W = lil_matrix((len(values), len(values)), dtype=np.float32)
        for i, neighbors in enumerate(neighbor_indices):
            for j in neighbors:
                if i != j:
                    W[i, j] = 1.0
        
        W = W.tocsr()
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums = np.maximum(row_sums, 1e-10)
        W_norm = W.multiply(1.0 / row_sums[:, np.newaxis])
        
        Wz = W_norm @ z
        
        colors_map = {0: 'gray', 1: 'red', 2: 'blue', 3: 'orange', 4: 'green'}
        for ctype in range(1, 5):
            mask = cluster_types == ctype
            if np.any(mask):
                ax.scatter(z[mask], Wz[mask], 
                         color=colors_map[ctype], alpha=0.6, s=20,
                         label=labels[ctype])
        
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('标准化建筑物密度 (Z)')
        ax.set_ylabel('邻接加权密度均值 (WZ)')
        ax.set_title('Moran散点图\n(显示聚类类型)', fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'spatial_analysis_optimized.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ 可视化已保存: {output_path}")
        plt.close()
    
    def run_full_analysis(self):
        """
        执行完整分析流程
        """
        logger.info("="*80)
        logger.info(" 开始优化版本多图像空间统计分析（KDTree邻接，内存<1GB）")
        logger.info("="*80)
        
        # 1. 加载数据
        logger.info("【步骤1】加载所有预测图像")
        logger.info("-"*80)
        densities, coords, info_df = self.load_all_predictions()
        
        if densities is None:
            return None
        
        print(f"【数据统计信息】")
        print(info_df[['image_id', 'filename', 'n_cells', 'mean_density', 'max_density']].head(10))
        print(f"总样本单元数: {len(densities):,}")
        
        # 2. 计算全局Moran's I
        logger.info("【步骤2】计算全局Moran's I")
        logger.info("-"*80)
        global_results = self.compute_morans_i(densities, coords)
        
        # 3. LISA聚类
        logger.info("【步骤3】进行LISA聚类分析")
        logger.info("-"*80)
        cluster_types, cluster_counts = self.identify_lisa_clusters(densities, coords)
        
        # 4. 可视化
        logger.info("【步骤4】生成可视化结果")
        logger.info("-"*80)
        self.visualize_results(densities, coords, cluster_types)
        
        # 5. 输出最终结果
        logger.info("="*80)
        logger.info(" 【最终分析结果汇总】")
        logger.info("="*80)
        
        print(f"Moran's I指数:  {global_results['morans_i']:.6f}")
        print(f"期望值 E(I):    {global_results['expected_i']:.6f}")
        print(f"Z-score:        {global_results['z_score']:.4f}")
        print(f"P-value:        {global_results['p_value']:.6f}")
        print(f"样本单元数:      {global_results['n_units']:,}")
        
        # 显著性判定
        print(f"【显著性判定】")
        if global_results['p_value'] < 0.05:
            print(f"✓ 在α=0.05水平下，拒绝零假设（显著）")
            if global_results['morans_i'] > 0:
                print(f"✓ 该地区建筑物密度存在显著的【正自相关】")
                print(f"✓ 表现为【聚集分布】特征")
            else:
                print(f"✓ 该地区建筑物密度存在显著的【负自相关】")
                print(f"✓ 表现为【离散分布】特征")
        else:
            print(f"✗ 在α=0.05水平下，无法拒绝零假设（不显著）")
            print(f"✗ 该地区建筑物密度空间分布无显著相关性（随机分布）")
        
        print(f"【LISA聚类分布汇总】")
        total = sum(cluster_counts.values())
        for ctype, count in cluster_counts.items():
            print(f"  {ctype}: {count:6d} ({100*count/total:5.1f}%)")
        
        logger.info("" + "="*80)
        logger.info(" 分析完成！")
        logger.info("="*80)
        
        return global_results, cluster_types


def main():
    analyzer = OptimizedSpatialAnalyzer(
        pred_dir="C:\\Users\\13977\\Desktop\\Homework\\Code\\Python\\PythonWorks\\Final\\Analyze",
        window_size=32,
        distance_threshold=64
    )
    
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()