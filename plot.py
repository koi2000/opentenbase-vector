import matplotlib.pyplot as plt
import numpy as np

# 数据（同前）
data = {
    'OpenTenbase': [
        {'m': 16, 'ef_search': 10,  'recall': 0.935, 'qps': 760.833},
        {'m': 16, 'ef_search': 20,  'recall': 0.979, 'qps': 584.853},
        {'m': 16, 'ef_search': 40,  'recall': 0.996, 'qps': 419.077},
        {'m': 16, 'ef_search': 80,  'recall': 0.999, 'qps': 288.759},
        {'m': 16, 'ef_search': 120, 'recall': 1.000, 'qps': 228.679},
        {'m': 16, 'ef_search': 200, 'recall': 1.000, 'qps': 167.557},
        {'m': 16, 'ef_search': 400, 'recall': 1.000, 'qps': 105.823},
        {'m': 16, 'ef_search': 800, 'recall': 1.000, 'qps': 64.858},
        {'m': 24, 'ef_search': 10,  'recall': 0.948, 'qps': 620.224},
        {'m': 24, 'ef_search': 20,  'recall': 0.985, 'qps': 478.108},
        {'m': 24, 'ef_search': 40,  'recall': 0.997, 'qps': 347.058},
        {'m': 24, 'ef_search': 80,  'recall': 0.999, 'qps': 239.282},
        {'m': 24, 'ef_search': 120, 'recall': 1.000, 'qps': 190.331},
        {'m': 24, 'ef_search': 200, 'recall': 1.000, 'qps': 139.841},
        {'m': 24, 'ef_search': 400, 'recall': 1.000, 'qps': 89.793},
        {'m': 24, 'ef_search': 800, 'recall': 1.000, 'qps': 55.531},
    ],
    'PGVector': [
        {'m': 16, 'ef_search': 10,  'recall': 0.968, 'qps': 3537.001},
        {'m': 16, 'ef_search': 20,  'recall': 0.989, 'qps': 2752.909},
        {'m': 16, 'ef_search': 40,  'recall': 0.997, 'qps': 2083.907},
        {'m': 16, 'ef_search': 80,  'recall': 0.999, 'qps': 1434.570},
        {'m': 16, 'ef_search': 120, 'recall': 1.000, 'qps': 1116.043},
        {'m': 16, 'ef_search': 200, 'recall': 1.000, 'qps': 789.983},
        {'m': 16, 'ef_search': 400, 'recall': 1.000, 'qps': 477.241},
        {'m': 16, 'ef_search': 800, 'recall': 1.000, 'qps': 278.101},
        {'m': 24, 'ef_search': 10,  'recall': 0.980, 'qps': 3039.511},
        {'m': 24, 'ef_search': 20,  'recall': 0.993, 'qps': 2340.378},
        {'m': 24, 'ef_search': 40,  'recall': 0.998, 'qps': 1701.742},
        {'m': 24, 'ef_search': 80,  'recall': 1.000, 'qps': 1167.392},
        {'m': 24, 'ef_search': 120, 'recall': 1.000, 'qps': 914.632},
        {'m': 24, 'ef_search': 200, 'recall': 1.000, 'qps': 655.881},
        {'m': 24, 'ef_search': 400, 'recall': 1.000, 'qps': 400.304},
        {'m': 24, 'ef_search': 800, 'recall': 1.000, 'qps': 236.527},
    ]
}

# 获取所有唯一的 m 值
m_values = sorted(set(item['m'] for item in data['OpenTenbase']))

# 设置绘图风格
plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(len(m_values), 2, figsize=(12, 5 * len(m_values)), sharex='col')
if len(m_values) == 1:
    axes = axes.reshape(1, -1)

# 颜色和标记
colors = {'OpenTenbase': 'red', 'PGVector': 'blue'}
markers = {'OpenTenbase': 'o', 'PGVector': 's'}

# 按 m 分组绘图
for row_idx, m in enumerate(m_values):
    ax_recall = axes[row_idx, 0]  # Recall 子图
    ax_qps = axes[row_idx, 1]    # QPS 子图

    for system in ['OpenTenbase', 'PGVector']:
        # 提取当前 m 和 system 的数据
        subset = [item for item in data[system] if item['m'] == m]
        ef_search_vals = [item['ef_search'] for item in subset]
        recall_vals = [item['recall'] for item in subset]
        qps_vals = [item['qps'] for item in subset]

        # 绘制 Recall 曲线
        ax_recall.plot(ef_search_vals, recall_vals,
                       label=system, color=colors[system], marker=markers[system], markersize=6)

        # 绘制 QPS 曲线
        ax_qps.plot(ef_search_vals, qps_vals,
                    label=system, color=colors[system], marker=markers[system], markersize=6)

    # 设置 Recall 子图
    ax_recall.set_title(f'm={m}, ef_construction=200 — Recall@k')
    ax_recall.set_ylabel('Recall@k')
    ax_recall.grid(True, linestyle='--', alpha=0.5)
    ax_recall.legend()
    ax_recall.set_ylim(0.9, 1.01)

    # 设置 QPS 子图
    ax_qps.set_title(f'm={m}, ef_construction=200 — Query Throughput (QPS)')
    ax_qps.set_ylabel('QPS')
    ax_qps.set_xlabel('ef_search')
    ax_qps.grid(True, linestyle='--', alpha=0.5)
    ax_qps.legend()
    ax_qps.set_yscale('log')  # QPS 跨度大，使用对数坐标

# 通用设置
plt.tight_layout()

# 🔽 保存图像
plt.savefig('ann_benchmark_grouped_by_m.png', dpi=300, bbox_inches='tight')
# plt.savefig('ann_benchmark_grouped_by_m.pdf', bbox_inches='tight')

print("✅ 图像已保存为：")
print("   - ann_benchmark_grouped_by_m.png")
# print("   - ann_benchmark_grouped_by_m.pdf")

# 显示图像
# plt.show()