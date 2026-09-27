# -*- coding: utf-8 -*-
"""
Matplotlib 图表生成器
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib

from ..metrics.models import AggregatedMetrics

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class ChartGenerator:
    """图表生成器"""

    color = "#4CAF50"  # 绿色

    def generate_all_charts(
        self,
        asr_results: List[AggregatedMetrics],
        output_dir: Path,
        timestamp: str,
    ) -> List[Path]:
        """
        生成所有图表

        Args:
            asr_results: ASR 测试结果
            output_dir: 输出目录
            timestamp: 时间戳

        Returns:
            生成的图表文件路径列表
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []

        # 1. 首次延迟图
        path = output_dir / f"first_latency_{timestamp}.png"
        self._generate_first_latency_chart(asr_results, path)
        generated_files.append(path)

        # 2. RTF 图
        path = output_dir / f"rtf_{timestamp}.png"
        self._generate_rtf_chart(asr_results, path)
        generated_files.append(path)

        # 3. 吞吐量图
        path = output_dir / f"throughput_{timestamp}.png"
        self._generate_throughput_chart(asr_results, path)
        generated_files.append(path)

        # 4. 总时间图
        path = output_dir / f"total_time_{timestamp}.png"
        self._generate_total_time_chart(asr_results, path)
        generated_files.append(path)

        return generated_files

    def _generate_first_latency_chart(
        self,
        asr_results: List[AggregatedMetrics],
        output_path: Path,
    ) -> None:
        """生成首次延迟图"""
        _fig, ax = plt.subplots(figsize=(10, 6))

        levels = [r.concurrency_level for r in asr_results]
        avg_values = [r.first_latency_avg for r in asr_results]
        p95_values = [r.first_latency_p95 for r in asr_results]

        ax.plot(levels, avg_values, 'o-', color=self.color,
               label='ASR 首次响应 (Avg)', linewidth=2, markersize=8)
        ax.plot(levels, p95_values, 's--', color=self.color,
               label='ASR 首次响应 (P95)', linewidth=1.5, markersize=6, alpha=0.7)

        ax.set_xlabel('并发数', fontsize=12)
        ax.set_ylabel('延迟 (ms)', fontsize=12)
        ax.set_title('首次响应延迟 vs 并发数', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(levels)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _generate_rtf_chart(
        self,
        asr_results: List[AggregatedMetrics],
        output_path: Path,
    ) -> None:
        """生成 RTF 图"""
        _fig, ax = plt.subplots(figsize=(10, 6))

        levels = [r.concurrency_level for r in asr_results]
        avg_values = [r.rtf_avg for r in asr_results]
        p95_values = [r.rtf_p95 for r in asr_results]

        ax.plot(levels, avg_values, 'o-', color=self.color,
               label='ASR RTF (Avg)', linewidth=2, markersize=8)
        ax.plot(levels, p95_values, 's--', color=self.color,
               label='ASR RTF (P95)', linewidth=1.5, markersize=6, alpha=0.7)

        # 添加 RTF=1.0 参考线
        ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1.5,
                  label='RTF = 1.0 (实时)')

        ax.set_xlabel('并发数', fontsize=12)
        ax.set_ylabel('RTF', fontsize=12)
        ax.set_title('RTF vs 并发数', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _generate_throughput_chart(
        self,
        asr_results: List[AggregatedMetrics],
        output_path: Path,
    ) -> None:
        """生成吞吐量柱状图"""
        _fig, ax = plt.subplots(figsize=(10, 6))

        labels = [str(r.concurrency_level) for r in asr_results]
        ax.bar(labels, [r.throughput for r in asr_results], label='ASR',
              color=self.color, alpha=0.8)

        ax.set_xlabel('并发数', fontsize=12)
        ax.set_ylabel('吞吐量 (req/s)', fontsize=12)
        ax.set_title('吞吐量 vs 并发数', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _generate_total_time_chart(
        self,
        asr_results: List[AggregatedMetrics],
        output_path: Path,
    ) -> None:
        """生成总时间图"""
        _fig, ax = plt.subplots(figsize=(10, 6))

        levels = [r.concurrency_level for r in asr_results]
        avg_values = [r.total_time_avg for r in asr_results]
        p95_values = [r.total_time_p95 for r in asr_results]

        ax.plot(levels, avg_values, 'o-', color=self.color,
               label='ASR 总时间 (Avg)', linewidth=2, markersize=8)
        ax.plot(levels, p95_values, 's--', color=self.color,
               label='ASR 总时间 (P95)', linewidth=1.5, markersize=6, alpha=0.7)

        ax.set_xlabel('并发数', fontsize=12)
        ax.set_ylabel('时间 (ms)', fontsize=12)
        ax.set_title('总处理时间 vs 并发数', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
