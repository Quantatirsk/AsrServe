# -*- coding: utf-8 -*-
"""
AsrServe 并发性能测试主入口

使用方法:
    python -m scripts.benchmark.run --audio-file /path/to/audio.wav

    # 自定义并发级别
    python -m scripts.benchmark.run --audio-file /path/to/audio.wav --concurrency 5 10 20
"""

import argparse
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List

from .clients.asr_client import ASRWebSocketClient
from .config import TestConfig
from .metrics.models import AggregatedMetrics, ASRMetrics
from .metrics.statistics import calculate_statistics
from .reporters.markdown_reporter import MarkdownReporter
from .utils.audio_utils import load_audio_file

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ConcurrentBenchmark:
    """并发性能测试运行器"""

    def __init__(self, config: TestConfig):
        self.config = config
        # 创建保存目录
        self._setup_output_dirs()

    def _setup_output_dirs(self):
        """创建输出目录结构"""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        # ASR 结果目录
        self.asr_output_dir = self.config.output_dir / "asr"
        self.asr_output_dir.mkdir(exist_ok=True)

    async def run_asr_benchmark(self) -> List[AggregatedMetrics]:
        """
        运行 ASR 并发测试

        Returns:
            各并发级别的聚合指标列表
        """
        logger.info("开始 ASR 并发性能测试...")

        # 检查音频文件
        if self.config.asr_audio_file is None:
            raise ValueError("ASR 测试需要提供音频文件")

        # 加载音频文件
        audio_data, audio_duration = load_audio_file(
            self.config.asr_audio_file,
            self.config.asr_sample_rate,
        )
        audio_duration_ms = audio_duration * 1000

        logger.info(f"音频文件已加载: {self.config.asr_audio_file}")
        logger.info(f"  - 时长: {audio_duration:.2f} 秒")
        logger.info(f"  - 大小: {len(audio_data) / 1024:.1f} KB")

        results = []

        for level in self.config.concurrency_levels:
            logger.info(f"\n测试并发级别: {level}")

            # 预热
            logger.info(f"  预热中 ({self.config.warmup_requests} 次请求)...")
            await self._run_asr_concurrent(
                audio_data,
                audio_duration_ms,
                self.config.warmup_requests,
                level,
                save_results=False,
            )

            # 正式测试
            logger.info("  正式测试中...")
            start_time = time.perf_counter()
            metrics_list = await self._run_asr_concurrent(
                audio_data,
                audio_duration_ms,
                level,
                level,
                save_results=True,  # 正式测试时保存结果
            )
            total_time = time.perf_counter() - start_time

            # 统计
            aggregated = calculate_statistics(metrics_list, level, total_time)
            results.append(aggregated)

            # 打印结果
            logger.info(
                f"  完成: 成功 {aggregated.successful_requests}/{aggregated.total_requests}"
            )
            logger.info(f"  首次响应延迟: {aggregated.first_latency_avg:.1f} ms (avg)")
            logger.info(f"  RTF: {aggregated.rtf_avg:.3f} (avg)")

        return results

    async def _run_asr_concurrent(
        self,
        audio_data: bytes,
        audio_duration_ms: float,
        num_requests: int,
        concurrency_level: int,
        save_results: bool = False,
    ) -> List[ASRMetrics]:
        """运行并发 ASR 请求"""
        tasks = []

        for _ in range(num_requests):
            client = ASRWebSocketClient(
                ws_url=self.config.asr_ws_url,
                audio_data=audio_data,
                audio_duration_ms=audio_duration_ms,
                sample_rate=self.config.asr_sample_rate,
                chunk_size=self.config.asr_chunk_size,
                timeout=self.config.timeout_seconds,
                save_result_dir=self.asr_output_dir if save_results else None,
            )
            tasks.append(client.run_test())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        metrics_list = []
        for result in results:
            if isinstance(result, ASRMetrics):
                result.concurrency_level = concurrency_level
                metrics_list.append(result)
            else:
                # 异常情况
                metrics = ASRMetrics(
                    request_id="error",
                    concurrency_level=concurrency_level,
                    start_time=0,
                    error_message=str(result),
                )
                metrics_list.append(metrics)

        return metrics_list

    def generate_report(self, asr_results: List[AggregatedMetrics]) -> Path:
        """
        生成测试报告

        Args:
            asr_results: ASR 测试结果

        Returns:
            报告文件路径
        """
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 配置信息
        config_info = {
            "host": self.config.host,
            "port": self.config.port,
            "concurrency_levels": self.config.concurrency_levels,
        }

        # 生成 Markdown 报告
        report_path = output_dir / f"{self.config.report_name}_{timestamp}.md"
        reporter = MarkdownReporter()
        reporter.generate(asr_results, report_path, config_info)
        logger.info(f"Markdown 报告已生成: {report_path}")

        # 生成图表 (matplotlib 为可选依赖，按需导入)
        from .reporters.chart_generator import ChartGenerator

        chart_generator = ChartGenerator()
        chart_files = chart_generator.generate_all_charts(asr_results, output_dir, timestamp)
        for chart_file in chart_files:
            logger.info(f"图表已生成: {chart_file}")

        return report_path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="AsrServe 并发性能测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m scripts.benchmark.run --audio-file test.wav

  # 自定义并发级别
  python -m scripts.benchmark.run --audio-file test.wav --concurrency 5 10 20 50
        """,
    )

    parser.add_argument(
        "--host",
        default="localhost",
        help="服务器主机名 (默认: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口 (默认: 8000)",
    )
    parser.add_argument(
        "--audio-file",
        type=Path,
        help="测试用音频文件路径 (必需)",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="并发级别列表 (默认: 1 2 4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./benchmark_results"),
        help="报告输出目录 (默认: ./benchmark_results)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="请求超时时间 (秒, 默认: 120)",
    )

    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_args()

    # 创建配置
    config = TestConfig(
        host=args.host,
        port=args.port,
        concurrency_levels=args.concurrency,
        asr_audio_file=args.audio_file,
        output_dir=args.output,
        timeout_seconds=args.timeout,
    )

    # 验证配置
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"配置错误: {e}")
        return

    # 运行测试
    benchmark = ConcurrentBenchmark(config)
    asr_results = await benchmark.run_asr_benchmark()

    # 生成报告
    report_path = benchmark.generate_report(asr_results)
    logger.info(f"\n测试完成! 报告已保存到: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
