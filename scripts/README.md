# 脚本

- `build.sh`（仓库根目录）：构建 GPU 镜像；`TARGET=cpu ./build.sh` 构建 CPU，统一标签 `quantatrisk/qwen3-asr:latest`。
- `build-rust.sh`：构建原生 CPU 动态库。
- `prepare-models.sh`：下载模型到 `models/`；`--export-dir /path` 导出离线模型。Linux CPU 设置 `DEVICE=cpu`；910B 设置 `DEVICE=npu:0 ALIGNMENT_MODE=uniform`，下载使用 CPU 环境且跳过 Aligner。
- `sync_gpu_env.sh`：安装锁定的 Linux CUDA 依赖。
- `validate_nemotron.py`、`analyze_audio_rms.py`：说话人分离验收与音频能量检查。

启动与配置见[部署说明](../docs/deployment.md)，回放与并发验收见 [benchmark](benchmark/README.md)。
