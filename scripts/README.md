# 脚本

- `build.sh`（仓库根目录）：构建 `quantatrisk/asrserve:gpu`；`TARGET=cpu ./build.sh` 构建 `quantatrisk/asrserve:cpu`。Compose 不含构建参数。
- `build-rust.sh`：构建原生 CPU 动态库。
- `prepare-models.sh`：用服务镜像预下载模型到 `models/`，供离线部署拷贝；不需要 GPU；`IMAGE=quantatrisk/asrserve:cpu` 改用 CPU 镜像。
- `sync_gpu_env.sh`：安装锁定的 Linux CUDA 依赖。
- `validate_nemotron.py`、`analyze_audio_rms.py`：说话人分离验收与音频能量检查。

启动与配置见[部署说明](../docs/deployment.md)，回放与并发验收见 [benchmark](benchmark/README.md)。
