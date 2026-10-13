# 最新 SH 与代码依赖裁剪清单

九个模型各保留一份最新记录所用的 `.sh`，共保留 55 个原始文件；从上一版移出 101 个文件。原始压缩包保持不变。

最新版依据最后一份带日期的任务提交脚本所调用的入口确定。多数 ZIP 时间戳无效，不能据此确认完整的版本先后关系。

| 模型 | 最新提交记录 | 唯一保留的 SH | 直接调用脚本 | 文件数 |
|---|---|---|---|---|
| OCSVM | 2026-07-01 14:46:54 | `run.sh` | `ocsvm_baseline.py` | 2 |
| DAOD | 2026-07-01 15:13:06 | `script_20260701151306.sh` | `run1.m` | 7 |
| OSBP | 2025-12-30 17:03:36 | `train1.sh` | `train1.py` | 8 |
| SDEOTR | 2026-06-16 11:03:32 | `run.sh` | `train.py`, `train.py` | 5 |
| ANNA | 2026-06-16 11:00:23 | `run_re.sh` | `train.py` | 6 |
| BCL | 2026-06-17 11:54:10 | `run1.sh` | `train.py` | 3 |
| STA | 2026-01-13 16:20:38 | `run3.sh` | `wavstep_1.py`, `wavstep_2.py` | 7 |
| STIENet | 2026-01-08 16:20:32 | `run3.sh` | `train1.py` | 6 |
| UADAL | 2025-12-30 16:00:53 | `run2.sh` | `train1.py` | 11 |

从 SH 调用脚本继续追踪 Python 本地导入、包初始化文件、MATLAB 本地函数以及源码中存在的本地配置资源。保留 ANNA、STA 原许可证。UADAL 的 SH 固定 MODEL=UADAL，因而排除仅由 cUADAL 分支使用的文件；其他不能确定不可达的导入仍保留。

STIENet 的 eval.py 导入 DataLoader.py，因此即使最新训练入口使用 DataLoader_w.py，两个加载器均需保留。UADAL 的 get_loader.py 会在模块导入时加载图像数据辅助模块，因此也保留这些间接依赖。

核对结果：55 个文件的原始来源已核对。随后对 53 个代码文件统一格式和英文注释，检查确认可执行结构保持一致；两份许可证逐字节不变。38 个 Python 文件及 SH 内嵌 Python 语法通过；9 份 SH 均通过 bash -n；最新入口与已追踪的本地代码依赖均存在。未执行完整训练或 MATLAB 实验。

运行前仍需设置原服务器绝对路径和依赖环境。BCL 最新入口仅使用 ResNet，ANNA 包含 ResNet 配置；ResNet 数据仍暂停。最新脚本的特征与任务覆盖各不相同，不能把这份裁剪包视为已经复现论文全部 540 项实验。

- [格式与英文注释说明](CODE_STYLE.md)
- [保留文件及来源](metadata/source-manifest.csv)
- [依赖关系和保留原因](metadata/dependency-graph.json)
- [移除的 101 个文件及原因](metadata/removed-model-files.json)
- [运行入口说明](RUNNERS.md)
