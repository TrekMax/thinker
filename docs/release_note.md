# Release Notes

## v3.0.9 - 2026-04-30

来源提交：`b58e741`

- 新增 `Gelu`、`Swish` 算子支持，包含执行器 C 实现、算子注册、VenusA 内核以及 tpacker 图分析支持。
- 优化 VenusA 平台 `LinearInt`、`GRUInt`、`iqSigmoid`、`SoftmaxInt`、`AvgPool`、`Concat`、`Conv1dInt` 等算子实现。
- 调整 Arcs `GRUInt`、Venus `Conv1dInt`/`GRUInt` 相关实现，修复已知执行问题。
- 完善 tpacker 对 `Conv1dInt`、`Conv2dInt`、`ConvTranspose2dInt`、`GRUInt`、`LinearInt`、`LstmInt`、`Quant`、`Requant`、`Slice`、`SoftmaxInt`、`iqMul`、`iqSigmoid` 等算子的分析与替换逻辑。
- 修复 `Expand`、shape infer、`load_model`、`remove_slice`、tvalidator 等模块的已知问题。
- 清理生成的 `tools/thinker.egg-info` 目录，版本升级至 `3.0.9`。

## v3.0.8 - 2026-04-17

来源提交：`caaa4ff`

- 优化 `LinearInt` 内存分配，覆盖 VenusA 执行器实现和 tpacker 图分析逻辑。
- 修复 `Slice`、`Transpose`、`Unsqueeze` 算子相关问题。
- 增强 `split_conv2d` 对分支场景的处理能力。

## v3.0.7 - 2026-04-03

来源提交：`adcc27f`

- 修复 Venus 与 Arcs 平台 `iqAdd` 运行时内存不足问题。
- 支持全局 layout 优化策略，更新 `OperatorLayout` 与 `layout_convert` 相关逻辑。
- 修复 `LinearInt`、`LstmInt` 已知问题。
- 完善 tpacker 中 `iqAdd`、layout 转换和工具函数处理逻辑。

## v3.0.6 - 2026-03-26

来源提交：`4c5c3ad`、`78377f0`

- 更新开发环境配置文档，补充 CUDA、nvcc、NVIDIA 驱动等环境说明和截图。
- 更新 `thinker_packer`、`thinker_validator` 等文档。
- 修复/优化 VenusA 平台 `BatchNormInt`、`Conv2dInt`、`DeConv2dInt`、`GRUInt`、`iqAdd`、`iqDiv`、`iqMul`、`iqPad`、`LayerNormInt`、`LinearInt`、`LstmInt` 等算子。
- 修复 Arcs/Venus/VenusA 平台 `BatchNormInt`、`iqSigmoid`、`PReLU`、`ReLU`、`ReluX`、`Transpose` 等算子相关问题。
- 更新 VenusA `libluna` 运行库，并新增/调整 release 版本库文件。
- 更新 tpacker 激活函数、池化、卷积转置、LSTM 等图分析逻辑及设备配置。

## v3.0.5 - 2026-02-06

来源提交：`145fc9c`

- 优化 C API 函数返回值处理，调整 `thinker_api`、公共头文件与状态定义。
- 批量规范执行器算子的返回值与错误处理逻辑，覆盖 Arcs、Venus、VenusA 多平台实现。
- 修复/优化 `ArgMax`、`AvgPool2dInt`、`BatchNormInt`、`BmmInt`、`Concat`、`Conv1dInt`、`Conv2dInt`、`ConvTranspose2dInt`、`FFNInt`、`GRUInt`、`LstmInt`、`LinearInt`、`Resize`、`Transpose` 等算子。
- 更新 shape infer、设备配置、tpacker 图分析和 tvalidator 相关逻辑。

## v3.0.4 - 2026-01-14

来源提交：`a5ca271`、`8160682`、`509c1e5`、`c2ae8f3`  
标签：`v3.0.4`

- 修复/优化 `iqAdd`、`Transpose`、`LstmInt`、`LinearInt`、layout 转换、tvalidator 等模块。
- 更新开发环境与验证器说明文档。
- 调整 demo、测试和构建配置。
- 更新 pip 包生成说明，调整 `tools/install.sh`、`tools/setup.py`。
- 删除生成的 `tools/thinker.egg-info` 目录并更新 README。

## v3.0.3 - 2025-12-26

来源提交：`aa222f5`

- 修复/调整核心执行器、Arcs、Venus、VenusA 平台的 `Transpose` 相关实现。
- 优化 `LinearInt`、`Pool`、`Conv1dInt`、`Conv2dInt`、`Cast`、`iqSigmoid` 等算子或图分析逻辑。
- 更新 `op_split`、layout 相关处理和 tvalidator 工具逻辑。
- 删除旧的 `tools/image_preprocess.py`，更新 README 与 README_EN。

## v3.0.2 - 2025-12-19

来源提交：`1d76a05`

- 同步版本至 `3.0.2`。
- 完成 3.x 目录结构同步，将工程内容整理到根级 `docs`、`executor`、`tools`、`demo`、`scripts` 等目录。
- 新增/刷新文档：算子说明、量化算子支持列表、API、自动测试、构建、编译、Docker、环境配置、打包、性能、profile、运行、技术亮点、validator 等。
- 新增/同步执行器 C API、算子注册、shape infer、公共头文件，以及 Arcs、Venus、VenusA 多平台算子实现和运行库。
- 新增/同步 tpacker、tprofile、tvalidator 工具链及安装、打包配置。
- 新增动态库调用 demo、test_thinker demo 和 x86 Linux 构建/测试脚本。

