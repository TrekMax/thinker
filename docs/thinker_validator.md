# ThinkerValidator 命令行使用文档

## 📌 简介

**ThinkerValidator** 是一个用于验证训练框架与推理框架输出一致性的工具模块。  
该工具基于训练端的 **Linger** 工具链与推理端的 **Thinker** 推理引擎，在两侧分别执行推理任务并导出中间张量（Tensor）的 **dump 文件**。

通过对比两侧生成的 dump 数据，**ThinkerValidator** 能够自动检测并报告计算结果上的差异，从而帮助用户快速定位模型在推理阶段可能存在的不一致算子或节点。

本章节介绍 **ThinkerValidator 的命令行（CLI）使用方式**。

---

## ⚙️ 安装与依赖

在使用 **ThinkerValidator CLI** 之前，请确保系统中已正确安装以下组件：

- **Linger**：量化训练工具  
- **Thinker**：推理引擎及其运行时动态库（`libthinker.so`）

### ⚠️ 注意

由于 **Linger** 与 **Thinker** 在部分依赖包（尤其是 ONNX 相关依赖）上可能存在版本冲突，  
在将 **Linger** 与 **Thinker** 分别安装完成后，建议执行以下命令更新 ONNX 以避免环境冲突：

```bash
pip install --upgrade onnx
```
## 🧩 参数说明
### 基本命令行格式
```bash
tvalidator [options]
```
### 参数列表
| 参数                   | 类型          | 是否必需 | 说明                                                   |
| -------------------- | ----------- | ---- | ---------------------------------------------------- |
| `-g`, `--onnx_path`  | `str`       | 是    | ONNX 模型文件路径，必须提供                                     |
| `-r`, `--res_path`   | `str`       | 否    | 模型资源文件路径（`.bin`）。当需要手动打包模型时必须提供                      |
| `-l`, `--lib_path`   | `str`       | 否    | Thinker 推理引擎动态库路径（如 `libthinker.so`）。当不在项目根目录执行时必须提供 |
| `-i`, `--input_path` | `str ...`   | 否    | 一个或多个输入文件路径。当需要手动指定输入时必须提供                           |
| `--cfg`              | `key=value` | 否    | 动态 shape 配置参数                                 |
## 🧩 动态 Shape 参数说明（--cfg）
value：以逗号分隔的整数列表，通常表示 (min:max:step)
- **key**：动态 shape 名称。  
- **value**：以逗号分隔的整数列表，通常表示 (min:max:step)。 
### 示例
```bash
--cfg seq_len=32:384:32
```
## 🚀 使用示例
### 1️⃣ 模型不需要手动打包，仅提供ONNX
```bash
tvalidator \
  -g model/track_id/arcs_trackid20250919.onnx \
  -l bin/libthinker.so
```
### 2️⃣ 模型需要手动打包，同时提供ONNX和打包后的资源
```bash
tvalidator \
  -g model/track_id/arcs_trackid20250919.onnx \
  -r model/track_id/arcs_trackid20250919.bin \
  -l bin/libthinker.so
```
### 3️⃣ 使用自定义的输入验证一致性
```bash
tvalidator \
  -g model/track_id/arcs_trackid20250919.onnx \
  -r model/track_id/arcs_trackid20250919.bin \
  -l bin/libthinker.so \
  -i input_0.npy input_1.npy
```
### 4️⃣ 验证动态shape图
```bash
tvalidator \
  -g model/track_id/arcs_trackid20250919.onnx \
  -r model/track_id/arcs_trackid20250919.bin \
  -l bin/libthinker.so \
  --cfg seq_len=32:384:32,yinsu_len=1:80:1
```
## 🧪 输出结果说明
在执行一致性验证后，**ThinkerValidator** 会根据比对结果输出两种状态信息：

---

### ✅ 一致性验证通过
当所有中间 Tensor 的内容完全一致时，系统将输出如下提示信息：
```bash
✔ Consistency verification passed!
```
表示训练框架与推理框架的计算结果完全对齐，无需进一步排查。
### ❌ 一致性验证未通过

若检测到任意 Tensor 文件存在差异，系统将输出如下错误信息，并自动打开vscode的compare功能（如果在vscode环境下），协助定位第一个出错的位置：
```
-> [!] First mismatch tensor: 142
  -> Shape: (1, 8, 64, 64)

    -> Showing first 16 mismatched entries:
    -----------------------------------------------------------------
    |     Index      |   Linger (training)   |  Thinker (inference) |
    -----------------------------------------------------------------
    | (0, 0, 0, 0)   |         16            |          0           |
    | (0, 0, 0, 1)   |          2            |         26           |
    | (0, 0, 0, 2)   |          0            |         12           |
    | (0, 0, 0, 3)   |         12            |          0           |
    | (0, 0, 0, 4)   |         34            |          0           |
    | (0, 0, 0, 5)   |         21            |         17           |
    | (0, 0, 0, 6)   |         11            |          8           |
    | (0, 0, 0, 7)   |          0            |         40           |
    | (0, 0, 0, 8)   |         13            |          7           |
    | (0, 0, 0, 11)  |         12            |         24           |
    | (0, 0, 0, 12)  |         17            |          7           |
    | (0, 0, 0, 13)  |         17            |          0           |
    | (0, 0, 0, 14)  |          0            |         12           |
    | (0, 0, 0, 15)  |          3            |          0           |
    | (0, 0, 0, 16)  |          0            |         38           |
    | (0, 0, 0, 17)  |         24            |          0           |
    -----------------------------------------------------------------
  -> Launching VSCode compare for the following files: 
      -> Linger : data/onnxrunner_int/142##_int_dump.txt
      -> Thinker: data/142##_1_8_64_64.txt
```