# jaxmech-demo

[English](#english) | [中文](#中文)

---

<a id="english"></a>
## English

A minimal demo of the **jaxmech** computational mechanics library, showcasing:

- **3D Solid Linear Elastic Analysis** — Finite element analysis using JAX
- **Shakedown Analysis** — Lower-bound shakedown via SOCP (C formulation, CVXPY/Clarabel)
- **Web UI** — Browser-based interface for model management, analysis submission, and result inspection

### Supported Features

| Feature | Status |
|---------|--------|
| 3D Solid Elements (C3D8, C3D8R, C3D6, C3D4) | ✅ |
| Linear Elastic Material | ✅ |
| Shakedown C Formulation (CVXPY) | ✅ |
| ABAQUS INP Parsing | ✅ |
| Web Dashboard | ✅ |
| Shell Elements | 🔒 Full version |
| Plastic Material Models | 🔒 Full version |
| Direct Cyclic Analysis | 🔒 Full version |
| Gurobi Solver Backend | 🔒 Full version |
| ODB Validation | 🔒 Full version |

### Quick Start

#### Prerequisites

- **WSL** (Windows Subsystem for Linux) with Python 3.9+
- **WSL Python packages**: `jax`, `jaxlib`, `numpy`, `scipy`, `meshio`, `cvxpy`, `clarabel`
- **Windows Python** for the Web UI: `pip install fastapi uvicorn pydantic websockets`

#### 1. Clone

```bash
git clone https://github.com/BJTUIntCoopBase/jaxmech-demo.git
cd jaxmech-demo
```

#### 2. Run Linear Elastic Analysis

```powershell
# From Windows PowerShell
.\Config\run_in_wsl.ps1 -m jaxmech.tools.build.build_model_mat -- --config Examples/PlateWithHole/inc_analysis/inc_analysis.template.cfg
```

#### 3. Run Shakedown Analysis

```powershell
.\Config\run_in_wsl.ps1 -m jaxmech.modules.shakedown.run -- --config Examples/PlateWithHole/shakedown/shakedown_analysis.template.cfg
```

#### 4. Launch Web UI

```cmd
Config\start_web.bat
```

Then open http://127.0.0.1:8080 in your browser.

### Included Examples

| Example | Description |
|---------|-------------|
| `PlateWithHole` | 3D plate with a circular hole, two load cases, pre-built elastic MAT |
| `PlateWithHoleMultiEle` | Mixed-element 3D plate (C3D8 + C3D6 + C3D4) |

### Full Version

This is a **minimal demo subset** of jaxmech. The full version includes:

- Shell element analysis (S4, S4R, STRI3, STRI65)
- Elastic-plastic incremental analysis (J2 perfect plasticity)
- Direct Cyclic Analysis (DCA)
- Gurobi SOCP solver backend
- ABAQUS ODB validation pipeline
- Extended shakedown formulations (MN, Ilyushin, layer-wise)
- Topology optimization
- Additional example models

For the full version or collaboration inquiries, contact: **gengchen@bjtu.edu.cn**

---

<a id="中文"></a>
## 中文

**jaxmech** 计算力学库的最小演示版本，包含以下核心功能：

- **三维实体线弹性分析** — 基于 JAX 的有限元分析
- **安定分析** — 基于 SOCP 的下限安定性分析（C 公式，CVXPY/Clarabel）
- **Web 界面** — 模型管理、分析提交、结果查看的浏览器界面

### 功能一览

| 功能 | 状态 |
|------|------|
| 三维实体单元（C3D8, C3D8R, C3D6, C3D4） | ✅ |
| 线弹性材料 | ✅ |
| 安定分析 C 公式（CVXPY） | ✅ |
| ABAQUS INP 解析 | ✅ |
| Web 工作台 | ✅ |
| 壳单元 | 🔒 完整版 |
| 弹塑性材料模型 | 🔒 完整版 |
| 直接循环分析（DCA） | 🔒 完整版 |
| Gurobi 求解器后端 | 🔒 完整版 |
| ODB 对标验证 | 🔒 完整版 |

### 快速上手

#### 环境要求

- **WSL**（Windows Subsystem for Linux），Python 3.9+
- **WSL Python 依赖**：`jax`、`jaxlib`、`numpy`、`scipy`、`meshio`、`cvxpy`、`clarabel`
- **Windows Python**（Web 界面）：`pip install fastapi uvicorn pydantic websockets`

#### 1. 克隆

```bash
git clone https://github.com/BJTUIntCoopBase/jaxmech-demo.git
cd jaxmech-demo
```

#### 2. 运行线弹性分析

```powershell
# 在 Windows PowerShell 中
.\Config\run_in_wsl.ps1 -m jaxmech.tools.build.build_model_mat -- --config Examples/PlateWithHole/inc_analysis/inc_analysis.template.cfg
```

#### 3. 运行安定分析

```powershell
.\Config\run_in_wsl.ps1 -m jaxmech.modules.shakedown.run -- --config Examples/PlateWithHole/shakedown/shakedown_analysis.template.cfg
```

#### 4. 启动 Web 界面

```cmd
Config\start_web.bat
```

在浏览器中打开 http://127.0.0.1:8080。

### 示例模型

| 示例 | 说明 |
|------|------|
| `PlateWithHole` | 含圆孔三维板，两个载荷工况，附带预构建弹性 MAT |
| `PlateWithHoleMultiEle` | 混合单元三维含孔板（C3D8 + C3D6 + C3D4） |

### 完整版

本仓库仅包含 jaxmech 的**最小演示子集**。完整版还包括：

- 壳单元分析（S4, S4R, STRI3, STRI65）
- 弹塑性增量分析（J2 ideal plasticity）
- 直接循环分析（DCA）
- Gurobi SOCP 求解器后端
- ABAQUS ODB 对标验证流程
- 扩展安定公式（MN、Ilyushin、逐层应力）
- 拓扑优化
- 更多示例模型

获取完整版或合作意向，请联系：**gengchen@bjtu.edu.cn**

---

## License / 许可证

GPL-3.0 — 详见 [LICENSE](LICENSE)。

第三方声明：[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)
