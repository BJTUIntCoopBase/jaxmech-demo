# jaxmech-demo

[English](#english) | [中文](#中文)

---

<a id="english"></a>
## English

A minimal demo of the **jaxmech** computational mechanics library, showcasing:

- **3D Solid Linear Elastic Analysis** — Finite element analysis using JAX
- **Shakedown Analysis** — Lower-bound shakedown via SOCP (C formulation, CVXPY/Clarabel)
- **Web UI** — Browser-based interface for model management, analysis submission, and result inspection
- **MAT Field Visualization** — Browser-based 3D visualization for generated elastic and shakedown MAT results

### Supported Features

| Feature | Status |
|---------|--------|
| 3D Solid Elements (C3D8, C3D8R, C3D6, C3D4) | ✅ |
| Linear Elastic Material | ✅ |
| Shakedown C Formulation (CVXPY) | ✅ |
| ABAQUS INP Parsing | ✅ |
| Web Dashboard | ✅ |
| MAT Field Visualization | ✅ |
| Shell Elements | 🔒 Full version |
| Plastic Material Models | 🔒 Full version |
| Direct Cyclic Analysis | 🔒 Full version |
| Gurobi Solver Backend | 🔒 Full version |
| ODB Validation | 🔒 Full version |

### Quick Start

#### Prerequisites

- **WSL** (Windows Subsystem for Linux) with Python 3.9+
- **Windows Python 3.9+** for the Web UI

#### 1. Clone

```bash
git clone https://github.com/BJTUIntCoopBase/jaxmech-demo.git
cd jaxmech-demo
```

#### 2. Bootstrap the environment

Recommended: let the repository create `Config/env.cfg`, install the Windows web
dependencies, and install the WSL analysis dependencies in one step.

```powershell
powershell -ExecutionPolicy Bypass -File .\Config\setup_demo_env.ps1
```

If you already have a dedicated WSL JAX environment, point the demo to that
interpreter explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\Config\setup_demo_env.ps1 `
	-WslPython /home/yourname/miniconda3/envs/jax-fem-env/bin/python
```

Dependency files shipped with the repo:

- `requirements-web.txt` — Windows-side Web UI and MAT inspection dependencies
- `requirements-wsl.txt` — WSL-side analysis dependencies (`jax[cpu]`, `meshio`, `cvxpy`, `clarabel`, ...)

Notes:

- The Web dependency set now includes `pyvista`, `vtk`, `Pillow`, `meshio`,
  and `playwright` so the demo can render MAT fields from the browser.
- `requirements-wsl.txt` installs `jax[cpu]` for a reproducible CPU demo setup.
- If you already manage a CUDA-enabled JAX environment, keep that environment
	and only set `wsl_python` in `Config/env.cfg`.
- `Config/start_web.bat` will also auto-install `requirements-web.txt` on demand,
	but `Config/setup_demo_env.ps1` is the recommended first-run path.

#### 3. Run Linear Elastic Analysis

```powershell
# From Windows PowerShell
.\Config\run_in_wsl.ps1 -m jaxmech.tools.build.build_model_mat -- --config Examples/PlateWithHole/inc_analysis/inc_analysis.template.cfg
```

#### 4. Run Shakedown Analysis

```powershell
.\Config\run_in_wsl.ps1 -m jaxmech.modules.shakedown.run -- --config Examples/PlateWithHole/shakedown/shakedown_analysis.template.cfg
```

#### 5. Launch Web UI

```cmd
Config\start_web.bat
```

Then open http://127.0.0.1:8080 in your browser.

#### 6. Visualize MAT Results

After running linear elastic or shakedown analysis, open **Field Visualization**
from the sidebar. Select a generated `.mat` file under `Examples/.../inc_analysis`
or `Examples/.../shakedown`, choose field variables such as `S`, `E`, `U`,
`NFORC`, `RS`, `CEQ`, or `CInEQ`, and inspect the result interactively.

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
- Validation A/B visualization against ABAQUS ODB outputs
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
- **MAT 场变量可视化** — 在浏览器中查看 demo 生成的弹性与安定分析 MAT 结果

### 功能一览

| 功能 | 状态 |
|------|------|
| 三维实体单元（C3D8, C3D8R, C3D6, C3D4） | ✅ |
| 线弹性材料 | ✅ |
| 安定分析 C 公式（CVXPY） | ✅ |
| ABAQUS INP 解析 | ✅ |
| Web 工作台 | ✅ |
| MAT 场变量可视化 | ✅ |
| 壳单元 | 🔒 完整版 |
| 弹塑性材料模型 | 🔒 完整版 |
| 直接循环分析（DCA） | 🔒 完整版 |
| Gurobi 求解器后端 | 🔒 完整版 |
| ODB 对标验证 | 🔒 完整版 |

### 快速上手

#### 环境要求

- **WSL**（Windows Subsystem for Linux），Python 3.9+
- **Windows Python 3.9+**（用于 Web 界面）

#### 1. 克隆

```bash
git clone https://github.com/BJTUIntCoopBase/jaxmech-demo.git
cd jaxmech-demo
```

#### 2. 初始化环境

推荐直接运行仓库自带脚本，让它一次性完成 `Config/env.cfg` 创建、
Windows Web 依赖安装，以及 WSL 分析依赖安装：

```powershell
powershell -ExecutionPolicy Bypass -File .\Config\setup_demo_env.ps1
```

如果你已经有单独的 WSL JAX 环境，可以显式指定解释器：

```powershell
powershell -ExecutionPolicy Bypass -File .\Config\setup_demo_env.ps1 `
	-WslPython /home/yourname/miniconda3/envs/jax-fem-env/bin/python
```

仓库内置的依赖清单：

- `requirements-web.txt`：Windows 侧 Web UI 与 MAT 元数据读取依赖
- `requirements-wsl.txt`：WSL 侧分析依赖（`jax[cpu]`、`meshio`、`cvxpy`、`clarabel` 等）

说明：

- Web 依赖已包含 `pyvista`、`vtk`、`Pillow`、`meshio` 和 `playwright`，
  用于在浏览器中渲染 MAT 场变量。
- `requirements-wsl.txt` 默认安装 `jax[cpu]`，适合作为公开 demo 的 CPU 环境。
- 如果你已经维护了 CUDA 版 JAX 环境，保留该环境即可，只需在 `Config/env.cfg` 中设置 `wsl_python`。
- `Config/start_web.bat` 也会按需自动安装 `requirements-web.txt`，但首次使用仍推荐先运行 `Config/setup_demo_env.ps1`。

#### 3. 运行线弹性分析

```powershell
# 在 Windows PowerShell 中
.\Config\run_in_wsl.ps1 -m jaxmech.tools.build.build_model_mat -- --config Examples/PlateWithHole/inc_analysis/inc_analysis.template.cfg
```

#### 4. 运行安定分析

```powershell
.\Config\run_in_wsl.ps1 -m jaxmech.modules.shakedown.run -- --config Examples/PlateWithHole/shakedown/shakedown_analysis.template.cfg
```

#### 5. 启动 Web 界面

```cmd
Config\start_web.bat
```

在浏览器中打开 http://127.0.0.1:8080。

#### 6. 查看 MAT 可视化

运行线弹性分析或安定分析后，从侧边栏进入 **场变量可视化**。选择
`Examples/.../inc_analysis` 或 `Examples/.../shakedown` 下生成的 `.mat`
文件，再选择 `S`、`E`、`U`、`NFORC`、`RS`、`CEQ`、`CInEQ` 等场变量进行交互查看。

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
- 与 ABAQUS ODB 结果对比的 validation A/B 可视化
- 扩展安定公式（MN、Ilyushin、逐层应力）
- 拓扑优化
- 更多示例模型

获取完整版或合作意向，请联系：**gengchen@bjtu.edu.cn**

---

## License / 许可证

GPL-3.0 — 详见 [LICENSE](LICENSE)。

第三方声明：[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)
