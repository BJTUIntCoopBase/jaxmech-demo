# jaxmech-demo

`jaxmech-demo` is a lightweight public demo of the jaxmech workflow.

The runnable code is intentionally limited to:

- 3D solid linear elastic `inc_analysis`
- Solid lower-bound shakedown with the C formulation and CVXPY/Clarabel backend
- Browser-based Web UI with the same layout as the full version
- Full task management and MAT field visualization for elastic and shakedown results

The Web UI keeps the full-version layout and menu shape. Unsupported analysis
entries are locked in the frontend and their analysis code/backend runners are
not included in this demo repository. The settings page also provides one-click
environment detection and one-click installation of supported Windows Web
libraries.

## Quick Start

### Requirements

- Windows with WSL
- WSL Python 3.9+ for JAX analysis
- Windows Python 3.9+ for the Web UI

### Install

```powershell
powershell -ExecutionPolicy Bypass -File .\Config\setup_demo_env.ps1
```

If you already have a WSL JAX environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\Config\setup_demo_env.ps1 `
  -WslPython /home/yourname/miniconda3/envs/jax-fem-env/bin/python
```

### Run Solid Elastic Analysis

```powershell
.\Config\run_in_wsl.ps1 -m jaxmech.tools.build.build_model_mat -- --config Examples/PlateWithHole/inc_analysis/inc_analysis.template.cfg
```

### Run Shakedown

Run elastic analysis first, then:

```powershell
.\Config\run_in_wsl.ps1 -m jaxmech.modules.shakedown.run -- --config Examples/PlateWithHole/shakedown/shakedown_analysis.template.cfg
```

### Launch Web UI

```cmd
Config\start_web.bat
```

Open http://127.0.0.1:8080.

## Included Examples

| Example | Description |
| --- | --- |
| `PlateWithHole` | Solid 3D plate with two elastic load cases and shakedown template |
| `PlateWithHoleMultiEle` | Solid mixed-element plate using C3D8, C3D6, and C3D4 |

## Demo Scope

Available in this repository:

| Area | Status |
| --- | --- |
| Solid elastic `inc_analysis` | Available |
| Solid C-formulation shakedown with CVXPY | Available |
| Web UI | Available, full-version layout with locked unsupported entries |
| Task management | Available |
| MAT visualization | Available |
| Shell analysis | Locked menu only |
| Elastic-plastic incremental analysis | Locked menu only |
| Direct methods / DCA | Locked menu only |
| ODB validation | Locked menu only |
| Gurobi backend | Locked menu only |
| Topology optimization | Locked menu only |

For the full version or collaboration inquiries, contact **gengchen@bjtu.edu.cn**.

## License

GPL-3.0. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).

---

# jaxmech-demo 中文说明

`jaxmech-demo` 是 jaxmech 的轻量公开演示版本。

本仓库中实际可运行的代码仅限于：

- 三维实体单元线弹性 `inc_analysis`
- 实体单元下限安定分析，C formulation，CVXPY/Clarabel backend
- 与完整版布局一致的 Web 端工作台
- 完整任务管理，以及弹性与 shakedown MAT 结果的场变量可视化

Web 端保留完整版布局和菜单结构。不支持的分析入口在前端锁定，对应分析代码和后端 runner 不包含在 demo 仓库中。设置页保留一键自动检测，并增加一键安装所有支持的 Windows Web 库。

## 快速开始

### 环境要求

- Windows + WSL
- WSL Python 3.9+，用于 JAX 分析
- Windows Python 3.9+，用于 Web UI

### 初始化环境

```powershell
powershell -ExecutionPolicy Bypass -File .\Config\setup_demo_env.ps1
```

如已有 WSL JAX 环境：

```powershell
powershell -ExecutionPolicy Bypass -File .\Config\setup_demo_env.ps1 `
  -WslPython /home/yourname/miniconda3/envs/jax-fem-env/bin/python
```

### 运行实体弹性分析

```powershell
.\Config\run_in_wsl.ps1 -m jaxmech.tools.build.build_model_mat -- --config Examples/PlateWithHole/inc_analysis/inc_analysis.template.cfg
```

### 运行 shakedown

先运行弹性分析生成 MAT，再执行：

```powershell
.\Config\run_in_wsl.ps1 -m jaxmech.modules.shakedown.run -- --config Examples/PlateWithHole/shakedown/shakedown_analysis.template.cfg
```

### 启动 Web UI

```cmd
Config\start_web.bat
```

浏览器打开 http://127.0.0.1:8080。

## 示例模型

| 示例 | 说明 |
| --- | --- |
| `PlateWithHole` | 三维实体含孔板，两个弹性载荷工况，附 shakedown 模板 |
| `PlateWithHoleMultiEle` | C3D8、C3D6、C3D4 混合实体单元模型 |

## 功能边界

本 demo 可用功能：

| 功能 | 状态 |
| --- | --- |
| 实体弹性 `inc_analysis` | 可用 |
| 实体 C formulation + CVXPY shakedown | 可用 |
| Web UI | 可用，完整版布局，不支持入口加锁 |
| 任务管理 | 可用 |
| MAT 场变量可视化 | 可用 |
| Shell 分析 | 仅锁定菜单 |
| 弹塑性增量分析 | 仅锁定菜单 |
| Direct methods / DCA | 仅锁定菜单 |
| ODB 验证 | 仅锁定菜单 |
| Gurobi backend | 仅锁定菜单 |
| 拓扑优化 | 仅锁定菜单 |

获取完整版本或合作咨询，请联系 **gengchen@bjtu.edu.cn**。
