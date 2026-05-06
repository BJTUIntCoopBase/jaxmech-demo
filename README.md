# jaxmech-demo

**jaxmech-demo** 是 jaxmech 的轻量公开演示版，用于展示从实体单元弹性分析、ODB validation、CVXPY shakedown 到 Web visualization 的完整最小工作流。

> Demo 保留完整版 Web GUI 的布局、sidebar、dashboard card、任务管理和 visualization 体验；未开放的模块以前端加锁形式呈现，对应后端代码和 runner 不进入 demo 仓库。

## ✨ 新版本亮点

| 图标 | 新增/改进 | 说明 |
| --- | --- | --- |
| 🧭 | 完整版菜单布局 | Demo 与 full version 保持同一套 GUI 风格，未开放入口显示锁定状态。 |
| 🧱 | 实体弹性 `inc_analysis` | 支持 3D solid linear elastic 分析；正式计算一次只生成一个结果 MAT。 |
| 🧪 | Solid elastic ODB validation | Validation 模块包含实体单元弹性对标流程，可从已有 JAX MAT 与 ABAQUS ODB/MAT 做比较。 |
| 🛡️ | CVXPY shakedown | 支持实体单元 lower-bound shakedown 的 C formulation + CVXPY/Clarabel backend。 |
| 🎨 | MAT 场变量可视化 | 保留 visualization 模块完整功能，支持 elastic、validation 和 shakedown MAT。 |
| 📋 | 任务管理 | Web task runs、日志、artifact 和历史任务恢复保留完整体验。 |
| 📁 | 模型浏览增强 | 支持浏览 `Examples/` 和 `StoredModels/`，并识别 timestamp run folder 中的结果。 |
| ⚙️ | 环境设置增强 | Settings 中提供一键自动检测，以及一键安装所有支持库；已检测到部分库时会补齐缺失项。 |

## 🚀 快速开始

### 环境要求

- Windows + WSL
- WSL Python 3.9+，用于 JAX 分析流程
- Windows Python 3.9+，用于 Web UI、任务管理和 visualization

### 安装 demo 环境

```powershell
powershell -ExecutionPolicy Bypass -File .\Config\setup_demo_env.ps1
```

如果你已经有可用的 WSL JAX 环境，可以显式指定：

```powershell
powershell -ExecutionPolicy Bypass -File .\Config\setup_demo_env.ps1 `
  -WslPython /home/yourname/miniconda3/envs/jax-fem-env/bin/python
```

### 启动 Web UI

```cmd
Config\start_web.bat
```

然后在浏览器打开：

```text
http://127.0.0.1:8080
```

## 🧩 推荐 Web 工作流

1. 打开 **设置**，执行一键自动检测；如有缺失，使用一键安装所有支持库补齐 Windows Web 依赖。
2. 进入 **模型浏览**，从 `Examples/` 或 `StoredModels/` 选择模型。
3. 进入 **增量分析**，选择 solid INP，完成 parse 后运行 linear elastic analysis。
4. 可选：进入 **ODB 验证**，选择已有 JAX MAT 与匹配的 ABAQUS ODB/MAT 进行 elastic validation。
5. 进入 **安定分析**，选择弹性分析生成的 MAT，运行 C formulation + CVXPY shakedown。
6. 在 **场变量可视化** 中查看 elastic、validation 或 shakedown 的 MAT 结果。
7. 在 **任务管理** 中查看日志、运行状态和生成的 MAT artifact。

## 🖥️ CLI 入口

Web UI 是推荐入口；如需命令行运行，可使用同一套 WSL wrapper。`--config` 可以指向 Web 生成的 run folder 内配置文件，或模块模板复制后得到的配置文件。

### Solid elastic analysis

```powershell
.\Config\run_in_wsl.ps1 -m jaxmech.tools.build_model_mat -- --config <path-to-inc-analysis-cfg>
```

### Solid elastic ODB validation

```powershell
.\Config\run_in_wsl.ps1 -m jaxmech.modules.validation.elastic.run -- --config <path-to-validation-cfg>
```

### CVXPY shakedown

```powershell
.\Config\run_in_wsl.ps1 -m jaxmech.modules.shakedown.run -- --config <path-to-shakedown-cfg>
```

## 📦 Included Examples

| Example | 内容 |
| --- | --- |
| `PlateWithHole` | 3D solid plate with hole，包含两个 elastic load cases，可用于 inc_analysis、validation 和 shakedown。 |
| `PlateWithHoleMultiEle` | 混合实体单元示例，包含 C3D8、C3D6 和 C3D4。 |

## 🔒 Demo Scope

### ✅ 可用功能

| 模块 | Demo 状态 |
| --- | --- |
| Solid linear elastic `inc_analysis` | 可运行 |
| Solid elastic ODB `validation` | 可运行 |
| Solid C formulation + CVXPY shakedown | 可运行 |
| Web UI | 可用，保持完整版布局 |
| Task management | 可用，保留完整功能 |
| MAT visualization | 可用，保留完整功能 |
| Model browser | 可用，浏览有限示例与本地模型 |
| Settings | 可用，支持一键检测和一键安装支持库 |

### 🔐 仅保留锁定入口

| 模块/能力 | Demo 状态 |
| --- | --- |
| Shell analysis / shell validation | 锁定入口 |
| Elastic-plastic incremental analysis | 锁定入口 |
| Direct Methods / DCA / RSDM | 锁定入口 |
| Nonlinear / DCA validation | 锁定入口 |
| Shell shakedown | 锁定入口 |
| Gurobi backend | 锁定入口 |
| Topology optimization | 锁定入口 |

## 🗂️ 目录速览

```text
Config/        环境初始化、WSL wrapper 和 Web 启动脚本
Examples/      随仓库提供的演示模型
StoredModels/  本地模型工作区，默认不随仓库发布
Cache/         Web task runs、日志和本地缓存
jaxmech/       demo 版可运行源码
```

## 🤝 Full Version

Demo 版用于公开演示和教学验证。若需要完整版本、更多模块或合作事宜，请联系：

**gengchen@bjtu.edu.cn**

## 📄 License

GPL-3.0. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).
