/**
 * jaxmech Web Frontend — Main Application
 * Single-page router, data fetching, and page rendering.
 */
const App = {
  currentPage: 'dashboard',
  models: [],
  tasks: [],
  docs: [],
  envStatus: null,
  selectedModel: null,
  _pendingModelTargetPage: null,
  activeWs: null,
  modelFilter: new Set(['examples', 'stored']),  // 双源筛选状态
  _elastic: {
    model: null,
    parsedInpMetaByName: {},
    parsedSelectionKey: '',
    lastParsedAt: '',
    resultMatMetaByName: {},
    cfgValues: {},
  },
  _directMethods: {
    model: null,
    parsedInpMetaByName: {},
    parsedSelectionKey: '',
    lastParsedAt: '',
    cfgValues: {},
  },
  _validation: { model: null, currentMatMeta: null },

  // Module availability (fetched from /api/status/modules)
  _moduleAvail: null,

  // Map page id → module probe key(s): page is locked when ALL keys are false.
  _pageModuleMap: {
    'shakedown':      ['shakedown'],
    'direct-methods': ['direct_methods', 'direct_methods_steady_state', 'direct_methods_shakedown'],
    'validation':     ['validation'],
  },

  // ──────────── Initialization ────────────
  async init() {
    this.bindNav();
    this.refreshSidebarStatus();
    // Fetch module availability early, then apply locks.
    this.api('/api/status/modules').then(avail => {
      this._moduleAvail = avail;
      this._applyModuleLocks();
    }).catch(() => {});
    this.navigate('dashboard');
  },

  bindNav() {
    document.querySelectorAll('.nav-item[data-page]').forEach(item => {
      item.addEventListener('click', () => {
        if (item.classList.contains('locked')) return;
        this.navigate(item.dataset.page);
      });
    });
  },

  /** Check whether a page should be locked based on module availability. */
  _isPageLocked(page) {
    if (!this._moduleAvail) return false;  // Not yet fetched → allow
    const keys = this._pageModuleMap[page];
    if (!keys) return false;  // Pages without module mapping are always unlocked
    return keys.every(k => this._moduleAvail[k] === false);
  },

  /** Apply locked class to sidebar nav items and existing dashboard cards. */
  _applyModuleLocks() {
    // Sidebar
    document.querySelectorAll('.nav-item[data-page]').forEach(item => {
      const page = item.dataset.page;
      if (this._isPageLocked(page)) {
        item.classList.add('locked');
      } else {
        item.classList.remove('locked');
      }
    });
    // Dashboard module cards (if already rendered)
    document.querySelectorAll('.module-card[data-module]').forEach(card => {
      const page = card.dataset.module;
      if (this._isPageLocked(page)) {
        card.classList.add('locked');
      } else {
        card.classList.remove('locked');
      }
    });
  },

  // ──────────── Router ────────────
  navigate(page) {
    if (this._isPageLocked(page)) return;  // Blocked — module not available
    this.currentPage = page;
    // Update nav active state
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const active = document.querySelector(`.nav-item[data-page="${page}"]`);
    if (active) active.classList.add('active');

    // Show/hide pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const pageEl = document.getElementById(`page-${page}`);
    if (pageEl) {
      pageEl.classList.add('active');
      this.loadPage(page);
    }
  },

  async loadPage(page) {
    switch (page) {
      case 'dashboard': return this.loadDashboard();
      case 'models': return this.loadModels();
      case 'elastic': return this.loadElastic();
      case 'direct-methods': return this.loadDirectMethods();
      case 'validation': return this.loadValidation();
      case 'shakedown': return this.loadShakedown();
      case 'docs': return this.loadDocs();
      case 'settings': return this.loadSettings();
      case 'tasks': return this.loadTasks();
      case 'visualization': return this.loadVisualization();
    }
  },

  // ──────────── API helpers ────────────
  async api(url) {
    const res = await fetch(url);
    return res.json();
  },
  async apiWithTimeout(url, timeoutMs = 8000, options = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      if (!res.ok) throw new Error(`API 错误 ${res.status}`);
      return res.json();
    } finally {
      clearTimeout(timeoutId);
    }
  },
  async apiPost(url, body) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return res.json();
  },
  async apiPut(url, body) {
    const res = await fetch(url, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return res.json();
  },

  winToWsl(path) {
    return String(path)
      .replace(/\\/g, '/')
      .replace(/^([A-Za-z]):\//, (_, drive) => `/mnt/${drive.toLowerCase()}/`);
  },

  _runTimestamp() {
    const now = new Date();
    const pad = (value) => String(value).padStart(2, '0');
    return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
      `_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  },

  _sanitizeRunToken(value, fallback = 'Run') {
    const text = String(value || '').trim()
      .replace(/\.[^.]+$/, '')
      .replace(/[^A-Za-z0-9_-]+/g, '_')
      .replace(/^_+|_+$/g, '');
    return text || fallback;
  },

  _joinWinPath(...parts) {
    return parts
      .map((part, index) => {
        let text = String(part || '').trim().replace(/[\\/]+$/g, '');
        if (index > 0) text = text.replace(/^[\\/]+/g, '');
        return text;
      })
      .filter(Boolean)
      .join('\\');
  },

  _joinWslPath(...parts) {
    return parts
      .map((part, index) => {
        let text = String(part || '').trim().replace(/[\\/]+$/g, '');
        if (index > 0) text = text.replace(/^[\\/]+/g, '');
        return text;
      })
      .filter(Boolean)
      .join('/');
  },

  _analysisRunPaths(model, workflowDir, label, cfgFileName) {
    const runLabel = `${this._runTimestamp()}_${this._sanitizeRunToken(label)}`;
    const modelRootWsl = this.winToWsl(model.path);
    const runDirWin = this._joinWinPath(model.path, workflowDir, runLabel);
    const runDirWsl = this._joinWslPath(modelRootWsl, workflowDir, runLabel);
    return {
      runLabel,
      runDirWin,
      runDirWsl,
      cfgWin: this._joinWinPath(runDirWin, cfgFileName),
      cfgWsl: this._joinWslPath(runDirWsl, cfgFileName),
    };
  },

  wslToWin(path) {
    const text = String(path || '').trim().replace(/\\/g, '/');
    const match = text.match(/^\/mnt\/([a-zA-Z])(?:\/(.*))?$/);
    if (!match) return String(path || '').trim();
    const tail = (match[2] || '').replace(/\//g, '\\');
    return tail ? `${match[1].toUpperCase()}:\\${tail}` : `${match[1].toUpperCase()}:\\`;
  },

  _normalizeWinPath(path) {
    let text = String(path || '').trim().replace(/\//g, '\\');
    if (/^[A-Za-z]:\\/.test(text)) {
      text = text[0].toUpperCase() + text.slice(1);
    }
    while (text.length > 3 && text.endsWith('\\')) {
      text = text.slice(0, -1);
    }
    return text;
  },

  _storedRootsEquivalent(winPath, wslPath) {
    const winNorm = this._normalizeWinPath(winPath);
    const wslAsWin = this._normalizeWinPath(this.wslToWin(wslPath));
    if (!winNorm || !wslAsWin) return false;
    return winNorm === wslAsWin;
  },

  _updateStoredRootHighlight() {
    const winInput = document.getElementById('cfg-win_stored_root');
    const wslInput = document.getElementById('cfg-wsl_stored_root');
    const note = document.getElementById('settings-stored-root-note');
    if (!winInput || !wslInput || !note) return;

    const same = this._storedRootsEquivalent(winInput.value, wslInput.value);
    [winInput, wslInput].forEach((input) => {
      input.style.borderColor = same ? 'rgba(52,211,153,0.55)' : '';
      input.style.boxShadow = same ? '0 0 0 3px rgba(52,211,153,0.12)' : '';
      input.style.background = same ? 'rgba(52,211,153,0.06)' : '';
    });

    if (same) {
      note.innerHTML = `<div style="padding:10px 14px;border-radius:8px;background:rgba(52,211,153,0.10);border:1px solid rgba(52,211,153,0.28);font-size:12px;color:#34d399">
        ✓ 当前 Windows / WSL 模型根目录指向同一个物理目录；这正是当前 MockWSL 架构下的推荐设置。
      </div>`;
      return;
    }

    note.innerHTML = `<div style="padding:10px 14px;border-radius:8px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);font-size:12px;color:var(--text-secondary)">
      Windows 模型根目录保留用于 Windows 原生流程；在当前 MockWSL 架构下，通常建议它与 WSL 模型根目录指向同一个物理目录，只是分别使用 Win/WSL 形式填写。
    </div>`;
  },

  _onStoredRootInputChange(fieldKey) {
    const winInput = document.getElementById('cfg-win_stored_root');
    const wslInput = document.getElementById('cfg-wsl_stored_root');
    if (!winInput || !wslInput) return;

    if (fieldKey === 'win_stored_root' && winInput.value.trim() && !wslInput.value.trim()) {
      wslInput.value = this.winToWsl(winInput.value.trim());
    } else if (fieldKey === 'wsl_stored_root' && wslInput.value.trim() && !winInput.value.trim()) {
      winInput.value = this.wslToWin(wslInput.value.trim());
    }
    this._updateStoredRootHighlight();
  },

  // ──────────── Dashboard ────────────
  async loadDashboard() {
    const container = document.getElementById('dashboard-content');
    container.innerHTML = '<div class="status-grid" id="status-grid">' +
      UI.statusCard('⏳', 'WSL 状态', '检测中...', 'amber') +
      UI.statusCard('📊', '模型数量', '...', 'blue') +
      UI.statusCard('🔬', 'JAX 版本', '检测中...', 'purple') +
      UI.statusCard('🐍', 'Win Python', '...', 'green') +
      '</div><div class="module-grid" id="dashboard-modules"></div>' +
      '<div id="dashboard-tasks"></div>';

    // Module cards — show immediately (no API needed)
    document.getElementById('dashboard-modules').innerHTML =
      UI.moduleCard('elastic', '📐', '增量分析',
        '根据 INP 自动切换增量弹性 / 弹塑性，并生成主 .mat 文件', 'blue', '开始分析') +
      UI.moduleCard('validation', '🔍', 'ODB 验证',
        '实体单元弹性 MAT 与 ABAQUS ODB 字段级对标', 'amber', '开始验证') +
      UI.moduleCard('shakedown', '🛡️', '安定分析',
        '基于弹性 .mat 结果进行结构安定性 SOCP 优化分析', 'purple', '开始分析') +
      UI.moduleCard('direct-methods', '∿', 'Direct Methods',
        '当前提供 DCA / RSDM / RSDM-S；结果 validation 统一收敛到验证模块', 'green', '进入模块');

    // Apply locked state to module cards that are unavailable
    this._applyModuleLocks();

    // Load fast data first (models + tasks)
    const [models, tasks] = await Promise.all([
      this.api('/api/models').catch(() => []),
      this.api('/api/tasks?limit=5').catch(() => []),
    ]);
    this.models = models;

    // Update model count card immediately
    const statusGrid = document.getElementById('status-grid');
    if (statusGrid) {
      // Replace only the model count card (second card)
      const cards = statusGrid.querySelectorAll('.status-card');
      if (cards[1]) {
        cards[1].outerHTML = UI.statusCard('📊', '可用模型', `${models.length} 个`, 'blue');
      }
    }

    // Recent tasks
    const tasksDiv = document.getElementById('dashboard-tasks');
    if (tasks.length > 0) {
      tasksDiv.innerHTML = '<div class="card"><div class="card-header"><span class="card-title">最近任务</span>' +
        '<button class="btn btn-secondary" onclick="App.navigate(\'tasks\')">查看全部</button></div>' +
        '<div class="task-list">' + tasks.map(t => UI.taskItem(t, { selectable: false })).join('') + '</div></div>';
    }

    // Load slow status data in background (WSL + JAX detection)
    this.api('/api/status').then(status => {
      this.envStatus = status;
      const wslOk = status?.wsl?.available === true;
      this.updateSidebarStatus(wslOk);
      if (this.currentPage !== 'dashboard') return;  // User navigated away
      const grid = document.getElementById('status-grid');
      if (!grid || !status) return;
      grid.innerHTML =
        UI.statusCard(
          wslOk ? '✅' : '❌', 'WSL 状态',
          wslOk ? '已安装' : '未安装',
          wslOk ? 'green' : 'red'
        ) +
        UI.statusCard('📊', '可用模型', `${models.length} 个`, 'blue') +
        UI.statusCard('🔬', 'JAX',
          status.jax?.version || '未检测到', status.jax?.available ? 'purple' : 'red') +
        UI.statusCard('🐍', 'Win Python',
          status.windows_python?.version || '?', 'green');
    }).catch(() => {
      this.updateSidebarStatus(false);
    });
  },

  updateSidebarStatus(wslAvailable) {
    const dot = document.getElementById('wsl-dot');
    const text = document.getElementById('wsl-text');
    if (dot) {
      dot.className = 'status-dot ' + (wslAvailable === true ? 'online' : wslAvailable === false ? 'offline' : 'checking');
    }
    if (text) {
      text.textContent = wslAvailable === true ? 'WSL 已安装' : wslAvailable === false ? 'WSL 未安装' : '检测中...';
    }
  },

  async refreshSidebarStatus() {
    try {
      const status = await this.apiWithTimeout('/api/status/quick', 8000);
      this.updateSidebarStatus(status?.wsl?.available === true);
    } catch (_) {
      this.updateSidebarStatus(false);
    }
  },

  // ──────────── Models ────────────
  async loadModels() {
    const container = document.getElementById('models-content');
    container.innerHTML = '<div class="empty-state"><div class="empty-icon">⏳</div><div class="empty-text">加载中...</div></div>';

    let models;
    try {
      const res = await fetch('/api/models');
      if (!res.ok) throw new Error(`API 错误 ${res.status}`);
      models = await res.json();
      if (!Array.isArray(models)) throw new Error('返回数据格式异常');
    } catch (err) {
      container.innerHTML = UI.hint('error', `模型列表加载失败：${err.message}。请检查服务器是否正常运行。`);
      return;
    }
    this.models = models;

    // Build header with toggle bar + model list container
    const toggleHtml = UI.toggleBar(
      [{ id: 'examples', label: '📚 Examples' }, { id: 'stored', label: '💾 StoredModels' }],
      this.modelFilter,
      'App.toggleModelFilter'
    );
    container.innerHTML = `<div style="margin-bottom:16px">${toggleHtml}</div><div id="model-list-body"></div>`;
    this.renderModelList();
  },

  toggleModelFilter(sourceId) {
    // 至少保留一个激活
    if (this.modelFilter.has(sourceId) && this.modelFilter.size === 1) return;
    if (this.modelFilter.has(sourceId)) {
      this.modelFilter.delete(sourceId);
    } else {
      this.modelFilter.add(sourceId);
    }
    // 更新 toggle 按钮样式
    const toggleBar = document.querySelector('.toggle-bar');
    if (toggleBar) {
      toggleBar.querySelectorAll('.toggle-btn').forEach(btn => {
        const id = btn.getAttribute('onclick').match(/'(\w+)'/)?.[1];
        btn.classList.toggle('active', id ? this.modelFilter.has(id) : false);
      });
    }
    this.renderModelList();
  },

  renderModelList() {
    const body = document.getElementById('model-list-body');
    if (!body) return;
    const filtered = this.models.filter(m => this.modelFilter.has(m.source));
    if (filtered.length === 0) {
      body.innerHTML = UI.empty('📁', '未找到任何模型。请将模型放入 Examples/ 或 StoredModels/ 目录。');
      return;
    }
    body.innerHTML = '<div class="model-list">' + filtered.map(m => UI.modelItem(m)).join('') + '</div>';
  },

  selectModel(source, name) {
    this.selectedModel = { source, name };
    const label = document.getElementById('sidebar-active-model');
    if (label) label.textContent = name;
    const targetPage = this._pendingModelTargetPage || 'elastic';
    this._pendingModelTargetPage = null;
    // Show confirmation modal before navigating
    UI.confirm(
      `已选取模型：<strong>${UI.escapeHtml(name)}</strong><br>点击确定后选择分析模块。`,
      () => this.navigate(targetPage)
    );

  },

  requestModelFor(page) {
    this._pendingModelTargetPage = page || this.currentPage || 'elastic';
    this.navigate('models');
  },

  _syncElasticDraftValues() {
    const draft = this._elastic?.cfgValues || {};
    const fieldMap = {
      material_model: 'elastic-material-model',
      E_override: 'elastic-E',
      nu_override: 'elastic-nu',
      yield_stress_override: 'elastic-yield',
      n_increments: 'elastic-n-increments',
      max_iterations: 'elastic-max-iterations',
      convergence_tol: 'elastic-convergence-tol',
    };
    Object.entries(fieldMap).forEach(([key, id]) => {
      const el = document.getElementById(id);
      if (el) draft[key] = el.value;
    });
    const useBExt = document.getElementById('elastic-b-ext');
    if (useBExt) draft.use_b_ext = useBExt.checked ? '1' : '0';
    this._elastic.cfgValues = draft;
  },

  _selectedElasticInpNames() {
    return [...document.querySelectorAll('#elastic-inp-list input[type=checkbox]')]
      .filter(item => item.checked)
      .map(item => item.dataset.inp);
  },

  _currentElasticSelectionKey(names = null) {
    const items = Array.isArray(names) ? names : this._selectedElasticInpNames();
    return [...items].sort().join('|');
  },

  _clearElasticParsedState() {
    if (!this._elastic) return;
    this._elastic.parsedInpMetaByName = {};
    this._elastic.parsedSelectionKey = '';
    this._elastic.lastParsedAt = '';
    this._elastic.cfgValues = {};
  },

  _selectedElasticInpMeta() {
    const currentKey = this._currentElasticSelectionKey();
    if (!currentKey || currentKey !== (this._elastic?.parsedSelectionKey || '')) {
      return [];
    }
    const metaByName = this._elastic?.parsedInpMetaByName || {};
    return this._selectedElasticInpNames().map(name => metaByName[name]).filter(Boolean);
  },

  _primaryElasticSavedMatMeta() {
    const primaryName = this._selectedElasticInpNames()[0];
    if (!primaryName) return null;
    const matName = String(primaryName).replace(/\.inp$/i, '.mat');
    return this._elastic?.resultMatMetaByName?.[matName] || null;
  },

  _isElasticBlankValue(value) {
    if (value === undefined || value === null) return true;
    const text = String(value).trim();
    if (!text) return true;
    const lowered = text.toLowerCase();
    return lowered === 'none' || lowered === 'nan';
  },

  _elasticValuesEqual(left, right) {
    if (this._isElasticBlankValue(left) && this._isElasticBlankValue(right)) return true;
    if (this._isElasticBlankValue(left) || this._isElasticBlankValue(right)) return false;

    const leftNum = Number(left);
    const rightNum = Number(right);
    if (Number.isFinite(leftNum) && Number.isFinite(rightNum)) {
      const scale = Math.max(1, Math.abs(leftNum), Math.abs(rightNum));
      return Math.abs(leftNum - rightNum) <= 1e-12 * scale;
    }
    return String(left).trim() === String(right).trim();
  },

  _formatElasticValue(value) {
    if (this._isElasticBlankValue(value)) return '未设置';
    const num = Number(value);
    if (Number.isFinite(num) && Math.abs(num) > 0 && (Math.abs(num) < 1e-3 || Math.abs(num) >= 1e4)) {
      return num.toExponential(6);
    }
    return String(value);
  },

  _buildElasticDiffItems(analysisInput, inpMeta) {
    if (!analysisInput || !inpMeta) {
      return { diffs: [], controls: [] };
    }

    const material = inpMeta.materials?.[0] || {};
    const elastic = material.elastic || {};
    const plastic = material.plastic || {};
    const parsedDefaults = analysisInput.parsed_defaults || {};
    const expectedMaterialModel = String(inpMeta.suggested_material_model || 'linear_elastic').trim() || 'linear_elastic';
    const diffs = [];
    const pushDiff = (label, current, original) => {
      if (this._elasticValuesEqual(current, original)) return;
      diffs.push({
        label,
        current: this._formatElasticValue(current),
        original: this._formatElasticValue(original),
      });
    };

    pushDiff('分析类型', analysisInput.material_model, expectedMaterialModel);
    pushDiff("Young's modulus (E)", analysisInput.E_override, parsedDefaults.material_E ?? elastic.E);
    pushDiff('Poisson ratio (nu)', analysisInput.nu_override, parsedDefaults.material_nu ?? elastic.nu);

    if (
      String(analysisInput.material_model || '').trim() === 'j2_perfect_plastic'
      || !this._isElasticBlankValue(analysisInput.yield_stress_override)
    ) {
      pushDiff('Yield stress', analysisInput.yield_stress_override, parsedDefaults.yield_stress ?? plastic.yield_stress);
    }

    pushDiff(
      'Increment count',
      analysisInput.n_increments,
      parsedDefaults.default_n_increments ?? inpMeta.step?.default_n_increments ?? 1,
    );

    const controls = [];
    if (String(analysisInput.use_b_ext ?? '0') !== '0') {
      controls.push({ label: 'B-bar 扩展', value: '启用' });
    }
    if (!this._isElasticBlankValue(analysisInput.max_iterations)) {
      controls.push({ label: 'Max iterations', value: this._formatElasticValue(analysisInput.max_iterations) });
    }
    if (!this._isElasticBlankValue(analysisInput.convergence_tol)) {
      controls.push({ label: 'Convergence tolerance', value: this._formatElasticValue(analysisInput.convergence_tol) });
    }
    return { diffs, controls };
  },

  _buildElasticLiveAnalysisInput(inpMeta) {
    if (!inpMeta) return null;

    const draft = this._elastic?.cfgValues || {};
    const material = inpMeta.materials?.[0] || {};
    const elastic = material.elastic || {};
    const plastic = material.plastic || {};
    const step = inpMeta.step || {};

    let materialModel = String(draft.material_model || inpMeta.suggested_material_model || 'linear_elastic').trim() || 'linear_elastic';
    if (!inpMeta.supports_elastoplastic) {
      materialModel = 'linear_elastic';
    }

    return {
      material_model: materialModel,
      E_override: draft.E_override,
      nu_override: draft.nu_override,
      yield_stress_override: materialModel === 'j2_perfect_plastic' ? draft.yield_stress_override : '',
      use_b_ext: draft.use_b_ext ?? '0',
      n_increments: this._isElasticBlankValue(draft.n_increments)
        ? (step.default_n_increments ?? 1)
        : draft.n_increments,
      max_iterations: draft.max_iterations ?? '',
      convergence_tol: draft.convergence_tol ?? '',
      parsed_defaults: {
        material_E: elastic.E ?? null,
        material_nu: elastic.nu ?? null,
        yield_stress: plastic.yield_stress ?? null,
        initial_increment: step.initial_increment ?? null,
        total_time: step.total_time ?? null,
        default_n_increments: step.default_n_increments ?? 1,
      },
    };
  },

  _renderElasticDiffSummary(summary, {
    currentLabel = '当前配置',
    originalLabel = '原 INP',
    emptyMessage = '当前配置与原 INP 一致。',
    controlsCaption = '附加求解控制（非 INP 原始字段）',
  } = {}) {
    if (!summary || (!summary.diffs.length && !summary.controls.length)) {
      return UI.hint('info', emptyMessage);
    }

    const diffRows = summary.diffs.map((entry) =>
      `<div style="padding:8px 10px;border-radius:8px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06)">
        <div style="font-size:12px;color:var(--text-primary)">${UI.escapeHtml(entry.label)}</div>
        <div style="font-size:11px;color:var(--text-secondary);margin-top:4px">${UI.escapeHtml(currentLabel)}: ${UI.escapeHtml(entry.current)} | ${UI.escapeHtml(originalLabel)}: ${UI.escapeHtml(entry.original)}</div>
      </div>`
    ).join('');

    const controlRows = summary.controls.map((entry) =>
      `<div style="padding:8px 10px;border-radius:8px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06)">
        <div style="font-size:12px;color:var(--text-primary)">${UI.escapeHtml(entry.label)}</div>
        <div style="font-size:11px;color:var(--text-secondary);margin-top:4px">${UI.escapeHtml(entry.value)}</div>
      </div>`
    ).join('');

    return `<div style="display:flex;flex-direction:column;gap:8px">
      ${diffRows}
      ${controlRows ? `<div style="font-size:12px;color:var(--text-secondary);margin-top:2px">${UI.escapeHtml(controlsCaption)}</div>${controlRows}` : ''}
    </div>`;
  },

  _renderElasticCurrentDiffPreview() {
    const container = document.getElementById('elastic-current-diff-summary');
    if (!container) return;

    const selectedNames = this._selectedElasticInpNames();
    if (!selectedNames.length) {
      container.innerHTML = UI.hint('warning', '请先勾选至少一个 INP。');
      return;
    }

    const currentKey = this._currentElasticSelectionKey(selectedNames);
    const selectedMeta = this._selectedElasticInpMeta();
    if (!currentKey || currentKey !== (this._elastic?.parsedSelectionKey || '') || selectedMeta.length !== selectedNames.length) {
      container.innerHTML = UI.hint('info', '请先对当前勾选的 INP 完成 fresh parse，再查看提交前差异提醒。');
      return;
    }

    const primary = selectedMeta[0] || null;
    const liveInput = this._buildElasticLiveAnalysisInput(primary);
    const summary = this._buildElasticDiffItems(liveInput, primary);
    const leadText = selectedNames.length > 1
      ? '当前按表单中的即时值展示首个工况的提醒；如果现在提交，同一套设置会统一应用到全部已选 INP，并一并写入各自结果 MAT。'
      : '当前按表单中的即时值展示；如果现在提交，这些差异和附加求解控制会一并写入结果 MAT。';

    container.innerHTML = `<div style="font-size:13px;color:var(--text-secondary);margin-top:10px;line-height:1.7">${UI.escapeHtml(leadText)}</div>
      <div style="margin-top:12px">${this._renderElasticDiffSummary(summary, {
        currentLabel: '当前表单',
        originalLabel: '原 INP',
        emptyMessage: '当前表单与原 INP 一致；如果现在提交，这次分析不会写入额外覆盖项。',
      })}</div>`;
  },

  _renderElasticParseStatus() {
    const statusEl = document.getElementById('elastic-parse-status');
    if (!statusEl) return;

    const selectedNames = this._selectedElasticInpNames();
    if (!selectedNames.length) {
      statusEl.style.color = 'var(--text-secondary)';
      statusEl.textContent = '请先勾选至少一个 INP。';
      return;
    }

    const currentKey = this._currentElasticSelectionKey(selectedNames);
    if (currentKey && currentKey === (this._elastic?.parsedSelectionKey || '') && this._elastic?.lastParsedAt) {
      statusEl.style.color = '#34d399';
      statusEl.textContent = `已按当前勾选完成 fresh parse：${this._elastic.lastParsedAt}`;
      return;
    }

    statusEl.style.color = 'var(--text-secondary)';
    statusEl.textContent = '当前勾选项尚未解析；提交计算前请先点击“解析所选 INP”。';
  },

  _buildElasticDraftFromParsed(primary) {
    if (!primary) return {};
    const material = primary.materials?.[0] || {};
    const elastic = material.elastic || {};
    const plastic = material.plastic || {};
    const step = primary.step || {};
    let materialModel = String(primary.suggested_material_model || 'linear_elastic').trim() || 'linear_elastic';
    if (!primary.supports_elastoplastic) {
      materialModel = 'linear_elastic';
    }
    return {
      material_model: materialModel,
      E_override: elastic.E ?? '',
      nu_override: elastic.nu ?? '',
      yield_stress_override: materialModel === 'j2_perfect_plastic' ? (plastic.yield_stress ?? '') : '',
      use_b_ext: '0',
      n_increments: step.default_n_increments ?? 1,
      max_iterations: '',
      convergence_tol: '',
    };
  },

  _renderElasticHistoryRows(savedMatMeta) {
    const history = Array.isArray(savedMatMeta?.analysis_input_history) ? savedMatMeta.analysis_input_history : [];
    const inpMeta = savedMatMeta?.inp_metadata || null;
    return history.slice(-6).reverse().map((item) => {
      const summary = this._buildElasticDiffItems(item?.analysis_input || null, inpMeta);
      const diffText = summary.diffs.length
        ? summary.diffs.map(entry => `${entry.label}: ${entry.current} (INP: ${entry.original})`).join('；')
        : '与原 INP 一致';
      const controlText = summary.controls.length
        ? `；附加求解控制: ${summary.controls.map(entry => `${entry.label} = ${entry.value}`).join('，')}`
        : '';
      const changedText = Array.isArray(item?.changed_fields) && item.changed_fields.length
        ? item.changed_fields.join(', ')
        : '首次写入';
      return `<div style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05)">
        <div style="font-size:12px;color:var(--text-primary)">${UI.escapeHtml(String(item?.saved_at || '-'))}</div>
        <div style="font-size:11px;color:var(--text-secondary);margin-top:2px">来源: ${UI.escapeHtml(String(item?.source || '-'))} | 变更字段: ${UI.escapeHtml(changedText)}</div>
        <div style="font-size:11px;color:var(--text-secondary);margin-top:4px;line-height:1.7">${UI.escapeHtml(diffText + controlText)}</div>
      </div>`;
    }).join('');
  },

  _onElasticInpSelectionChange() {
    this._clearElasticParsedState();
    this._renderElasticConfigForm();
  },

  _onElasticMaterialModelChange() {
    this._syncElasticDraftValues();
    this._renderElasticConfigForm();
  },

  _onElasticDraftChange() {
    this._syncElasticDraftValues();
    this._renderElasticCurrentDiffPreview();
  },

  async parseSelectedElasticInps() {
    const model = this._elastic?.model;
    const formArea = document.getElementById('elastic-analysis-config-area');
    if (!model || !formArea) return;

    const selectedNames = this._selectedElasticInpNames();
    if (!selectedNames.length) {
      alert('请至少勾选一个 INP。');
      this._renderElasticConfigForm();
      return;
    }

    formArea.innerHTML = '<div class="card" style="margin-bottom:24px"><div class="card-title">增量分析配置</div><div style="font-size:13px;color:var(--text-secondary);margin-top:10px">正在重新解析当前勾选的 INP，请稍候...</div></div>';
    this._renderElasticParseStatus();

    const inpMetaRes = await this.apiPost('/api/models/inp-meta', {
      paths: selectedNames.map(f => `${model.path}\\abaqus\\${f}`),
    }).catch(() => null);

    if (!inpMetaRes?.details?.length) {
      formArea.innerHTML = UI.hint('error', 'INP 解析失败，请检查后重试。');
      return;
    }

    const parsedInpMetaByName = {};
    (inpMetaRes.details || []).forEach(item => {
      parsedInpMetaByName[item.name] = item;
    });

    const primary = selectedNames.map(name => parsedInpMetaByName[name]).filter(Boolean)[0] || null;
    this._elastic.parsedInpMetaByName = parsedInpMetaByName;
    this._elastic.parsedSelectionKey = this._currentElasticSelectionKey(selectedNames);
    this._elastic.lastParsedAt = new Date().toLocaleString('zh-CN', { hour12: false });
    this._elastic.cfgValues = this._buildElasticDraftFromParsed(primary);
    this._renderElasticConfigForm();
  },

  _renderElasticConfigForm() {
    const formArea = document.getElementById('elastic-analysis-config-area');
    if (!formArea) return;

    this._syncElasticDraftValues();
    const draft = this._elastic?.cfgValues || {};
    const selectedNames = this._selectedElasticInpNames();
    const currentKey = this._currentElasticSelectionKey(selectedNames);
    const selectedMeta = this._selectedElasticInpMeta();
    this._renderElasticParseStatus();

    if (!selectedNames.length) {
      formArea.innerHTML = UI.hint('warning', '请至少勾选一个 INP 后再配置增量分析。');
      return;
    }
    if (!currentKey || currentKey !== (this._elastic?.parsedSelectionKey || '')) {
      formArea.innerHTML = `<div class="card" style="margin-bottom:24px">
        <div class="card-title">增量分析配置</div>
        <div style="font-size:13px;color:var(--text-secondary);margin-top:10px;line-height:1.7">
          当前勾选项还没有 fresh parse 结果。请先点击上方的“解析所选 INP”，再基于解析结果填写和提交增量分析参数。
        </div>
      </div>`;
      return;
    }
    if (selectedMeta.length !== selectedNames.length) {
      formArea.innerHTML = UI.hint('error', '部分已勾选 INP 的元数据解析失败，请取消异常文件后重试。');
      return;
    }
    if (selectedMeta.some(item => item?.supported === false || item?.error)) {
      const invalidNames = selectedMeta.filter(item => item?.supported === false || item?.error).map(item => item.name).join('、');
      formArea.innerHTML = UI.hint('error', `这些 INP 当前无法用于 Web 增量分析：${invalidNames}`);
      return;
    }

    const families = [...new Set(selectedMeta.map(item => item.family || 'unknown'))];
    const plasticModes = [...new Set(selectedMeta.map(item => Boolean(item.supports_elastoplastic)))];
    if (families.length > 1 || plasticModes.length > 1) {
      formArea.innerHTML = UI.hint('warning', '当前勾选的 INP 在 family 或材料模型能力上不一致，请拆分成多次运行。');
      return;
    }

    const primary = selectedMeta[0];
    const savedMatMeta = this._primaryElasticSavedMatMeta();
    const material = primary.materials?.[0];
    if (!material) {
      formArea.innerHTML = UI.hint('error', '未能读取当前 INP 的主材料信息。');
      return;
    }

    const isSolid = primary.family === 'solid';
    if (!isSolid) {
      formArea.innerHTML = UI.hint('warning', 'demo 版 inc_analysis 仅开放实体单元线弹性流程，请选择 solid INP。');
      return;
    }
    const supportsPlastic = false;
    let materialModel = String(draft.material_model || primary.suggested_material_model || 'linear_elastic').trim();
    materialModel = 'linear_elastic';

    const elastic = material.elastic || {};
    const plastic = material.plastic || {};
    const resolveDraft = (key, fallback) => {
      const raw = draft[key];
      if (raw !== undefined && raw !== null && String(raw).trim() !== '' && String(raw).trim().toLowerCase() !== 'none') {
        return raw;
      }
      return fallback ?? '';
    };
    const EValue = resolveDraft('E_override', elastic.E);
    const nuValue = resolveDraft('nu_override', elastic.nu);
    const yieldValue = resolveDraft('yield_stress_override', plastic.yield_stress);
    const useBExt = String(draft.use_b_ext ?? '0') !== '0';
    const step = primary.step || {};
    const nIncrementsValue = resolveDraft('n_increments', step.default_n_increments || 1);
    const maxIterationsValue = resolveDraft('max_iterations', '');
    const convergenceTolValue = resolveDraft('convergence_tol', '');

    let analysisModeField = '';
    if (supportsPlastic) {
      analysisModeField = `<div class="form-group">
        <label class="form-label">分析类型</label>
        <select class="form-input" id="elastic-material-model" onchange="App._onElasticMaterialModelChange()">
          <option value="linear_elastic" ${materialModel === 'linear_elastic' ? 'selected' : ''}>增量弹性</option>
          <option value="j2_perfect_plastic" ${materialModel === 'j2_perfect_plastic' ? 'selected' : ''}>弹塑性 J2 ideal</option>
        </select>
        <div class="form-hint" style="margin-top:4px">当前 INP 检测到 plastic 材料定义，支持在弹性与 small-strain J2 perfect plasticity 之间切换。</div>
      </div>`;
    } else {
      analysisModeField = `<div class="form-group">
        <label class="form-label">分析类型</label>
        <input class="form-input" id="elastic-material-model" value="linear_elastic" readonly style="opacity:0.72;cursor:not-allowed" />
        <div class="form-hint" style="margin-top:4px">${UI.escapeHtml(primary.support_note || '当前 INP 未检测到可用 plastic 材料定义，默认走增量弹性。')}</div>
      </div>`;
    }

    let html = `<div class="card" style="margin-bottom:16px">
      <div class="card-title">增量分析配置</div>
      <div style="font-size:13px;color:var(--text-secondary);margin-top:10px;line-height:1.7">
        默认值来自刚刚解析的 <b style="color:var(--text-primary)">${UI.escapeHtml(primary.name)}</b>。
        ${selectedNames.length > 1 ? `当前共勾选 ${selectedNames.length} 个 INP，以下设置会统一应用到全部已选工况。` : '当前仅对这个 INP 运行分析。'}
      </div>
      <div style="font-size:12px;color:var(--accent);margin-top:8px">最近解析时间：${UI.escapeHtml(this._elastic?.lastParsedAt || '-')}</div>
      <div style="margin-top:16px;display:flex;flex-direction:column;gap:16px">
        ${analysisModeField}
        <div class="form-group">
          <label class="form-label">材料名称</label>
          <input class="form-input" type="text" value="${UI.escapeAttr(material.name || '')}" readonly style="opacity:0.72;cursor:not-allowed" />
        </div>
        <div class="form-group">
          <label class="form-label">Young's modulus (E)</label>
          <input class="form-input" type="number" id="elastic-E" value="${UI.escapeAttr(String(EValue))}" step="any" oninput="App._onElasticDraftChange()" onchange="App._onElasticDraftChange()" />
        </div>
        <div class="form-group">
          <label class="form-label">Poisson ratio (nu)</label>
          <input class="form-input" type="number" id="elastic-nu" value="${UI.escapeAttr(String(nuValue))}" step="any" oninput="App._onElasticDraftChange()" onchange="App._onElasticDraftChange()" />
        </div>`;

    if (materialModel === 'j2_perfect_plastic') {
      html += `<div class="form-group">
        <label class="form-label">Yield stress</label>
        <input class="form-input" type="number" id="elastic-yield" value="${UI.escapeAttr(String(yieldValue))}" step="any" oninput="App._onElasticDraftChange()" onchange="App._onElasticDraftChange()" />
        <div class="form-hint" style="margin-top:4px">当前 hardening: ${UI.escapeHtml(plastic.hardening || 'perfect')}。</div>
      </div>`;
    }

    if (elastic.thickness !== undefined) {
      html += `<div style="font-size:12px;color:var(--text-secondary)">Shell thickness = ${UI.escapeHtml(String(elastic.thickness))}（当前仅展示，不在 Web 中覆盖）。</div>`;
    }

    html += `</div></div>`;

    html += `<div class="card" style="margin-bottom:16px">
      <div class="card-title">求解设置</div>
      <div style="margin-top:14px;display:flex;flex-direction:column;gap:14px">
        <div class="form-group" style="margin:0">
          <label class="form-label">单元族</label>
          <input class="form-input" type="text" value="${UI.escapeAttr(primary.family || '')}" readonly style="opacity:0.72;cursor:not-allowed" />
        </div>`;

    if (isSolid) {
      html += `<div class="form-group" style="margin:0">
        <label class="form-label">B-bar 扩展</label>
        <div style="display:flex;align-items:center;gap:10px;margin-top:6px">
          <input type="checkbox" id="elastic-b-ext" ${useBExt ? 'checked' : ''} onchange="App._onElasticDraftChange()" style="width:16px;height:16px;accent-color:var(--accent);cursor:pointer">
          <label for="elastic-b-ext" style="font-size:14px;cursor:pointer">对 C3D8 启用 B-bar 扩展</label>
        </div>
        <div class="form-hint" style="margin-top:4px">适用于实体单元 C3D8 的体积锁定缓解；shell 分支不会使用这个选项。</div>
      </div>`;
    } else {
      html += `<div style="font-size:12px;color:var(--text-secondary)">Shell 分支当前固定为增量弹性；B-bar 选项不适用。</div>`;
    }

    html += `<div style="font-size:12px;color:var(--text-secondary);line-height:1.7">
      INP Step 信息：procedure = ${UI.escapeHtml(String(step.procedure || '-'))}，
      initial_increment = ${UI.escapeHtml(String(step.initial_increment ?? '-'))}，
      total_time = ${UI.escapeHtml(String(step.total_time ?? '-'))}，
      默认 increment 数 = ${UI.escapeHtml(String(step.default_n_increments || 1))}。
    </div>
        <div class="form-group" style="margin:0">
          <label class="form-label">Increment count</label>
          <input class="form-input" type="number" id="elastic-n-increments" min="1" step="1" value="${UI.escapeAttr(String(nIncrementsValue))}" oninput="App._onElasticDraftChange()" onchange="App._onElasticDraftChange()" />
          <div class="form-hint" style="margin-top:4px">这个默认值直接来自当前 INP step 解析结果；提交计算时会和用户修改一起写入最终结果 MAT。</div>
        </div>
        <div class="form-group" style="margin:0">
          <label class="form-label">Max iterations</label>
          <input class="form-input" type="number" id="elastic-max-iterations" min="1" step="1" value="${UI.escapeAttr(String(maxIterationsValue))}" placeholder="留空则使用程序默认值" oninput="App._onElasticDraftChange()" onchange="App._onElasticDraftChange()" />
        </div>
        <div class="form-group" style="margin:0">
          <label class="form-label">Convergence tolerance</label>
          <input class="form-input" type="number" id="elastic-convergence-tol" step="any" value="${UI.escapeAttr(String(convergenceTolValue))}" placeholder="留空则使用程序默认值" oninput="App._onElasticDraftChange()" onchange="App._onElasticDraftChange()" />
        </div>
      </div>
    </div>`;

    const historyRows = this._renderElasticHistoryRows(savedMatMeta);
    const savedHistorySection = (() => {
      if (!savedMatMeta?.inp_metadata) {
        return '';
      }
      return `<div style="margin-top:14px;font-size:12px;color:var(--text-secondary)">当前正式结果 MAT 最近保存历史</div>
      <div style="margin-top:6px">${historyRows || '<div style="font-size:12px;color:var(--text-secondary)">暂无历史记录。</div>'}</div>`;
    })();

    html += `<div class="card" style="margin-bottom:16px">
      <div class="card-title">提交前差异提醒</div>
      <div id="elastic-current-diff-summary"></div>
      ${savedHistorySection}
    </div>`;

    html += `<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:4px;margin-bottom:28px">
      <button class="btn btn-primary" onclick="App.saveAndRunElastic()">▶ 基于当前解析结果运行增量分析</button>
    </div>`;
    formArea.innerHTML = html;
    this._renderElasticCurrentDiffPreview();
    this._renderElasticParseStatus();
  },

  // ──────────── Elastic Analysis ────────────
  async loadElastic() {
    const container = document.getElementById('elastic-content');
    if (!this.selectedModel) {
      container.innerHTML = UI.hint('warning', '请先在模型浏览中选择一个工作模型。', '前往模型浏览', "App.requestModelFor('elastic')");
      return;
    }
    const { source, name } = this.selectedModel;
    container.innerHTML = '<div class="empty-state"><div class="empty-icon">⏳</div><div class="empty-text">加载中...</div></div>';

    const model = await this.api(`/api/models/${source}/${name}`).catch(() => null);
    if (!model) {
      container.innerHTML = UI.hint('error', '无法获取模型状态。');
      return;
    }
    if (model.abaqus.inp_count === 0) {
      container.innerHTML = UI.hint('error', '未找到 INP 文件。请确认 ABAQUS INP 文件已放入 abaqus/ 目录。');
      return;
    }

    // Load cfg values only for selected INP list.
    const cfgRes = await this.api(`/api/config/inc_analysis?model_path=${encodeURIComponent(model.path)}`).catch(() => ({ values: {} }));
    const vals = cfgRes.values || {};

    // Parse which INPs are selected in cfg (if any)
    let selectedInps = new Set(model.abaqus.inp_files); // default: all selected
    if (vals['inp_files']) {
      // cfg has explicit list; extract basenames
      const listed = vals['inp_files'].replace(/[\[\]\n]/g, ' ')
        .split(',').map(s => s.trim().replace(/"/g, '').split('/').pop().split('\\').pop()).filter(Boolean);
      if (listed.length) selectedInps = new Set(listed);
    }

    const resultMatMetaByName = {};
    const resultMatPaths = (model.inc_analysis.mat_files || []).map(name => `${model.path}\\inc_analysis\\${name}`);
    if (resultMatPaths.length) {
      const resultMetaRes = await this.apiPost('/api/models/mat-meta', {
        paths: resultMatPaths,
      }).catch(() => null);
      (resultMetaRes?.details || []).forEach(item => {
        resultMatMetaByName[item.name] = item;
      });
    }

    this._elastic = {
      model,
      parsedInpMetaByName: {},
      parsedSelectionKey: '',
      lastParsedAt: '',
      resultMatMetaByName,
      cfgValues: {},
    };

    let html = `<div style="font-size:13px;color:var(--text-secondary);margin-bottom:12px;padding:10px 14px;
        background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid rgba(255,255,255,0.06)">
      模型: <b style="color:var(--text-primary)">${UI.escapeHtml(model.name)}</b>
      <span style="color:rgba(255,255,255,0.25);margin:0 8px">|</span>
      <code style="font-size:11px;color:var(--accent)">${UI.escapeHtml(model.path)}</code>
    </div>`;

    html += `<div class="card" style="margin-bottom:16px">
      <div class="card-title">工况（INP）选择</div>
      <div style="font-size:13px;color:var(--text-secondary);margin:10px 0 14px">勾选本次需要计算的工况。若已勾选多个 INP，下面的材料与求解设置会统一应用到全部已选文件。</div>
      <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px">
        <button class="btn btn-secondary" onclick="App.parseSelectedElasticInps()">解析所选 INP</button>
        <span id="elastic-parse-status" style="font-size:12px;color:var(--text-secondary)">当前勾选项尚未解析；提交计算前请先点击“解析所选 INP”。</span>
      </div>
      <div id="elastic-inp-list">`;
    model.abaqus.inp_files.forEach((f, i) => {
      const checked = selectedInps.has(f) ? 'checked' : '';
      html += `<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06)">
        <input type="checkbox" id="inp-check-${i}" data-inp="${UI.escapeAttr(f)}" ${checked} onchange="App._onElasticInpSelectionChange()" style="width:16px;height:16px;flex-shrink:0;accent-color:var(--accent);cursor:pointer">
        <label for="inp-check-${i}" style="font-size:14px;cursor:pointer;user-select:none">📄 ${UI.escapeHtml(f)}</label>
      </div>`;
    });
    html += `</div></div>`;

    html += `<div id="elastic-analysis-config-area"></div>`;

    // Current results with timestamps
    if (model.inc_analysis.has_mat) {
      const fmtTime = (ts) => {
        const d = new Date(ts * 1000);
        return d.toLocaleString('zh-CN', { hour12: false });
      };
      const infos = model.inc_analysis.mat_files_info || model.inc_analysis.mat_files.map(n => ({ name: n, mtime: 0 }));
      const matList = infos.map(f =>
        `<div style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05)">
          <div style="font-size:13px;font-family:monospace;color:var(--text-secondary)">📦 ${model.path}\\inc_analysis\\${f.name}</div>
          ${f.mtime ? `<div style="font-size:11px;color:var(--text-secondary);margin-top:2px;padding-left:20px">生成时间: ${fmtTime(f.mtime)}</div>` : ''}
        </div>`
      ).join('');
      html += `<div class="card" style="margin-top:28px;margin-bottom:16px">
        <div class="card-title">当前增量分析结果</div>
        <div style="margin-top:8px">${matList}</div>
        <div style="font-size:12px;color:var(--text-secondary);margin-top:10px">如需重新计算，先重新解析当前 INP，再调整选项并点击运行。安定分析读取的是这里的正式结果 MAT，不会使用临时解析 sidecar。</div>
      </div>`;
    }
    container.innerHTML = html;
    this._renderElasticConfigForm();
    this._renderElasticParseStatus();
  },

  _collectElasticSubmission() {
    this._syncElasticDraftValues();
    const model = this._elastic?.model;
    if (!model) return null;

    // Collect selected INPs
    const selectedInps = this._selectedElasticInpNames();
    if (!selectedInps.length) {
      alert('请至少勾选一个 INP。');
      return null;
    }

    const currentKey = this._currentElasticSelectionKey(selectedInps);
    const selectedMeta = this._selectedElasticInpMeta();
    if (!currentKey || currentKey !== (this._elastic?.parsedSelectionKey || '') || selectedMeta.length !== selectedInps.length) {
      alert('请先对当前勾选的 INP 执行解析。');
      return null;
    }

    const materialModel = String(document.getElementById('elastic-material-model')?.value || 'linear_elastic').trim();
    const useBExt = document.getElementById('elastic-b-ext')?.checked ? 1 : 0;
    const eValue = document.getElementById('elastic-E')?.value?.trim();
    const nuValue = document.getElementById('elastic-nu')?.value?.trim();
    const yieldValue = document.getElementById('elastic-yield')?.value?.trim();
    const nIncrements = document.getElementById('elastic-n-increments')?.value?.trim() || String(selectedMeta[0]?.step?.default_n_increments || '1');
    const maxIterations = document.getElementById('elastic-max-iterations')?.value?.trim() || '';
    const convergenceTol = document.getElementById('elastic-convergence-tol')?.value?.trim() || '';

    return {
      model,
      selectedInps,
      selectedMeta,
      materialModel,
      useBExt,
      eValue,
      nuValue,
      yieldValue,
      nIncrements,
      maxIterations,
      convergenceTol,
    };
  },

  async saveAndRunElastic() {
    const payload = this._collectElasticSubmission();
    if (!payload) return;

    const { model, selectedInps, materialModel, useBExt, eValue, nuValue, yieldValue, nIncrements, maxIterations, convergenceTol } = payload;

    const modelRoot = this.winToWsl(model.path);
    const materialLabel = materialModel === 'j2_perfect_plastic' ? 'J2' : 'Linear';
    const run = this._analysisRunPaths(
      model,
      'inc_analysis',
      `Inc_${materialLabel}_${nIncrements}Step`,
      'inc_analysis.template.cfg',
    );

    await this.apiPost('/api/models/prepare-inc-analysis', {
      model_path: model.path,
      inp_paths: selectedInps.map(f => `${model.path}\\abaqus\\${f}`),
      analysis_input: {
        material_model: materialModel,
        E_override: eValue,
        nu_override: nuValue,
        yield_stress_override: materialModel === 'j2_perfect_plastic' ? yieldValue : '',
        use_b_ext: useBExt,
        n_increments: nIncrements,
        max_iterations: maxIterations,
        convergence_tol: convergenceTol,
      },
      source: 'web_inc_analysis',
    });

    let cfgLines = [
      '# inc_analysis configuration (auto-generated by jaxmech web incremental form)',
      `# run_dir: ${run.runLabel}`,
      '',
    ];
    const fullPaths = selectedInps.map(f => `    "${modelRoot}/abaqus/${f}"`);
    cfgLines.push('inp_files = [');
    cfgLines.push(fullPaths.join(',\n'));
    cfgLines.push(']');
    cfgLines.push('');
    if (useBExt) cfgLines.push('use_b_ext = 1');
    cfgLines.push(`n_increments = ${nIncrements}`);
    if (maxIterations) cfgLines.push(`max_iterations = ${maxIterations}`);
    if (convergenceTol) cfgLines.push(`convergence_tol = ${convergenceTol}`);
    cfgLines.push(`material_model = ${materialModel}`);
    if (eValue) cfgLines.push(`E_override = ${eValue}`);
    if (nuValue) cfgLines.push(`nu_override = ${nuValue}`);
    if (materialModel === 'j2_perfect_plastic' && yieldValue) cfgLines.push(`yield_stress_override = ${yieldValue}`);
    cfgLines.push('');
    const content = cfgLines.join('\n');

    await this.apiPut(`/api/config/inc_analysis?model_path=${encodeURIComponent(run.runDirWin)}`, { content });

    const result = await this.apiPost('/api/run', {
      module: 'jaxmech.tools.build_model_mat',
      args: ['--config', run.cfgWsl],
    });
    this.viewTask(result.id);
  },


  // ──────────── Direct Methods ────────────
  _selectedDcaInpName() {
    return document.querySelector('input[name="dca-inp"]:checked')?.value?.trim() || '';
  },

  _currentDcaSelectionKey() {
    return this._selectedDcaInpName();
  },

  _selectedDcaInpMeta() {
    const currentKey = this._currentDcaSelectionKey();
    if (!currentKey || currentKey !== (this._directMethods?.parsedSelectionKey || '')) {
      return null;
    }
    const inpName = this._selectedDcaInpName();
    return this._directMethods?.parsedInpMetaByName?.[inpName] || null;
  },

  _clearDcaParsedState() {
    if (!this._directMethods) return;
    this._directMethods.parsedInpMetaByName = {};
    this._directMethods.parsedSelectionKey = '';
    this._directMethods.lastParsedAt = '';
    this._directMethods.cfgValues = {};
  },

  _dcaDefaultNTimePoints(meta) {
    const value = Number(meta?.n_time_points);
    return Number.isFinite(value) && value >= 3 ? String(Math.round(value)) : '33';
  },

  _dcaDefaultNHarmonics(meta) {
    const value = Number(meta?.n_harmonics);
    return Number.isFinite(value) && value >= 0 ? String(Math.round(value)) : '15';
  },

  _dcaDefaultMaxIterations(meta) {
    const value = Number(meta?.max_iterations);
    return Number.isFinite(value) && value > 0 ? String(Math.round(value)) : '100';
  },

  _dcaDefaultConvergenceTol(meta) {
    return this._isElasticBlankValue(meta?.default_convergence_tol) ? '1.0e-7' : String(meta.default_convergence_tol);
  },

  _dcaDefaultCycleTolerance(meta) {
    return this._isElasticBlankValue(meta?.default_cycle_tolerance) ? '1.0e-7' : String(meta.default_cycle_tolerance);
  },

  _dcaDefaultUseBExt() {
    return '1';
  },

  _buildDcaDraftFromParsed(primary) {
    if (!primary) return {};
    const material = primary.materials?.[0] || {};
    const elastic = material.elastic || {};
    const plastic = material.plastic || {};
    return {
      n_time_points: this._dcaDefaultNTimePoints(primary),
      n_harmonics: this._dcaDefaultNHarmonics(primary),
      max_iterations: this._dcaDefaultMaxIterations(primary),
      load_factor: '1.0',
      convergence_tol: this._dcaDefaultConvergenceTol(primary),
      cycle_tolerance: this._dcaDefaultCycleTolerance(primary),
      use_b_ext: this._dcaDefaultUseBExt(),
      E: elastic.E ?? '',
      nu: elastic.nu ?? '',
      yield_stress: plastic.yield_stress ?? '',
    };
  },

  _syncDcaDraftValues() {
    if (!this._directMethods) return;
    const draft = this._directMethods.cfgValues || {};
    this._directMethods.cfgValues = {
      ...draft,
      n_time_points: document.getElementById('dca-n-time-points')?.value ?? draft.n_time_points ?? '33',
      n_harmonics: document.getElementById('dca-n-harmonics')?.value ?? draft.n_harmonics ?? '',
      max_iterations: document.getElementById('dca-max-iterations')?.value ?? draft.max_iterations ?? '100',
      load_factor: document.getElementById('dca-load-factor')?.value ?? draft.load_factor ?? '1.0',
      convergence_tol: document.getElementById('dca-convergence-tol')?.value ?? draft.convergence_tol ?? '',
      cycle_tolerance: document.getElementById('dca-cycle-tolerance')?.value ?? draft.cycle_tolerance ?? '',
      use_b_ext: document.getElementById('dca-use-b-ext')
        ? (document.getElementById('dca-use-b-ext')?.checked ? '1' : '0')
        : (draft.use_b_ext ?? '1'),
      E: document.getElementById('dca-E')?.value ?? draft.E ?? '',
      nu: document.getElementById('dca-nu')?.value ?? draft.nu ?? '',
      yield_stress: document.getElementById('dca-yield-stress')?.value ?? draft.yield_stress ?? '',
    };
  },

  _buildDcaLiveAnalysisInput(inpMeta) {
    if (!inpMeta) return null;

    const draft = this._directMethods?.cfgValues || {};
    const material = inpMeta.materials?.[0] || {};
    const elastic = material.elastic || {};
    const plastic = material.plastic || {};
    const defaultNTimePoints = this._dcaDefaultNTimePoints(inpMeta);
    const defaultNHarmonics = this._dcaDefaultNHarmonics(inpMeta);
    const defaultMaxIterations = this._dcaDefaultMaxIterations(inpMeta);
    const defaultConvergenceTol = this._dcaDefaultConvergenceTol(inpMeta);
    const defaultCycleTolerance = this._dcaDefaultCycleTolerance(inpMeta);
    const defaultUseBExt = this._dcaDefaultUseBExt();

    return {
      n_time_points: this._isElasticBlankValue(draft.n_time_points) ? defaultNTimePoints : draft.n_time_points,
      n_harmonics: this._isElasticBlankValue(draft.n_harmonics) ? defaultNHarmonics : draft.n_harmonics,
      max_iterations: this._isElasticBlankValue(draft.max_iterations) ? defaultMaxIterations : draft.max_iterations,
      load_factor: this._isElasticBlankValue(draft.load_factor) ? '1.0' : draft.load_factor,
      convergence_tol: this._isElasticBlankValue(draft.convergence_tol) ? defaultConvergenceTol : draft.convergence_tol,
      cycle_tolerance: this._isElasticBlankValue(draft.cycle_tolerance) ? defaultCycleTolerance : draft.cycle_tolerance,
      use_b_ext: this._isElasticBlankValue(draft.use_b_ext) ? defaultUseBExt : draft.use_b_ext,
      E: this._isElasticBlankValue(draft.E) ? (elastic.E ?? '') : draft.E,
      nu: this._isElasticBlankValue(draft.nu) ? (elastic.nu ?? '') : draft.nu,
      yield_stress: this._isElasticBlankValue(draft.yield_stress) ? (plastic.yield_stress ?? '') : draft.yield_stress,
      parsed_defaults: {
        n_time_points: defaultNTimePoints,
        n_harmonics: defaultNHarmonics,
        max_iterations: defaultMaxIterations,
        load_factor: '1.0',
        material_E: elastic.E ?? null,
        material_nu: elastic.nu ?? null,
        yield_stress: plastic.yield_stress ?? null,
        convergence_tol: defaultConvergenceTol,
        cycle_tolerance: defaultCycleTolerance,
        use_b_ext: defaultUseBExt,
      },
    };
  },

  _buildDcaDiffItems(analysisInput, inpMeta) {
    if (!analysisInput || !inpMeta) {
      return { diffs: [], controls: [] };
    }

    const parsedDefaults = analysisInput.parsed_defaults || {};
    const diffs = [];
    const controls = [];
    const pushDiff = (label, current, original) => {
      if (this._elasticValuesEqual(current, original)) return;
      diffs.push({
        label,
        current: this._formatElasticValue(current),
        original: this._formatElasticValue(original),
      });
    };
    const pushControlOverride = (label, current, original) => {
      if (this._elasticValuesEqual(current, original)) return;
      controls.push({
        label,
        value: `${this._formatElasticValue(current)} (默认 ${this._formatElasticValue(original)})`,
      });
    };
    const pushToggleOverride = (label, current, original) => {
      if (String(current ?? '') === String(original ?? '')) return;
      controls.push({
        label,
        value: `${String(current ?? '1') !== '0' ? '启用' : '关闭'} (默认 ${String(original ?? '1') !== '0' ? '启用' : '关闭'})`,
      });
    };

    pushDiff('Load factor', analysisInput.load_factor, parsedDefaults.load_factor ?? '1.0');
    pushDiff("Young's modulus (E)", analysisInput.E, parsedDefaults.material_E);
    pushDiff('Poisson ratio (nu)', analysisInput.nu, parsedDefaults.material_nu);
    pushDiff('Yield stress', analysisInput.yield_stress, parsedDefaults.yield_stress);

    pushControlOverride(
      'Time points',
      analysisInput.n_time_points,
      parsedDefaults.n_time_points ?? this._dcaDefaultNTimePoints(inpMeta),
    );
    pushControlOverride(
      'Harmonics',
      analysisInput.n_harmonics,
      parsedDefaults.n_harmonics ?? this._dcaDefaultNHarmonics(inpMeta),
    );
    pushControlOverride(
      'Max iterations',
      analysisInput.max_iterations,
      parsedDefaults.max_iterations ?? this._dcaDefaultMaxIterations(inpMeta),
    );
    pushControlOverride(
      'Convergence tolerance',
      analysisInput.convergence_tol,
      parsedDefaults.convergence_tol ?? this._dcaDefaultConvergenceTol(inpMeta),
    );
    pushControlOverride(
      'Cycle tolerance',
      analysisInput.cycle_tolerance,
      parsedDefaults.cycle_tolerance ?? this._dcaDefaultCycleTolerance(inpMeta),
    );
    pushToggleOverride(
      'B_ext / B-bar',
      analysisInput.use_b_ext,
      parsedDefaults.use_b_ext ?? this._dcaDefaultUseBExt(),
    );
    return {
      diffs,
      controls,
    };
  },

  _onDcaDraftChange() {
    this._syncDcaDraftValues();
    this._renderDcaCurrentDiffPreview();
  },

  _onDcaInpSelectionChange() {
    if (this._directMethods) {
      this._directMethods.dcaSelectedInp = this._selectedDcaInpName();
    }
    this._clearDcaParsedState();
    this._renderDcaParseStatus();
    this._renderDirectMethodsConfigForm();
  },

  _renderDcaParseStatus() {
    const statusEl = document.getElementById('dca-parse-status');
    if (!statusEl) return;
    const inpName = this._selectedDcaInpName();
    const meta = this._directMethods?.parsedInpMetaByName?.[inpName];
    if (!inpName) {
      statusEl.textContent = '请先选择一个 INP。';
      statusEl.style.color = 'var(--text-secondary)';
      return;
    }
    if (!meta || this._currentDcaSelectionKey() !== (this._directMethods?.parsedSelectionKey || '')) {
      statusEl.textContent = '当前所选 INP 尚未解析；提交 DCA 前请先点击“解析所选 INP”。';
      statusEl.style.color = 'var(--text-secondary)';
      return;
    }
    if (meta.supported === false) {
      statusEl.textContent = `该 INP 未检测到可用 DCA step：${meta.error || '解析失败'}`;
      statusEl.style.color = '#fca5a5';
      return;
    }
    const parsedAt = this._directMethods?.lastParsedAt || '刚刚';
    statusEl.textContent = `已按当前所选 INP 完成 fresh parse：${parsedAt}`;
    statusEl.style.color = '#34d399';
  },

  async parseSelectedDcaInp() {
    const model = this._directMethods?.model;
    const inpName = this._selectedDcaInpName();
    if (!model) return;
    if (!inpName) {
      alert('请先选择一个 INP。');
      return;
    }

    const res = await this.apiPost('/api/models/dca-inp-meta', {
      paths: [`${model.path}\\abaqus\\${inpName}`],
    }).catch(() => null);
    const detail = res?.details?.[0] || null;
    if (!detail) {
      alert('DCA INP 解析失败。');
      return;
    }
    this._directMethods.parsedInpMetaByName[inpName] = detail;
    this._directMethods.parsedSelectionKey = this._currentDcaSelectionKey();
    this._directMethods.lastParsedAt = new Date().toLocaleString('zh-CN', { hour12: false });
    this._directMethods.cfgValues = this._buildDcaDraftFromParsed(detail);
    this._renderDcaParseStatus();
    this._renderDirectMethodsConfigForm();
  },

  _renderDcaCurrentDiffPreview() {
    const target = document.getElementById('dca-current-diff-summary');
    if (!target) return;
    this._syncDcaDraftValues();

    const currentKey = this._currentDcaSelectionKey();
    const meta = this._selectedDcaInpMeta();
    if (!currentKey) {
      target.innerHTML = UI.hint('warning', '请先选择一个 INP。');
      return;
    }
    if (!currentKey || currentKey !== (this._directMethods?.parsedSelectionKey || '') || !meta) {
      target.innerHTML = UI.hint('info', '请先对当前所选 INP 完成 fresh parse，再查看提交前差异提醒。');
      return;
    }
    if (meta.supported === false) {
      target.innerHTML = UI.hint('error', meta.error || '当前 INP 解析失败。');
      return;
    }

    const liveInput = this._buildDcaLiveAnalysisInput(meta);
    const summary = this._buildDcaDiffItems(liveInput, meta);
    target.innerHTML = `<div style="font-size:13px;color:var(--text-secondary);margin-top:10px;line-height:1.7">当前提醒只基于这次 fresh parse 得到的临时输入快照；不会拿已有 DCA MAT 或旧 cfg 反推差异，也不会把未改动的 DCA 默认求解控制误报成提醒项。</div>
      <div style="margin-top:12px">${this._renderElasticDiffSummary(summary, {
        currentLabel: '当前表单',
        originalLabel: '解析快照',
        emptyMessage: '当前表单与 fresh parse 快照一致；如果现在提交，这次 DCA 会直接使用解析默认值。',
        controlsCaption: '附加求解控制覆盖项（非 INP 原始字段）',
      })}</div>`;
  },

  _renderDirectMethodsConfigForm() {
    const area = document.getElementById('direct-methods-config-area');
    if (!area) return;

    this._syncDcaDraftValues();

    const model = this._directMethods?.model;
    const inpName = this._selectedDcaInpName();
    const currentKey = this._currentDcaSelectionKey();
    const meta = this._selectedDcaInpMeta();
    if (!model || !inpName) {
      area.innerHTML = '';
      return;
    }
    this._renderDcaParseStatus();

    if (!currentKey || currentKey !== (this._directMethods?.parsedSelectionKey || '') || !meta) {
      area.innerHTML = `<div class="card" style="margin-bottom:16px">${UI.hint('info', '先对当前所选 INP 执行 DCA 解析，再展开周期求解配置。')}</div>`;
      return;
    }
    if (meta.supported === false) {
      area.innerHTML = `<div class="card" style="margin-bottom:16px">${UI.hint('error', `当前 INP 不适合作为 DCA workflow：${meta.error || '解析失败'}`)}</div>`;
      return;
    }
    if (!meta.materials?.length) {
      area.innerHTML = `<div class="card" style="margin-bottom:16px">${UI.hint('error', '当前 INP 的材料信息解析失败，暂时无法在 Web 中生成 DCA 求解表单。')}</div>`;
      return;
    }
    if (meta.family && meta.family !== 'solid') {
      area.innerHTML = `<div class="card" style="margin-bottom:16px">${UI.hint('error', '当前 DCA Web 求解只支持 solid 模型。')}</div>`;
      return;
    }

    const vals = this._directMethods?.cfgValues || {};
    const material = meta.materials?.[0] || {};
    const elastic = material.elastic || {};
    const plastic = material.plastic || {};
    const defaultConvergenceTol = this._dcaDefaultConvergenceTol(meta);
    const defaultCycleTolerance = this._dcaDefaultCycleTolerance(meta);
    const defaultUseBExt = this._dcaDefaultUseBExt();
    const nTimePoints = this._isElasticBlankValue(vals.n_time_points) ? this._dcaDefaultNTimePoints(meta) : vals.n_time_points;
    const nHarmonics = this._isElasticBlankValue(vals.n_harmonics) ? this._dcaDefaultNHarmonics(meta) : vals.n_harmonics;
    const maxIterations = this._isElasticBlankValue(vals.max_iterations) ? this._dcaDefaultMaxIterations(meta) : vals.max_iterations;
    const loadFactor = this._isElasticBlankValue(vals.load_factor) ? '1.0' : vals.load_factor;
    const convergenceTol = this._isElasticBlankValue(vals.convergence_tol) ? defaultConvergenceTol : vals.convergence_tol;
    const cycleTolerance = this._isElasticBlankValue(vals.cycle_tolerance) ? defaultCycleTolerance : vals.cycle_tolerance;
    const eValue = this._isElasticBlankValue(vals.E) ? (elastic.E ?? '') : vals.E;
    const nuValue = this._isElasticBlankValue(vals.nu) ? (elastic.nu ?? '') : vals.nu;
    const yieldValue = this._isElasticBlankValue(vals.yield_stress) ? (plastic.yield_stress ?? '') : vals.yield_stress;
    const useBExtValue = this._isElasticBlankValue(vals.use_b_ext) ? defaultUseBExt : vals.use_b_ext;
    const yieldHint = meta.supports_elastoplastic
      ? `当前 hardening: ${UI.escapeHtml(plastic.hardening || 'perfect')}。`
      : '当前未从 INP 中解析出 plastic 定义时，需要在这里显式填写屈服强度。';

    area.innerHTML = `<div class="card" style="margin-bottom:16px">
      <div class="card-title">DCA 周期求解配置</div>
      <div style="font-size:13px;color:var(--text-secondary);margin-top:10px;line-height:1.7">
        默认值来自刚刚解析的 <b style="color:var(--text-primary)">${UI.escapeHtml(meta.name || inpName)}</b>。DCA 表单现在只保留周期离散、公差、B-bar 开关和材料参数；step、amplitude、时间离散等仍以 fresh parse 结果为准。
      </div>
      <div style="font-size:12px;color:var(--accent);margin-top:8px">最近解析时间：${UI.escapeHtml(this._directMethods?.lastParsedAt || '-')}</div>
      <div style="margin-top:14px;display:flex;flex-direction:column;gap:18px">
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px">
          <div class="form-group" style="margin:0">
            <label class="form-label">Load factor</label>
            <input class="form-input" type="number" step="any" id="dca-load-factor" value="${UI.escapeAttr(String(loadFactor))}" oninput="App._onDcaDraftChange()" onchange="App._onDcaDraftChange()" />
            <div class="form-hint" style="margin-top:4px">按同一个比例放大 INP 中定义的载荷。</div>
          </div>
          <div class="form-group" style="margin:0">
            <label class="form-label">B_ext / B-bar</label>
            <label style="display:flex;align-items:center;gap:10px;min-height:38px">
              <input type="checkbox" id="dca-use-b-ext" ${String(useBExtValue ?? '1') !== '0' ? 'checked' : ''} onchange="App._onDcaDraftChange()" style="width:16px;height:16px;accent-color:var(--accent-blue)">
              <span style="font-size:13px;color:var(--text-secondary)">启用 B_ext 组装路径（默认 1）</span>
            </label>
          </div>
        </div>

        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px">
          <div class="form-group" style="margin:0">
            <label class="form-label">Time points</label>
            <input class="form-input" type="number" min="3" step="1" id="dca-n-time-points" value="${UI.escapeAttr(String(nTimePoints))}" oninput="App._onDcaDraftChange()" onchange="App._onDcaDraftChange()" />
            <div class="form-hint" style="margin-top:4px">周期内输出与扫描的离散帧数；默认来自 DCA step 解析结果，缺省为 33。</div>
          </div>
          <div class="form-group" style="margin:0">
            <label class="form-label">Harmonics</label>
            <input class="form-input" type="number" min="0" step="1" id="dca-n-harmonics" value="${UI.escapeAttr(String(nHarmonics))}" oninput="App._onDcaDraftChange()" onchange="App._onDcaDraftChange()" />
            <div class="form-hint" style="margin-top:4px">控制周期位移与残差在频域中保留的谐波阶数；默认来自 DCA step 解析结果，缺省为 15。</div>
          </div>
          <div class="form-group" style="margin:0">
            <label class="form-label">Max iterations</label>
            <input class="form-input" type="number" min="1" step="1" id="dca-max-iterations" value="${UI.escapeAttr(String(maxIterations))}" oninput="App._onDcaDraftChange()" onchange="App._onDcaDraftChange()" />
            <div class="form-hint" style="margin-top:4px">DCA 外迭代上限；INP 未给出时默认 100。</div>
          </div>
        </div>

        <div class="form-group" style="margin:0">
          <label class="form-label">Convergence tolerance</label>
          <input class="form-input" type="number" step="any" id="dca-convergence-tol" value="${UI.escapeAttr(String(convergenceTol))}" oninput="App._onDcaDraftChange()" onchange="App._onDcaDraftChange()" />
          <div class="form-hint" style="margin-top:4px">控制频域平衡残差的收敛阈值；数值越小，要求越严格，运行时间通常也会更长。</div>
        </div>

        <div class="form-group" style="margin:0">
          <label class="form-label">Cycle tolerance</label>
          <input class="form-input" type="number" step="any" id="dca-cycle-tolerance" value="${UI.escapeAttr(String(cycleTolerance))}" oninput="App._onDcaDraftChange()" onchange="App._onDcaDraftChange()" />
          <div class="form-hint" style="margin-top:4px">控制周期首尾闭合误差的判定阈值；这个值越小，对 steady-state 周期一致性的要求越高。</div>
        </div>

        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px">
          <div class="form-group" style="margin:0">
            <label class="form-label">材料名称</label>
            <input class="form-input" type="text" value="${UI.escapeAttr(material.name || '')}" readonly style="opacity:0.72;cursor:not-allowed" />
          </div>
          <div class="form-group" style="margin:0">
            <label class="form-label">Young's modulus (E)</label>
            <input class="form-input" type="number" step="any" id="dca-E" value="${UI.escapeAttr(String(eValue))}" oninput="App._onDcaDraftChange()" onchange="App._onDcaDraftChange()" />
          </div>
          <div class="form-group" style="margin:0">
            <label class="form-label">Poisson ratio (nu)</label>
            <input class="form-input" type="number" step="any" id="dca-nu" value="${UI.escapeAttr(String(nuValue))}" oninput="App._onDcaDraftChange()" onchange="App._onDcaDraftChange()" />
          </div>
          <div class="form-group" style="margin:0">
            <label class="form-label">Yield stress</label>
            <input class="form-input" type="number" step="any" id="dca-yield-stress" value="${UI.escapeAttr(String(yieldValue))}" oninput="App._onDcaDraftChange()" onchange="App._onDcaDraftChange()" />
            <div class="form-hint" style="margin-top:4px">${yieldHint}</div>
          </div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-bottom:16px">
      <div class="card-title">提交前差异提醒</div>
      <div id="dca-current-diff-summary"></div>
    </div>

    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:4px;margin-bottom:24px">
      <button class="btn btn-primary" onclick="App.saveAndRunDirectCyclic()">基于当前解析结果运行 DCA</button>
    </div>`;

    this._renderDcaCurrentDiffPreview();
  },

  async saveAndRunDirectCyclic() {
    const model = this._directMethods?.model;
    const inpName = this._selectedDcaInpName();
    const meta = this._selectedDcaInpMeta();
    if (!model || !inpName || !meta || meta.supported === false) {
      alert('请先解析一个可用的 DCA INP。');
      return;
    }
    this._syncDcaDraftValues();
    const vals = this._directMethods?.cfgValues || {};

    const modelRoot = this.winToWsl(model.path);
    const inpPath = `${modelRoot}/abaqus/${inpName}`;
    const abaqusReferenceMat = meta.abaqus_reference_mat ? this.winToWsl(meta.abaqus_reference_mat) : '';
    const stepName = meta.step_name;
    const loadFactor = document.getElementById('dca-load-factor')?.value?.trim() || '1.0';
    const run = this._analysisRunPaths(
      model,
      'steady_state_DCA',
      `DCA_${this._sanitizeRunToken(inpName, 'Inp')}_${this._sanitizeRunToken(loadFactor, '1')}LF`,
      'steady_state.template.cfg',
    );
    const nTimePoints = document.getElementById('dca-n-time-points')?.value?.trim() || this._dcaDefaultNTimePoints(meta);
    const nHarmonics = document.getElementById('dca-n-harmonics')?.value?.trim() || this._dcaDefaultNHarmonics(meta);
    const periodicStart = '2';
    const maxIterations = document.getElementById('dca-max-iterations')?.value?.trim() || this._dcaDefaultMaxIterations(meta);
    const defaultConvergenceTol = this._dcaDefaultConvergenceTol(meta);
    const defaultCycleTolerance = this._dcaDefaultCycleTolerance(meta);
    const convergenceTol = document.getElementById('dca-convergence-tol')?.value?.trim() || defaultConvergenceTol;
    const cycleTolerance = document.getElementById('dca-cycle-tolerance')?.value?.trim() || defaultCycleTolerance;
    const constitutiveBackend = 'j2_perfect_plastic';
    const useBExt = String(vals.use_b_ext ?? this._dcaDefaultUseBExt()) !== '0' ? '1' : '0';
    const eOverride = document.getElementById('dca-E')?.value?.trim();
    const nuOverride = document.getElementById('dca-nu')?.value?.trim();
    const yieldOverride = document.getElementById('dca-yield-stress')?.value?.trim();

    const lines = [
      '# Direct Methods steady-state configuration (auto-generated by jaxmech web)',
      `# run_dir: ${run.runLabel}`,
      '',
      'method = dca',
      `inp_file = "${inpPath}"`,
      'inp_files = []',
      `load_factor = ${loadFactor}`,
    ];
    if (eOverride) lines.push(`E_override = ${eOverride}`);
    if (nuOverride) lines.push(`nu_override = ${nuOverride}`);
    if (yieldOverride) lines.push(`yield_stress_override = ${yieldOverride}`);
    if (abaqusReferenceMat) lines.push(`abaqus_reference_mat = "${abaqusReferenceMat}"`);
    lines.push(`step_name = "${stepName}"`);
    lines.push(`n_harmonics = ${nHarmonics}`);
    lines.push(`n_time_points = ${nTimePoints}`);
    lines.push(`periodic_start_iteration = ${periodicStart}`);
    lines.push(`max_iterations = ${maxIterations}`);
    lines.push(`convergence_tol = ${convergenceTol}`);
    lines.push(`cycle_tolerance = ${cycleTolerance}`);
    lines.push('relaxation_factor = 1.0');
    lines.push('');
    lines.push(`constitutive_backend = "${constitutiveBackend}"`);
    lines.push('constitutive_options = {}');
    lines.push(`use_b_ext = ${useBExt}`);
    lines.push('');

    await this.apiPut(`/api/config/direct_methods_steady_state_dca?model_path=${encodeURIComponent(run.runDirWin)}`, {
      content: lines.join('\n'),
    });

    const result = await this.apiPost('/api/run', {
      module: 'jaxmech.modules.direct_methods.run_steady_state',
      args: ['--config', run.cfgWsl],
    });
    this.viewTask(result.id);
  },

  _dmSetActiveWorkflow(workflow) {
    if (!this._directMethods) return;
    this._directMethods.activeWorkflow = workflow;
    this._renderDirectMethodsShell();
  },

  _dmSetSteadyMethod(method) {
    if (!this._directMethods) return;
    this._directMethods.steadyMethod = method;
    this._renderDirectMethodsShell();
  },

  _dmSelectedInpNames(groupName) {
    return Array.from(document.querySelectorAll(`input[name="${groupName}"]:checked`))
      .map(el => el.value?.trim())
      .filter(Boolean);
  },

  _dmInpNamesFromCfgValue(value, availableInps) {
    const available = new Set(availableInps || []);
    if (!available.size) return [];
    const text = Array.isArray(value) ? value.join('\n') : String(value ?? '');
    const matches = text.match(/[^"'\[\],\s]+\.inp/gi) || [];
    const names = [];
    matches.forEach(item => {
      const name = String(item).split(/[\\/]/).pop();
      if (available.has(name) && !names.includes(name)) names.push(name);
    });
    return names.slice(0, 2);
  },

  _dmRsdmState(kind) {
    if (!this._directMethods) return null;
    this._directMethods.rsdm ??= {};
    this._directMethods.rsdm[kind] ??= {
      parsedSelectionKey: '',
      selectedNames: [],
      selectionTouched: false,
      lastParsedAt: '',
      metaDetails: [],
      cfgValues: {},
      warnings: [],
      errors: [],
    };
    this._directMethods.rsdm[kind].selectedNames ??= [];
    this._directMethods.rsdm[kind].selectionTouched ??= false;
    return this._directMethods.rsdm[kind];
  },

  _dmNormalizeRsdmCfgValues(values) {
    const out = { ...(values || {}) };
    ['method', 'time_grid_type', 'inner_iteration_cap_mode', 'use_angles'].forEach(key => {
      if (typeof out[key] === 'string') {
        out[key] = out[key].trim().replace(/^["']|["']$/g, '');
      }
    });
    const copyAlias = (from, to) => {
      if (!this._isElasticBlankValue(out[from]) && this._isElasticBlankValue(out[to])) {
        out[to] = out[from];
      }
    };
    copyAlias('yield_stress_override', 'yield_stress');
    copyAlias('E_override', 'E');
    copyAlias('nu_override', 'nu');
    copyAlias('n_vertices', 'n_vert');
    return out;
  },

  _dmRsdmGroupName(kind) {
    return `rsdm-${kind}-inp`;
  },

  _dmCurrentRsdmSelectionKey(kind) {
    return this._dmSelectedInpNames(this._dmRsdmGroupName(kind)).join('|');
  },

  _dmMaterialFromMeta(meta) {
    return meta?.materials?.[0] || {};
  },

  _dmElasticFromMeta(meta) {
    return this._dmMaterialFromMeta(meta).elastic || {};
  },

  _dmYieldFromMeta(meta) {
    const value = this._dmMaterialFromMeta(meta).plastic?.yield_stress;
    return this._isElasticBlankValue(value) ? null : Number(value);
  },

  _dmBuildRsdmDefaults(kind, details) {
    const first = details?.[0] || {};
    const elastic = this._dmElasticFromMeta(first);
    const yields = (details || []).map(meta => this._dmYieldFromMeta(meta));
    const firstYield = yields.find(v => Number.isFinite(v));
    const selectedCount = (details || []).length;
    const isShakedown = kind === 'shakedown';
    return {
      load_factor: '1.0',
      yield_stress: Number.isFinite(firstYield) ? String(firstYield) : '',
      E: this._isElasticBlankValue(elastic.E) ? '' : String(elastic.E),
      nu: this._isElasticBlankValue(elastic.nu) ? '' : String(elastic.nu),
      time_grid_type: 'gauss_legendre',
      n_time_points: '3',
      n_fourier_terms: '3',
      inner_tol: '1.0e-2',
      classification_rel_tol: '1.0e-2',
      max_inner_iterations: '100',
      inner_iteration_cap_mode: 'fixed',
      adaptive_inner_min_iterations: '100',
      adaptive_inner_rel_high: '0.20',
      adaptive_inner_rel_low: '0.01',
      chi: '3.0',
      outer_tol: '1.0e-2',
      max_outer_iterations: '30',
      extract_residual_field: '1',
      residual_field_max_iterations: '500',
      residual_field_constant_rel_tol: '1.0e-8',
      n_vert: '2',
      angle_deg: '0.0',
      use_angles: isShakedown ? 'yes' : 'no',
      theta_deg: '0',
      R_ratios: Array(Math.max(1, selectedCount)).fill('0.0').join(', '),
      load_factor_set: selectedCount >= 2 ? '1.0, 1.0' : '',
      fix_load2: '0',
      use_b_ext: '1',
      method: isShakedown ? 'rsdm_s' : 'rsdm',
    };
  },

  _dmExtractNumbers(value) {
    if (Array.isArray(value)) {
      return value.flat(Infinity).map(Number).filter(Number.isFinite);
    }
    const text = String(value ?? '').trim();
    if (!text) return [];
    const matches = text.match(/[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g) || [];
    return matches.map(Number).filter(Number.isFinite);
  },

  _dmRsdmThetaValues(vals) {
    const fromTheta = this._dmExtractNumbers(vals?.theta_deg);
    if (fromTheta.length) return fromTheta;
    const angle = Number(vals?.angle_deg);
    return [Number.isFinite(angle) ? angle : 0];
  },

  _dmRsdmLfsRows(vals, colCount) {
    const nCols = Math.max(1, Number(colCount) || 1);
    const numbers = this._dmExtractNumbers(vals?.load_factor_set);
    const rows = [];
    for (let index = 0; index + nCols <= numbers.length; index += nCols) {
      rows.push(numbers.slice(index, index + nCols));
    }
    return rows.length ? rows : [Array(nCols).fill(1.0)];
  },

  _dmRsdmRRatios(vals, colCount) {
    const nCols = Math.max(1, Number(colCount) || 1);
    const numbers = this._dmExtractNumbers(vals?.R_ratios);
    const out = [];
    for (let i = 0; i < nCols; i += 1) {
      out.push(Number.isFinite(numbers[i]) ? numbers[i] : 0.0);
    }
    return out;
  },

  _dmRsdmLoadMode(vals) {
    const raw = String(vals?.use_angles ?? 'yes').trim().toLowerCase();
    return ['0', 'false', 'no', 'n', 'off'].includes(raw) ? 'no' : 'yes';
  },

  _dmBoolFlag(value, fallback = false) {
    if (value === undefined || value === null || value === '') return !!fallback;
    if (typeof value === 'boolean') return value;
    const raw = String(value).trim().toLowerCase();
    if (['0', 'false', 'no', 'n', 'off'].includes(raw)) return false;
    if (['1', 'true', 'yes', 'y', 'on'].includes(raw)) return true;
    return !!fallback;
  },

  _dmRsdmNVertOptions(selectedCount) {
    return Number(selectedCount) >= 2 ? ['2', '4'] : ['2'];
  },

  _dmRsdmCoerceNVert(value, selectedCount) {
    const options = this._dmRsdmNVertOptions(selectedCount);
    const text = String(value ?? '').trim();
    if (options.includes(text)) return text;
    return '2';
  },

  _collectRsdmLoadCombination(kind, colCount) {
    const prefix = `rsdm-${kind}`;
    const nCols = Math.max(1, Number(colCount) || 1);
    if (nCols <= 1) {
      return {
        use_angles: 'yes',
        theta_deg: [],
        load_factor_set: [],
        angle_deg: 0,
      };
    }

    const useAngles = (document.getElementById(`${prefix}-use-angles`)?.value || 'yes') !== 'no';
    if (useAngles) {
      const thetas = Array.from(document.querySelectorAll(`#${prefix}-theta-list .rsdm-theta-item`))
        .map(input => Number(input.value))
        .filter(Number.isFinite);
      const values = thetas.length ? thetas : [0];
      return {
        use_angles: 'yes',
        theta_deg: values,
        load_factor_set: [],
        angle_deg: values[0],
      };
    }

    const rows = Array.from(document.querySelectorAll(`#${prefix}-lfs-body .rsdm-lfs-row`)).map(row => {
      const values = Array.from(row.querySelectorAll('.rsdm-lfs-cell'))
        .map(input => Number(input.value))
        .map(value => Number.isFinite(value) ? value : 0);
      while (values.length < nCols) values.push(0);
      return values.slice(0, nCols);
    }).filter(row => row.some(value => Math.abs(value) > 0));
    const lfsRows = rows.length ? rows : [Array(nCols).fill(1.0)];
    const first = lfsRows[0] || [1.0, 0.0];
    const angleDeg = nCols >= 2
      ? Math.atan2(Number(first[1]) || 0, Number(first[0]) || 0) * 180 / Math.PI
      : 0;
    return {
      use_angles: 'no',
      theta_deg: [],
      load_factor_set: lfsRows,
      angle_deg: Number.isFinite(angleDeg) ? angleDeg : 0,
    };
  },

  _onRsdmLoadCombinationChange(kind) {
    this._syncRsdmDraftValues(kind);
    const prefix = `rsdm-${kind}`;
    const useAngles = (document.getElementById(`${prefix}-use-angles`)?.value || 'yes') !== 'no';
    const anglesSection = document.getElementById(`${prefix}-angles-section`);
    const lfsSection = document.getElementById(`${prefix}-lfs-section`);
    if (anglesSection) {
      anglesSection.style.display = useAngles ? 'flex' : 'none';
      anglesSection.style.flexDirection = 'column';
    }
    if (lfsSection) {
      lfsSection.style.display = useAngles ? 'none' : 'flex';
      lfsSection.style.flexDirection = 'column';
    }
    this._refreshRsdmDiffSummary(kind);
  },

  _addRsdmAngle(kind) {
    const prefix = `rsdm-${kind}`;
    const list = document.getElementById(`${prefix}-theta-list`);
    if (!list) return;
    const count = list.querySelectorAll('.rsdm-theta-item').length + 1;
    const wrapper = document.createElement('div');
    wrapper.className = 'rsdm-angle-row';
    wrapper.style.cssText = 'display:flex;align-items:center;gap:6px';
    wrapper.innerHTML = `<span style="font-size:11px;color:var(--text-secondary);min-width:22px">${count}.</span>
      <input class="form-input rsdm-theta-item" type="number" step="any" value="0" style="width:96px" placeholder="deg" oninput="App._onRsdmDraftChange('${kind}')" onchange="App._onRsdmDraftChange('${kind}')" />
      <button class="btn btn-secondary" style="padding:0 8px;height:30px;font-size:13px" onclick="this.closest('.rsdm-angle-row').remove();App._onRsdmDraftChange('${kind}')">-</button>`;
    list.appendChild(wrapper);
    this._onRsdmDraftChange(kind);
  },

  _addRsdmLfsRow(kind, colCount) {
    const prefix = `rsdm-${kind}`;
    const tbody = document.getElementById(`${prefix}-lfs-body`);
    if (!tbody) return;
    const nCols = Math.max(1, Number(colCount) || 1);
    const tr = document.createElement('tr');
    tr.className = 'rsdm-lfs-row';
    tr.innerHTML = Array(nCols).fill(0).map(() =>
      `<td style="padding:4px 6px"><input type="number" step="any" class="form-input rsdm-lfs-cell" style="width:100%;padding:6px 8px" value="0.0" oninput="App._onRsdmDraftChange('${kind}')" onchange="App._onRsdmDraftChange('${kind}')" /></td>`
    ).join('') + `<td style="padding:4px 6px"><button class="btn btn-secondary" style="padding:0 8px;height:30px;font-size:13px" onclick="this.closest('tr').remove();App._onRsdmDraftChange('${kind}')">-</button></td>`;
    tbody.appendChild(tr);
    this._onRsdmDraftChange(kind);
  },

  _normalizeRsdmLfs(kind) {
    const prefix = `rsdm-${kind}`;
    const rows = document.querySelectorAll(`#${prefix}-lfs-body .rsdm-lfs-row`);
    rows.forEach(row => {
      const cells = row.querySelectorAll('.rsdm-lfs-cell');
      const vals = Array.from(cells).map(cell => Number(cell.value) || 0);
      const norm = Math.sqrt(vals.reduce((sum, value) => sum + value * value, 0));
      if (norm > 1e-12) {
        cells.forEach((cell, index) => {
          cell.value = (vals[index] / norm).toFixed(8);
        });
      }
    });
    this._onRsdmDraftChange(kind);
  },

  _applyRsdmFixLoad2Constraint(kind) {
    const state = this._dmRsdmState(kind);
    if (!state) return false;
    const prefix = `rsdm-${kind}`;
    const selectedCount = this._dmSelectedInpNames(this._dmRsdmGroupName(kind)).length
      || (state.parsedSelectionKey || '').split('|').filter(Boolean).length;
    const nVert = this._dmRsdmCoerceNVert(
      document.getElementById(`${prefix}-n-vert`)?.value ?? state.cfgValues?.n_vert ?? '2',
      selectedCount
    );
    const fixActive = selectedCount === 2
      && nVert === '2'
      && !!document.getElementById(`${prefix}-fix-load2`)?.checked;
    const r2Input = document.getElementById(`${prefix}-r-ratio-1`);
    if (r2Input) {
      if (fixActive) {
        r2Input.value = '1';
        r2Input.disabled = true;
        r2Input.title = 'Fix load2 启用时，R2 固定为 1；Load2 factor 仍作为定载幅值倍率生效。';
      } else {
        r2Input.disabled = false;
        r2Input.title = '';
      }
    }
    return fixActive;
  },

  _onRsdmFixLoad2Change(kind) {
    this._applyRsdmFixLoad2Constraint(kind);
    this._onRsdmDraftChange(kind);
  },

  _dmAnalyzeRsdmMeta(details, yieldOverride = null) {
    const warnings = [];
    const errors = [];
    if (!details?.length) {
      errors.push('尚未解析 INP。');
      return { warnings, errors };
    }
    const first = details[0];
    const firstElastic = this._dmElasticFromMeta(first);
    const firstYield = this._dmYieldFromMeta(first);
    const hasYieldOverride = !this._isElasticBlankValue(yieldOverride) && Number.isFinite(Number(yieldOverride));
    if (!Number.isFinite(firstYield) && !hasYieldOverride) {
      warnings.push('所选 INP 未解析到 yield；提交前必须填写 Yield stress。');
    }
    details.forEach((meta, index) => {
      if (meta.error || meta.supported === false) {
        errors.push(`${meta.name || `INP ${index + 1}`} 解析失败：${meta.error || meta.support_note || 'unsupported'}`);
      }
      if (first.family && meta.family && meta.family !== first.family) {
        errors.push(`${meta.name} 的 family=${meta.family} 与第一个 INP 的 ${first.family} 不一致。`);
      }
      const elastic = this._dmElasticFromMeta(meta);
      if (!this._elasticValuesEqual(elastic.E, firstElastic.E) || !this._elasticValuesEqual(elastic.nu, firstElastic.nu)) {
        errors.push(`${meta.name} 的 elastic material 与第一个 INP 不一致。`);
      }
      const y = this._dmYieldFromMeta(meta);
      if (Number.isFinite(firstYield) && Number.isFinite(y) && !this._elasticValuesEqual(y, firstYield)) {
        warnings.push(`${meta.name} 的 yield=${y} 与第一个 INP 的 ${firstYield} 不同；默认采用第一个，可手动 override。`);
      }
    });
    if (details.length > 2) {
      errors.push('当前 RSDM runner 只支持 1 或 2 个 elastic load-case INP。');
    }
    return { warnings, errors };
  },

  _syncRsdmDraftValues(kind) {
    const state = this._dmRsdmState(kind);
    if (!state) return;
    const prefix = `rsdm-${kind}`;
    const draft = state.cfgValues || {};
    const val = (name, fallback = '') => document.getElementById(`${prefix}-${name}`)?.value ?? fallback;
    const selectedCount = (state.parsedSelectionKey || '').split('|').filter(Boolean).length;
    const nVert = this._dmRsdmCoerceNVert(val('n-vert', draft.n_vert ?? '2'), selectedCount);
    const canUseFixLoad2 = selectedCount === 2 && nVert === '2';
    const fixLoad2 = canUseFixLoad2 && document.getElementById(`${prefix}-fix-load2`)
      ? (document.getElementById(`${prefix}-fix-load2`)?.checked ? '1' : '0')
      : '0';
    if (fixLoad2 === '1') {
      this._applyRsdmFixLoad2Constraint(kind);
    }
    const combo = selectedCount >= 2 && document.getElementById(`${prefix}-use-angles`)
      ? this._collectRsdmLoadCombination(kind, selectedCount)
      : {
          use_angles: draft.use_angles ?? 'yes',
          theta_deg: this._dmRsdmThetaValues(draft),
          load_factor_set: this._dmRsdmLfsRows(draft, Math.max(1, selectedCount)),
          angle_deg: Number(draft.angle_deg ?? 0) || 0,
        };
    const rRatioValues = Array.from(document.querySelectorAll(`#${prefix}-r-ratios .rsdm-r-ratio-item`))
      .map((input, index) => {
        if (selectedCount === 2 && fixLoad2 === '1' && index === 1) {
          input.value = '1';
          return '1';
        }
        return input.value;
      });
    state.cfgValues = {
      ...draft,
      load_factor: kind === 'shakedown' ? (draft.load_factor ?? '1.0') : val('load-factor', draft.load_factor ?? '1.0'),
      yield_stress: val('yield-stress', draft.yield_stress ?? ''),
      E: val('E', draft.E ?? ''),
      nu: val('nu', draft.nu ?? ''),
      time_grid_type: val('time-grid-type', draft.time_grid_type ?? 'gauss_legendre'),
      n_time_points: val('n-time-points', draft.n_time_points ?? '3'),
      n_fourier_terms: val('n-fourier-terms', draft.n_fourier_terms ?? '3'),
      inner_tol: val('inner-tol', draft.inner_tol ?? '1.0e-2'),
      classification_rel_tol: val('classification-rel-tol', draft.classification_rel_tol ?? '1.0e-2'),
      max_inner_iterations: val('max-inner-iterations', draft.max_inner_iterations ?? '100'),
      inner_iteration_cap_mode: val('inner-cap-mode', draft.inner_iteration_cap_mode ?? 'fixed'),
      adaptive_inner_min_iterations: val('adaptive-min-iterations', draft.adaptive_inner_min_iterations ?? '100'),
      adaptive_inner_rel_high: val('adaptive-rel-high', draft.adaptive_inner_rel_high ?? '0.20'),
      adaptive_inner_rel_low: val('adaptive-rel-low', draft.adaptive_inner_rel_low ?? '0.01'),
      chi: val('chi', draft.chi ?? '3.0'),
      outer_tol: val('outer-tol', draft.outer_tol ?? '1.0e-2'),
      max_outer_iterations: val('max-outer-iterations', draft.max_outer_iterations ?? '30'),
      extract_residual_field: document.getElementById(`${prefix}-extract-field`)
        ? (document.getElementById(`${prefix}-extract-field`)?.checked ? '1' : '0')
        : (draft.extract_residual_field ?? '1'),
      residual_field_max_iterations: val('residual-field-max-iterations', draft.residual_field_max_iterations ?? '500'),
      residual_field_constant_rel_tol: val('residual-field-rel-tol', draft.residual_field_constant_rel_tol ?? '1.0e-8'),
      n_vert: nVert,
      angle_deg: String(combo.angle_deg ?? '0.0'),
      use_angles: combo.use_angles,
      theta_deg: (combo.theta_deg || []).join(', '),
      R_ratios: rRatioValues.join(', ') || (draft.R_ratios ?? '0.0'),
      load_factor_set: (combo.load_factor_set || []).map(row => row.join(', ')).join('; '),
      fix_load2: fixLoad2,
      use_b_ext: document.getElementById(`${prefix}-use-b-ext`)
        ? (document.getElementById(`${prefix}-use-b-ext`)?.checked ? '1' : '0')
        : (draft.use_b_ext ?? '1'),
    };
  },

  _onRsdmDraftChange(kind) {
    this._syncRsdmDraftValues(kind);
    this._renderRsdmParseStatus(kind);
    this._refreshRsdmDiffSummary(kind);
  },

  _onRsdmNVertChange(kind) {
    this._syncRsdmDraftValues(kind);
    const area = document.getElementById(`rsdm-${kind}-config-area`);
    if (area) {
      area.innerHTML = this._renderRsdmForm(kind);
      this._typesetMath(area);
    }
    this._renderRsdmParseStatus(kind);
  },

  _onRsdmInpSelectionChange(kind) {
    const state = this._dmRsdmState(kind);
    if (!state) return;
    const group = this._dmRsdmGroupName(kind);
    let selected = this._dmSelectedInpNames(group);
    if (selected.length > 2) {
      const keep = selected.slice(0, 2);
      document.querySelectorAll(`input[name="${group}"]`).forEach(input => {
        input.checked = keep.includes(input.value?.trim());
      });
      selected = keep;
      alert('当前 RSDM runner 只支持 1 或 2 个 elastic load-case INP。');
    }
    state.selectionTouched = true;
    state.selectedNames = selected;
    state.parsedSelectionKey = '';
    state.metaDetails = [];
    state.warnings = [];
    state.errors = [];
    state.lastParsedAt = '';
    this._renderDirectMethodsShell();
  },

  async parseSelectedRsdmInp(kind) {
    const model = this._directMethods?.model;
    const names = this._dmSelectedInpNames(this._dmRsdmGroupName(kind));
    if (!model) return;
    if (!names.length) {
      alert('请至少选择一个 elastic load-case INP。');
      return;
    }
    if (names.length > 2) {
      alert('当前 RSDM runner 只支持 1 或 2 个 elastic load-case INP。');
      return;
    }
    const res = await this.apiPost('/api/models/inp-meta', {
      paths: names.map(name => `${model.path}\\abaqus\\${name}`),
    }).catch(() => null);
    const details = res?.details || [];
    if (!details.length) {
      alert('RSDM INP metadata 解析失败。');
      return;
    }
    const state = this._dmRsdmState(kind);
    const previousValues = this._dmNormalizeRsdmCfgValues(state.cfgValues || {});
    const checks = this._dmAnalyzeRsdmMeta(details, previousValues.yield_stress);
    state.selectionTouched = true;
    state.selectedNames = names;
    state.metaDetails = details;
    state.warnings = checks.warnings;
    state.errors = checks.errors;
    state.parsedSelectionKey = names.join('|');
    state.lastParsedAt = new Date().toLocaleString('zh-CN', { hour12: false });
    state.cfgValues = this._dmNormalizeRsdmCfgValues({
      ...this._dmBuildRsdmDefaults(kind, details),
      ...Object.fromEntries(
        Object.entries(previousValues).filter(([, value]) => !this._isElasticBlankValue(value))
      ),
    });
    this._renderDirectMethodsShell();
  },

  _renderRsdmParseStatus(kind) {
    const statusEl = document.getElementById(`rsdm-${kind}-parse-status`);
    if (!statusEl) return;
    const state = this._dmRsdmState(kind);
    const selectedKey = this._dmCurrentRsdmSelectionKey(kind);
    if (!selectedKey) {
      statusEl.textContent = '请至少选择一个 elastic load-case INP。';
      statusEl.style.color = 'var(--text-secondary)';
      return;
    }
    if (!state || state.parsedSelectionKey !== selectedKey) {
      statusEl.textContent = '当前所选 INP 尚未解析；提交前请先解析。';
      statusEl.style.color = 'var(--text-secondary)';
      return;
    }
    if (state.errors?.length) {
      statusEl.textContent = `轻量检查发现 ${state.errors.length} 个问题；runner 还会做最终一致性检查。`;
      statusEl.style.color = '#fca5a5';
      return;
    }
    statusEl.textContent = `已解析 ${state.metaDetails.length} 个 INP：${state.lastParsedAt}`;
    statusEl.style.color = '#34d399';
  },

  _dmFormatRsdmDiffValue(value) {
    if (Array.isArray(value)) {
      return value.map(item => Array.isArray(item) ? `[${item.join(', ')}]` : String(item)).join('; ');
    }
    return this._formatElasticValue(value);
  },

  _dmRsdmDiffEqual(current, original) {
    const currentNums = this._dmExtractNumbers(current);
    const originalNums = this._dmExtractNumbers(original);
    if (currentNums.length || originalNums.length) {
      if (currentNums.length !== originalNums.length) return false;
      return currentNums.every((value, index) => this._elasticValuesEqual(value, originalNums[index]));
    }
    return String(current ?? '').trim() === String(original ?? '').trim();
  },

  _buildRsdmDiffItems(kind) {
    const state = this._dmRsdmState(kind);
    if (!state?.metaDetails?.length) {
      return { diffs: [], controls: [] };
    }
    const isShakedown = kind === 'shakedown';
    const selectedCount = (state.parsedSelectionKey || '').split('|').filter(Boolean).length;
    const defaults = this._dmNormalizeRsdmCfgValues(this._dmBuildRsdmDefaults(kind, state.metaDetails));
    const current = this._dmNormalizeRsdmCfgValues(state.cfgValues || {});
    const diffs = [];
    const controls = [];
    const push = (target, label, key, formatter = value => value) => {
      const currentValue = formatter(current[key]);
      const defaultValue = formatter(defaults[key]);
      if (this._dmRsdmDiffEqual(currentValue, defaultValue)) return;
      target.push({
        label,
        current: this._dmFormatRsdmDiffValue(currentValue),
        original: this._dmFormatRsdmDiffValue(defaultValue),
      });
    };
    const pushBool = (target, label, key) => {
      const currentValue = this._dmBoolFlag(current[key], this._dmBoolFlag(defaults[key], false)) ? '启用' : '关闭';
      const defaultValue = this._dmBoolFlag(defaults[key], false) ? '启用' : '关闭';
      if (currentValue === defaultValue) return;
      target.push({ label, current: currentValue, original: defaultValue });
    };
    const pushControl = (label, key) => {
      const currentValue = current[key];
      const defaultValue = defaults[key];
      if (this._dmRsdmDiffEqual(currentValue, defaultValue)) return;
      controls.push({
        label,
        value: `${this._dmFormatRsdmDiffValue(currentValue)} (默认 ${this._dmFormatRsdmDiffValue(defaultValue)})`,
      });
    };
    const pushControlBool = (label, key, fallback = false) => {
      const currentValue = this._dmBoolFlag(current[key], fallback) ? '启用' : '关闭';
      const defaultValue = this._dmBoolFlag(defaults[key], fallback) ? '启用' : '关闭';
      if (currentValue === defaultValue) return;
      controls.push({ label, value: `${currentValue} (默认 ${defaultValue})` });
    };

    if (!isShakedown) push(diffs, 'Global load factor', 'load_factor');
    push(diffs, 'Yield stress', 'yield_stress');
    push(diffs, "Young's modulus (E)", 'E');
    push(diffs, 'Poisson ratio (nu)', 'nu');
    push(diffs, 'Vertex count', 'n_vert');
    push(diffs, 'Stress ratios R_i', 'R_ratios');
    if (selectedCount >= 2) {
      push(diffs, 'Load combination mode', 'use_angles');
      push(diffs, 'theta_deg', 'theta_deg');
      push(diffs, 'load_factor_set', 'load_factor_set');
      pushBool(diffs, 'Fix load2', 'fix_load2');
    }

    pushControl('Time grid', 'time_grid_type');
    pushControl('Time points', 'n_time_points');
    pushControl('Fourier terms', 'n_fourier_terms');
    pushControl('Inner tol', 'inner_tol');
    pushControl('Classification tol', 'classification_rel_tol');
    pushControl('Max inner iterations', 'max_inner_iterations');
    pushControlBool('B_ext / B-bar', 'use_b_ext', true);
    pushControlBool('Extract residual field', 'extract_residual_field', true);
    if (isShakedown) {
      pushControl('Inner cap mode', 'inner_iteration_cap_mode');
      pushControl('Adaptive min iterations', 'adaptive_inner_min_iterations');
      pushControl('Adaptive rel high', 'adaptive_inner_rel_high');
      pushControl('Adaptive rel low', 'adaptive_inner_rel_low');
      pushControl('Chi', 'chi');
      pushControl('Outer tol', 'outer_tol');
      pushControl('Max outer iterations', 'max_outer_iterations');
      pushControl('Residual field max iterations', 'residual_field_max_iterations');
      pushControl('Residual field constant rel tol', 'residual_field_constant_rel_tol');
    }
    return { diffs, controls };
  },

  _renderRsdmDiffSummary(kind) {
    const state = this._dmRsdmState(kind);
    const selectedKey = this._dmCurrentRsdmSelectionKey(kind);
    if (!selectedKey) {
      return UI.hint('warning', '请先选择至少一个 elastic load-case INP。');
    }
    if (!state?.metaDetails?.length || state.parsedSelectionKey !== selectedKey) {
      return UI.hint('info', '请先对当前所选 INP 完成 fresh parse，再查看提交前差异提醒。');
    }
    const summary = this._buildRsdmDiffItems(kind);
    return `<div style="font-size:13px;color:var(--text-secondary);margin-top:10px;line-height:1.7">当前提醒只基于这次 fresh parse 得到的 RSDM 默认配置；如果现在提交，下面列出的差异会写入 cfg 并进入 runner。</div>
      <div style="margin-top:12px">${this._renderElasticDiffSummary(summary, {
        currentLabel: '当前表单',
        originalLabel: '解析默认值',
        emptyMessage: '当前表单与 RSDM 默认配置一致；如果现在提交，不会写入额外覆盖项。',
        controlsCaption: '附加求解控制覆盖项',
      })}</div>`;
  },

  _refreshRsdmDiffSummary(kind) {
    const target = document.getElementById(`rsdm-${kind}-current-diff-summary`);
    if (!target) return;
    target.innerHTML = this._renderRsdmDiffSummary(kind);
  },

  _renderRsdmMessages(state) {
    const rows = [];
    (state?.errors || []).forEach(msg => rows.push(UI.hint('error', msg)));
    (state?.warnings || []).forEach(msg => rows.push(UI.hint('warning', msg)));
    if (!rows.length && (state?.metaDetails?.length || 0) > 1) {
      rows.push(UI.hint('info', '轻量检查已确认 family 与 elastic material 一致；mesh、boundary condition 与 free DOF 仍由 runner 做最终检查。'));
    }
    return rows.length ? `<div style="display:flex;flex-direction:column;gap:8px;margin-top:12px">${rows.join('')}</div>` : '';
  },

  _renderRsdmForm(kind) {
    const state = this._dmRsdmState(kind);
    const selectedKey = this._dmCurrentRsdmSelectionKey(kind)
      || (state?.selectedNames || []).join('|')
      || (state?.parsedSelectionKey || '');
    if (!state || state.parsedSelectionKey !== selectedKey || !state.metaDetails.length) {
      return `<div class="card" style="margin-bottom:16px">${UI.hint('info', '先解析当前所选 elastic load-case INP，再展开 RSDM 配置。')}</div>`;
    }
    const vals = state.cfgValues || {};
    const prefix = `rsdm-${kind}`;
    const isShakedown = kind === 'shakedown';
    const onChange = `App._onRsdmDraftChange('${kind}')`;
    const selectedNames = selectedKey.split('|').filter(Boolean);
    const selectedCount = selectedNames.length;
    const useBExtValue = this._dmBoolFlag(vals.use_b_ext, true);
    const loadFactorField = isShakedown ? '' : `<div class="form-group">
        <label class="form-label">${selectedCount >= 2 ? 'Global load factor' : 'Load factor'}</label>
        <input class="form-input" type="number" step="any" id="${prefix}-load-factor" value="${UI.escapeAttr(vals.load_factor ?? '1.0')}" oninput="${onChange}" onchange="${onChange}" />
        <div class="form-hint">${selectedCount >= 2 ? '整体放大 Load1/Load2 组合向量；单独倍率在 load_factor_set 中设置。' : '固定比例放大 INP 中定义的载荷。'}</div>
      </div>`;

    const commonFields = `<div class="dm-field-grid">
      ${loadFactorField}
      <div class="form-group">
        <label class="form-label">Yield stress</label>
        <input class="form-input" type="number" step="any" id="${prefix}-yield-stress" value="${UI.escapeAttr(vals.yield_stress ?? '')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
      <div class="form-group">
        <label class="form-label">Young's modulus override</label>
        <input class="form-input" type="number" step="any" id="${prefix}-E" value="${UI.escapeAttr(vals.E ?? '')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
      <div class="form-group">
        <label class="form-label">Poisson ratio override</label>
        <input class="form-input" type="number" step="any" id="${prefix}-nu" value="${UI.escapeAttr(vals.nu ?? '')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
      <div class="form-group">
        <label class="form-label">B_ext / B-bar</label>
        <label style="display:flex;align-items:center;gap:10px;min-height:38px">
          <input type="checkbox" id="${prefix}-use-b-ext" ${useBExtValue ? 'checked' : ''} onchange="${onChange}" style="width:16px;height:16px;accent-color:var(--accent-blue)">
          <span style="font-size:13px;color:var(--text-secondary)">启用 B_ext 组装路径（默认 1）</span>
        </label>
      </div>
    </div>`;

    const coreFields = `<div class="dm-field-grid">
      <div class="form-group">
        <label class="form-label">Time grid</label>
        <select class="form-input" id="${prefix}-time-grid-type" onchange="${onChange}">
          <option value="gauss_legendre" ${vals.time_grid_type === 'gauss_legendre' ? 'selected' : ''}>gauss_legendre</option>
          <option value="uniform" ${vals.time_grid_type === 'uniform' ? 'selected' : ''}>uniform</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Time points</label>
        <input class="form-input" type="number" min="3" step="1" id="${prefix}-n-time-points" value="${UI.escapeAttr(vals.n_time_points ?? '3')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
      <div class="form-group">
        <label class="form-label">Fourier terms</label>
        <input class="form-input" type="number" min="1" step="1" id="${prefix}-n-fourier-terms" value="${UI.escapeAttr(vals.n_fourier_terms ?? '3')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
      <div class="form-group">
        <label class="form-label">Inner tol</label>
        <input class="form-input" type="number" step="any" id="${prefix}-inner-tol" value="${UI.escapeAttr(vals.inner_tol ?? '1.0e-2')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
      <div class="form-group">
        <label class="form-label">Classification tol</label>
        <input class="form-input" type="number" step="any" id="${prefix}-classification-rel-tol" value="${UI.escapeAttr(vals.classification_rel_tol ?? '1.0e-2')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
      <div class="form-group">
        <label class="form-label">Max inner iterations</label>
        <input class="form-input" type="number" min="1" step="1" id="${prefix}-max-inner-iterations" value="${UI.escapeAttr(vals.max_inner_iterations ?? '100')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
    </div>`;

    const nVertValue = this._dmRsdmCoerceNVert(vals.n_vert ?? '2', selectedCount);
    const nVertOptions = this._dmRsdmNVertOptions(selectedCount).map(value =>
      `<option value="${value}" ${nVertValue === value ? 'selected' : ''}>${value}</option>`
    ).join('');
    const fixLoad2Field = selectedCount === 2 && nVertValue === '2' ? `<div class="form-group">
        <label class="form-label">Fix load2</label>
        <label style="display:flex;align-items:center;gap:10px;min-height:38px">
          <input type="checkbox" id="${prefix}-fix-load2" ${this._dmBoolFlag(vals.fix_load2, false) ? 'checked' : ''} onchange="App._onRsdmFixLoad2Change('${kind}')" style="width:16px;height:16px;accent-color:var(--accent-blue)">
          <span style="font-size:13px;color:var(--text-secondary)">固定第二个 load case 的时间路径；Load2 factor 仍作为定载倍率生效</span>
        </label>
      </div>` : '';
    const loadMode = this._dmRsdmLoadMode(vals);
    const thetaValues = this._dmRsdmThetaValues(vals);
    const thetaRows = thetaValues.map((value, index) => `<div class="rsdm-angle-row" style="display:flex;align-items:center;gap:6px">
      <span style="font-size:11px;color:var(--text-secondary);min-width:22px">${index + 1}.</span>
      <input class="form-input rsdm-theta-item" type="number" step="any" value="${UI.escapeAttr(value)}" style="width:96px" placeholder="deg" oninput="${onChange}" onchange="${onChange}" />
      ${index === 0 ? '' : `<button class="btn btn-secondary" style="padding:0 8px;height:30px;font-size:13px" onclick="this.closest('.rsdm-angle-row').remove();${onChange}">-</button>`}
    </div>`).join('');
    const lfsRows = this._dmRsdmLfsRows(vals, Math.max(1, selectedCount));
    const lfsHeaders = selectedNames.map((name, index) =>
      `\\(\\sigma^E_{${index + 1}}\\) <span style="font-size:10px;color:var(--text-secondary)">(${UI.escapeHtml(name)})</span>`
    );
    const lfsBody = lfsRows.map(row => `<tr class="rsdm-lfs-row">
      ${Array.from({ length: Math.max(1, selectedCount) }).map((_, index) => `<td style="padding:4px 6px"><input type="number" step="any" class="form-input rsdm-lfs-cell" style="width:100%;padding:6px 8px" value="${UI.escapeAttr(row[index] ?? 0)}" oninput="${onChange}" onchange="${onChange}" /></td>`).join('')}
      <td style="padding:4px 6px"><button class="btn btn-secondary" style="padding:0 8px;height:30px;font-size:13px" onclick="this.closest('tr').remove();${onChange}">-</button></td>
    </tr>`).join('');
    const rRatios = this._dmRsdmRRatios(vals, Math.max(1, selectedCount));
    const fixLoad2Active = selectedCount === 2
      && nVertValue === '2'
      && this._dmBoolFlag(vals.fix_load2, false);
    if (fixLoad2Active) rRatios[1] = 1;
    const rRatioFields = `<div id="${prefix}-r-ratios" style="display:flex;flex-direction:column;gap:8px;padding:14px 16px;background:rgba(255,255,255,0.04);border-radius:10px;border:1px solid rgba(255,255,255,0.08)">
      <div style="font-size:12px;font-weight:600;color:var(--text-secondary)">Stress ratios \\(R_i\\)</div>
      <div class="form-hint">\\(R_i = \\sigma_i^{min}/\\sigma_i^{max}\\)，用于确定第 \\(i\\) 个 load case 在载荷域另一侧角点的位置；\\(R_i=0\\) 表示从零载荷到最大载荷，\\(R_i=-1\\) 表示 fully reversed。</div>
      <div class="dm-field-grid">
        ${Array.from({ length: Math.max(1, selectedCount) }).map((_, index) => `<div class="form-group">
          <label class="form-label">R${index + 1} ${selectedNames[index] ? `(${UI.escapeHtml(selectedNames[index])})` : ''}</label>
          <input id="${prefix}-r-ratio-${index}" class="form-input rsdm-r-ratio-item" type="number" step="any" value="${UI.escapeAttr(rRatios[index] ?? 0)}" ${fixLoad2Active && index === 1 ? 'disabled title="Fix load2 启用时，R2 固定为 1；Load2 factor 仍作为定载倍率生效。"' : ''} oninput="${onChange}" onchange="${onChange}" />
        </div>`).join('')}
      </div>
    </div>`;
    const loadCombinationHint = isShakedown
      ? 'RSDM-S 可用 theta 生成载荷方向，也可直接填写 load_factor_set 矩阵；多个 load point 会逐一求解并写入同一个 summary MAT。'
      : 'RSDM 可用 theta 生成载荷方向，也可直接填写 load_factor_set；steady RSDM 使用第一行作为 Load1/Load2 的倍率向量。';
    const lfsHint = isShakedown
      ? '每行对应一个载荷比例向量；RSDM-S 会逐行遍历并写入同一个 summary MAT。'
      : '第一行对应本次 RSDM steady-state 的载荷比例向量；默认 Load1=1、Load2=1，可分别修改。';
    const loadCombinationFields = selectedCount >= 2 ? `<div style="display:flex;flex-direction:column;gap:12px">
      ${rRatioFields}
      <div class="form-group">
        <label class="form-label">Load combination definition (use_angles)</label>
        <select class="form-input" id="${prefix}-use-angles" onchange="App._onRsdmLoadCombinationChange('${kind}')">
          <option value="yes" ${loadMode !== 'no' ? 'selected' : ''}>yes - angle parameterization</option>
          <option value="no" ${loadMode === 'no' ? 'selected' : ''}>no - manual load_factor_set</option>
        </select>
        <div class="form-hint">${loadCombinationHint}</div>
      </div>
      <div id="${prefix}-angles-section" style="display:${loadMode === 'no' ? 'none' : 'flex'};flex-direction:column;gap:10px;padding:14px 16px;background:rgba(255,255,255,0.04);border-radius:10px;border:1px solid rgba(255,255,255,0.08)">
        <div style="font-size:12px;font-weight:600;color:var(--text-secondary)">theta_deg list (degree)</div>
        <div id="${prefix}-theta-list" style="display:flex;flex-direction:column;gap:4px;margin-top:2px">${thetaRows}</div>
        <button class="btn btn-secondary" style="width:max-content;padding:4px 12px;font-size:12px" onclick="App._addRsdmAngle('${kind}')">添加 theta</button>
        <div class="form-hint">当前 RSDM runner 使用第一个 theta 作为 angle_deg 兼容字段；完整列表会写入 cfg，便于后续扩展。</div>
      </div>
      <div id="${prefix}-lfs-section" style="display:${loadMode === 'no' ? 'flex' : 'none'};flex-direction:column;gap:10px;padding:14px 16px;background:rgba(255,255,255,0.04);border-radius:10px;border:1px solid rgba(255,255,255,0.08)">
        <div style="font-size:12px;font-weight:600;color:var(--text-secondary)">load_factor_set</div>
        <div class="form-hint">${lfsHint}</div>
        <table style="border-collapse:collapse;width:100%;margin-top:4px">
          <thead>
            <tr>
              ${lfsHeaders.map(header => `<th style="font-size:12px;color:var(--text-secondary);padding:4px 8px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.1)">${header}</th>`).join('')}
              <th style="width:36px"></th>
            </tr>
          </thead>
          <tbody id="${prefix}-lfs-body">${lfsBody}</tbody>
        </table>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <button class="btn btn-secondary" style="padding:6px 14px;font-size:13px" onclick="App._addRsdmLfsRow('${kind}', ${selectedCount})">添加行</button>
          <button class="btn btn-secondary" style="padding:6px 14px;font-size:13px" onclick="App._normalizeRsdmLfs('${kind}')">归一化</button>
        </div>
      </div>
    </div>` : `<div style="display:flex;flex-direction:column;gap:12px">${rRatioFields}<div class="form-hint">单个 INP 时不需要载荷组合矩阵；RSDM 会按该 load case 构造周期路径。</div></div>`;
    const vertexFieldsV2 = `<div class="dm-field-grid">
      <div class="form-group">
        <label class="form-label">Vertex count</label>
        <select class="form-input" id="${prefix}-n-vert" onchange="App._onRsdmNVertChange('${kind}')">
          ${nVertOptions}
        </select>
      </div>
      <div class="form-group" style="grid-column:1 / -1">
        ${loadCombinationFields}
      </div>
      ${fixLoad2Field}
    </div>`;

    const outerFields = isShakedown ? `<div class="dm-field-grid">
      <div class="form-group">
        <label class="form-label">Inner cap mode</label>
        <select class="form-input" id="${prefix}-inner-cap-mode" onchange="${onChange}">
          <option value="fixed" ${vals.inner_iteration_cap_mode === 'fixed' ? 'selected' : ''}>fixed</option>
          <option value="adaptive" ${vals.inner_iteration_cap_mode === 'adaptive' ? 'selected' : ''}>adaptive</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Adaptive min iterations</label>
        <input class="form-input" type="number" min="1" step="1" id="${prefix}-adaptive-min-iterations" value="${UI.escapeAttr(vals.adaptive_inner_min_iterations ?? '100')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
      <div class="form-group">
        <label class="form-label">Adaptive rel high</label>
        <input class="form-input" type="number" step="any" id="${prefix}-adaptive-rel-high" value="${UI.escapeAttr(vals.adaptive_inner_rel_high ?? '0.20')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
      <div class="form-group">
        <label class="form-label">Adaptive rel low</label>
        <input class="form-input" type="number" step="any" id="${prefix}-adaptive-rel-low" value="${UI.escapeAttr(vals.adaptive_inner_rel_low ?? '0.01')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
      <div class="form-group">
        <label class="form-label">Chi</label>
        <input class="form-input" type="number" step="any" id="${prefix}-chi" value="${UI.escapeAttr(vals.chi ?? '3.0')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
      <div class="form-group">
        <label class="form-label">Outer tol</label>
        <input class="form-input" type="number" step="any" id="${prefix}-outer-tol" value="${UI.escapeAttr(vals.outer_tol ?? '1.0e-2')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
      <div class="form-group">
        <label class="form-label">Max outer iterations</label>
        <input class="form-input" type="number" min="1" step="1" id="${prefix}-max-outer-iterations" value="${UI.escapeAttr(vals.max_outer_iterations ?? '30')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
      <div class="form-group">
        <label class="form-label">Extract residual field</label>
        <label style="display:flex;align-items:center;gap:10px;min-height:38px">
          <input type="checkbox" id="${prefix}-extract-field" ${this._dmBoolFlag(vals.extract_residual_field, true) ? 'checked' : ''} onchange="${onChange}" style="width:16px;height:16px;accent-color:var(--accent-blue)">
          <span style="font-size:13px;color:var(--text-secondary)">导出 residual field 可视化字段</span>
        </label>
      </div>
      <div class="form-group">
        <label class="form-label">Residual field max iterations</label>
        <input class="form-input" type="number" min="1" step="1" id="${prefix}-residual-field-max-iterations" value="${UI.escapeAttr(vals.residual_field_max_iterations ?? '500')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
      <div class="form-group">
        <label class="form-label">Residual field constant rel tol</label>
        <input class="form-input" type="number" step="any" id="${prefix}-residual-field-rel-tol" value="${UI.escapeAttr(vals.residual_field_constant_rel_tol ?? '1.0e-8')}" oninput="${onChange}" onchange="${onChange}" />
      </div>
    </div>` : `<label style="display:flex;align-items:center;gap:10px;min-height:38px">
      <input type="checkbox" id="${prefix}-extract-field" ${this._dmBoolFlag(vals.extract_residual_field, true) ? 'checked' : ''} onchange="${onChange}" style="width:16px;height:16px;accent-color:var(--accent-blue)">
      <span style="font-size:13px;color:var(--text-secondary)">导出 residual field 可视化字段</span>
    </label>`;

    return `<div class="card" style="margin-bottom:16px">
      <div class="card-title">${isShakedown ? 'RSDM-S Shakedown 配置' : 'RSDM Steady-state 配置'}</div>
      <div class="dm-status-line" style="margin-top:8px">最近解析时间：${UI.escapeHtml(state.lastParsedAt || '-')}</div>
      ${this._renderRsdmMessages(state)}
      <div style="display:flex;flex-direction:column;gap:18px;margin-top:16px">
        ${commonFields}
        ${coreFields}
        ${vertexFieldsV2}
        ${outerFields}
      </div>
    </div>
    <div class="card" style="margin-bottom:16px">
      <div class="card-title">提交前差异提醒</div>
      <div id="${prefix}-current-diff-summary">${this._renderRsdmDiffSummary(kind)}</div>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px">
      <button class="btn btn-primary" onclick="App.saveAndRunRsdmWorkflow('${kind}')">${isShakedown ? '运行 RSDM-S' : '运行 RSDM'}</button>
    </div>`;
  },

  _renderRsdmWorkflow(kind) {
    const model = this._directMethods?.model;
    const state = this._dmRsdmState(kind);
    const isShakedown = kind === 'shakedown';
    const group = this._dmRsdmGroupName(kind);
    const parsedNames = (state?.parsedSelectionKey || '').split('|').filter(Boolean);
    const availableInps = model?.abaqus?.inp_files || [];
    const rememberedNames = (state?.selectedNames || []).filter(name => availableInps.includes(name));
    const defaultNames = state?.selectionTouched
      ? rememberedNames
      : (rememberedNames.length
          ? rememberedNames
          : (parsedNames.length ? parsedNames : availableInps.slice(0, isShakedown ? 2 : 1)));
    const options = (model?.abaqus?.inp_files || []).map((name, index) => {
      const checked = defaultNames.includes(name) ? 'checked' : '';
      return `<label class="dm-inp-option" for="${group}-${index}">
        <input type="checkbox" name="${group}" id="${group}-${index}" value="${UI.escapeAttr(name)}" ${checked} onchange="App._onRsdmInpSelectionChange('${kind}')">
        <span class="dm-inp-name">${UI.escapeHtml(name)}</span>
      </label>`;
    }).join('');
    return `<div class="card" style="margin-bottom:16px">
      <div class="card-title">${isShakedown ? 'RSDM-S workflow — elastic load-case INP' : 'RSDM workflow — steady-state INP'}</div>
      <div class="dm-status-line" style="margin-top:8px">${isShakedown ? '每个 INP 表示同一模型和边界条件下的一个载荷施加方式。' : 'RSDM steady-state 可从一个或两个 elastic load-case INP 构建载荷路径。'}</div>
      <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:14px 0 12px">
        <button class="btn btn-secondary" onclick="App.parseSelectedRsdmInp('${kind}')">解析所选 INP</button>
        <span id="rsdm-${kind}-parse-status" class="dm-status-line">当前所选 INP 尚未解析。</span>
      </div>
      <div class="dm-option-list">${options}</div>
    </div>
    <div id="rsdm-${kind}-config-area">${this._renderRsdmForm(kind)}</div>`;
  },

  _dmBuildPathList(paths) {
    return `[\n${paths.map(path => `    "${path}"`).join(',\n')}\n]`;
  },

  async saveAndRunRsdmWorkflow(kind) {
    const model = this._directMethods?.model;
    const state = this._dmRsdmState(kind);
    const selected = this._dmSelectedInpNames(this._dmRsdmGroupName(kind));
    if (!model || !state || !selected.length) {
      alert('请先选择并解析 RSDM INP。');
      return;
    }
    if (state.parsedSelectionKey !== selected.join('|') || !state.metaDetails.length) {
      alert('请先解析当前所选 INP。');
      return;
    }
    this._syncRsdmDraftValues(kind);
    const vals = state.cfgValues || {};
    if (!String(vals.yield_stress || '').trim()) {
      alert('RSDM/RSDM-S 需要 Yield stress。请填写后再提交。');
      return;
    }
    if (state.errors?.length) {
      alert(`轻量检查仍有问题：\n${state.errors.join('\n')}`);
      return;
    }

    const isShakedown = kind === 'shakedown';
    const modelRoot = this.winToWsl(model.path);
    const inpPaths = selected.map(name => `${modelRoot}/abaqus/${name}`);
    const loadCombination = this._collectRsdmLoadCombination(kind, selected.length);
    const workflowDir = isShakedown ? 'RSDM_shakedown' : 'steady_state_RSDM';
    const cfgFileName = isShakedown ? 'rsdm_analysis.template.cfg' : 'steady_state.template.cfg';
    const run = this._analysisRunPaths(
      model,
      workflowDir,
      `${isShakedown ? 'RSDMS' : 'RSDM'}_${selected.length}LC_${vals.n_vert || '2'}P`,
      cfgFileName,
    );
    const lines = [
      '# RSDM workflow configuration (auto-generated by jaxmech web)',
      `# run_dir: ${run.runLabel}`,
      '',
      `method = ${isShakedown ? 'rsdm_s' : 'rsdm'}`,
      `inp_files = ${this._dmBuildPathList(inpPaths)}`,
      `yield_stress_override = ${vals.yield_stress}`,
    ];
    if (!isShakedown) {
      lines.splice(4, 0, `load_factor = ${vals.load_factor || '1.0'}`);
    }
    if (String(vals.E || '').trim()) lines.push(`E_override = ${vals.E}`);
    if (String(vals.nu || '').trim()) lines.push(`nu_override = ${vals.nu}`);
    lines.push('');
    lines.push(`time_grid_type = ${vals.time_grid_type || 'gauss_legendre'}`);
    lines.push(`n_time_points = ${vals.n_time_points || '3'}`);
    lines.push(`n_fourier_terms = ${vals.n_fourier_terms || '3'}`);
    lines.push(`inner_tol = ${vals.inner_tol || '1.0e-2'}`);
    lines.push(`classification_rel_tol = ${vals.classification_rel_tol || '1.0e-2'}`);
    lines.push(`max_inner_iterations = ${vals.max_inner_iterations || '100'}`);
    if (isShakedown) {
      lines.push(`inner_iteration_cap_mode = ${vals.inner_iteration_cap_mode || 'fixed'}`);
      lines.push(`adaptive_inner_min_iterations = ${vals.adaptive_inner_min_iterations || '100'}`);
      lines.push(`adaptive_inner_rel_high = ${vals.adaptive_inner_rel_high || '0.20'}`);
      lines.push(`adaptive_inner_rel_low = ${vals.adaptive_inner_rel_low || '0.01'}`);
      lines.push(`chi = ${vals.chi || '3.0'}`);
      lines.push(`outer_tol = ${vals.outer_tol || '1.0e-2'}`);
      lines.push(`max_outer_iterations = ${vals.max_outer_iterations || '30'}`);
      lines.push(`residual_field_max_iterations = ${vals.residual_field_max_iterations || '500'}`);
      lines.push(`residual_field_constant_rel_tol = ${vals.residual_field_constant_rel_tol || '1.0e-8'}`);
    }
    lines.push(`extract_residual_field = ${this._dmBoolFlag(vals.extract_residual_field, true) ? 'true' : 'false'}`);
    lines.push(`n_vert = ${vals.n_vert || '2'}`);
    const rRatios = this._dmRsdmRRatios(vals, selected.length);
    lines.push(`R_ratios = [${rRatios.join(', ')}]`);
    if (selected.length >= 2) {
      lines.push(`use_angles = ${loadCombination.use_angles}`);
      if (loadCombination.use_angles === 'yes') {
        const thetas = loadCombination.theta_deg?.length ? loadCombination.theta_deg : [0];
        lines.push(`theta_deg = [${thetas.join(', ')}]`);
      } else {
        const rows = loadCombination.load_factor_set?.length
          ? loadCombination.load_factor_set
          : [Array(selected.length).fill(1.0)];
        lines.push(`load_factor_set = [\n${rows.map(row => `    [${row.join(', ')}]`).join(',\n')}\n]`);
      }
    }
    lines.push(`angle_deg = ${loadCombination.angle_deg ?? vals.angle_deg ?? '0.0'}`);
    lines.push(`fix_load2 = ${selected.length === 2 && String(vals.n_vert || '2') === '2' && this._dmBoolFlag(vals.fix_load2, false) ? 'true' : 'false'}`);
    lines.push(`use_b_ext = ${this._dmBoolFlag(vals.use_b_ext, true) ? '1' : '0'}`);
    lines.push('');

    const moduleKey = isShakedown ? 'direct_methods_shakedown' : 'direct_methods_steady_state_rsdm';
    await this.apiPut(`/api/config/${moduleKey}?model_path=${encodeURIComponent(run.runDirWin)}`, {
      content: lines.join('\n'),
    });
    const result = await this.apiPost('/api/run', {
      module: isShakedown
        ? 'jaxmech.modules.direct_methods.rsdm.run'
        : 'jaxmech.modules.direct_methods.run_steady_state',
      args: ['--config', run.cfgWsl],
    });
    this.viewTask(result.id);
  },

  _renderDirectMethodsShell() {
    const container = document.getElementById('direct-methods-content');
    const model = this._directMethods?.model;
    if (!container || !model) return;
    const active = this._directMethods.activeWorkflow || 'steady_state';
    const steadyMethod = this._directMethods.steadyMethod || 'dca';
    let html = `<div style="font-size:13px;color:var(--text-secondary);margin-bottom:12px;padding:10px 14px;background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid rgba(255,255,255,0.06)">
      模型: <b style="color:var(--text-primary)">${UI.escapeHtml(model.name)}</b>
      <span style="color:rgba(255,255,255,0.25);margin:0 8px">|</span>
      <code style="font-size:11px;color:var(--accent)">${UI.escapeHtml(model.path)}</code>
    </div>
    <div class="dm-tabs">
      <button class="dm-tab ${active === 'steady_state' ? 'active' : ''}" onclick="App._dmSetActiveWorkflow('steady_state')">Steady-state</button>
      <button class="dm-tab ${active === 'shakedown' ? 'active' : ''}" onclick="App._dmSetActiveWorkflow('shakedown')">Shakedown</button>
    </div>`;

    if (active === 'steady_state') {
      html += `<div class="dm-method-tabs">
        <button class="dm-method-tab ${steadyMethod === 'dca' ? 'active' : ''}" onclick="App._dmSetSteadyMethod('dca')">DCA</button>
        <button class="dm-method-tab ${steadyMethod === 'rsdm' ? 'active' : ''}" onclick="App._dmSetSteadyMethod('rsdm')">RSDM</button>
      </div>`;
      if (steadyMethod === 'dca') {
        const cfgInpName = this._directMethods.dcaSelectedInp || model.abaqus.inp_files[0];
        html += `<div class="card" style="margin-bottom:16px">
          <div class="card-title">DCA workflow — INP 选择与解析</div>
          <div class="dm-status-line" style="margin:10px 0 14px">一个 Direct Cyclic INP 对应一个 DCA steady-state workflow。</div>
          <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px">
            <button class="btn btn-secondary" onclick="App.parseSelectedDcaInp()">解析所选 INP</button>
            <span id="dca-parse-status" class="dm-status-line">当前所选 INP 尚未解析。</span>
          </div>
          <div id="dca-inp-list" class="dm-option-list">`;
        model.abaqus.inp_files.forEach((f, i) => {
          const checked = cfgInpName === f ? 'checked' : '';
          html += `<label class="dm-inp-option" for="dca-inp-${i}">
            <input type="radio" name="dca-inp" id="dca-inp-${i}" value="${UI.escapeAttr(f)}" ${checked} onchange="App._onDcaInpSelectionChange()">
            <span class="dm-inp-name">${UI.escapeHtml(f)}</span>
          </label>`;
        });
        html += `</div></div><div id="direct-methods-config-area"></div>`;
      } else {
        html += this._renderRsdmWorkflow('steady');
      }
    } else {
      html += this._renderRsdmWorkflow('shakedown');
    }
    container.innerHTML = html;
    if (active === 'steady_state' && steadyMethod === 'dca') {
      this._renderDcaParseStatus();
      this._renderDirectMethodsConfigForm();
    } else if (active === 'steady_state') {
      this._renderRsdmParseStatus('steady');
    } else {
      this._renderRsdmParseStatus('shakedown');
    }
    this._typesetMath(container);
  },

  async loadDirectMethods() {
    const container = document.getElementById('direct-methods-content');
    if (!this.selectedModel) {
      container.innerHTML = UI.hint('warning', '请先在模型浏览中选择一个工作模型。', '前往模型浏览', "App.requestModelFor('direct-methods')");
      return;
    }
    const { source, name } = this.selectedModel;
    container.innerHTML = '<div class="empty-state"><div class="empty-icon">⏳</div><div class="empty-text">加载中...</div></div>';

    const model = await this.api(`/api/models/${source}/${name}`).catch(() => null);
    if (!model) {
      container.innerHTML = UI.hint('error', '无法获取模型状态。');
      return;
    }
    if (model.abaqus.inp_count === 0) {
      container.innerHTML = UI.hint('error', '未找到 INP 文件。Direct Methods 需要先从 abaqus/ 目录读取 INP。');
      return;
    }

    const dcaCfg = await this.api(`/api/config/direct_methods_steady_state_dca?model_path=${encodeURIComponent(model.path)}`).catch(() => ({ values: {} }));
    const steadyRsdmCfg = await this.api(`/api/config/direct_methods_steady_state_rsdm?model_path=${encodeURIComponent(model.path)}`).catch(() => ({ values: {} }));
    const shakedownCfg = await this.api(`/api/config/direct_methods_shakedown?model_path=${encodeURIComponent(model.path)}`).catch(() => ({ values: {} }));
    const dcaVals = dcaCfg?.values || {};
    const steadyRsdmVals = steadyRsdmCfg?.source === 'custom'
      ? this._dmNormalizeRsdmCfgValues(steadyRsdmCfg?.values || {})
      : {};
    const shakedownVals = shakedownCfg?.source === 'custom'
      ? this._dmNormalizeRsdmCfgValues(shakedownCfg?.values || {})
      : {};
    const steadyMethod = steadyRsdmCfg?.source === 'custom' && dcaCfg?.source !== 'custom' ? 'rsdm' : 'dca';
    const cfgInpBase = String(dcaVals.inp_file || '').trim().split(/[\\/]/).pop();
    const cfgInpName = model.abaqus.inp_files.includes(cfgInpBase)
      ? cfgInpBase
      : model.abaqus.inp_files[0];
    const steadyRsdmSelected = this._dmInpNamesFromCfgValue(
      [steadyRsdmVals.inp_files, steadyRsdmVals.inp_file, steadyRsdmCfg?.content].filter(Boolean).join('\n'),
      model.abaqus.inp_files
    );
    const shakedownRsdmSelected = this._dmInpNamesFromCfgValue(
      [shakedownVals.inp_files, shakedownVals.inp_file, shakedownCfg?.content].filter(Boolean).join('\n'),
      model.abaqus.inp_files
    );

    this._directMethods = {
      model,
      activeWorkflow: 'steady_state',
      steadyMethod,
      dcaSelectedInp: cfgInpName,
      parsedInpMetaByName: {},
      parsedSelectionKey: '',
      lastParsedAt: '',
      cfgValues: { ...dcaVals },
      rsdm: {
        steady: {
          parsedSelectionKey: '',
          selectedNames: steadyRsdmSelected,
          selectionTouched: steadyRsdmSelected.length > 0,
          lastParsedAt: '',
          metaDetails: [],
          cfgValues: this._dmNormalizeRsdmCfgValues({ ...this._dmBuildRsdmDefaults('steady', []), ...steadyRsdmVals }),
          warnings: [],
          errors: [],
        },
        shakedown: {
          parsedSelectionKey: '',
          selectedNames: shakedownRsdmSelected,
          selectionTouched: shakedownRsdmSelected.length > 0,
          lastParsedAt: '',
          metaDetails: [],
          cfgValues: this._dmNormalizeRsdmCfgValues({ ...this._dmBuildRsdmDefaults('shakedown', []), ...shakedownVals }),
          warnings: [],
          errors: [],
        },
      },
    };
    this._renderDirectMethodsShell();
  },

  async _browseDcaValOdb() {
    let initialDir = 'C:\\';
    try {
      const cfg = await this.api('/api/config/env').catch(() => null);
      const winRoot = cfg?.values?.win_stored_root?.trim();
      if (winRoot) {
        initialDir = winRoot;
      } else {
        initialDir = this._validation?.model?.path || this._directMethods?.model?.path || 'C:\\';
      }
    } catch (_) {
      initialDir = this._validation?.model?.path || this._directMethods?.model?.path || 'C:\\';
    }
    const res = await this.apiPost('/api/browse', {
      field_type: 'odb_file',
      title: '选择 DCA 对应 ODB 文件',
      initial_dir: initialDir,
    }).catch(() => null);
    if (res?.path) {
      const input = document.getElementById('dca-val-odb');
      if (input) input.value = res.path;
      const hint = document.getElementById('dca-val-odb-hint');
      if (hint) {
        hint.style.color = 'var(--text-secondary)';
        hint.textContent = '';
      }
    }
  },

  async _startDirectMethodValidation(matAbsPath) {
    const formArea = document.getElementById('validation-form-area') || document.getElementById('direct-methods-validation-form');
    if (!formArea) return;
    const model = this._validation?.model || this._directMethods?.model;
    const metaRes = await this.apiPost('/api/models/mat-meta', { paths: [matAbsPath] }).catch(() => null);
    const matMeta = metaRes?.details?.[0] || null;
    if (this._validation) {
      this._validation.currentDcaMatMeta = matMeta;
    }

    const directInfo = matMeta?.direct_method_info || {};
    const supportsInterpolation = Boolean(matMeta?.supports_interpolated_validation);
    const matStem = this._sanitizeRunToken(String(matMeta?.name || 'dca_result').replace(/\.mat$/i, ''), 'DCA');
    const run = model ? this._analysisRunPaths(model, 'validation', `DCAValidation_${matStem}`, 'validation.template.cfg') : null;
    const validationDir = run?.runDirWin || '';

    formArea.innerHTML = `<div class="card">
      <div class="card-title">DCA ODB 验证配置</div>
      <div style="margin-top:10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap"><span style="display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;background:rgba(52,211,153,0.14);color:#34d399;font-size:12px;font-weight:600">DCA / ODB</span></div>
      <div style="margin-top:12px;display:flex;flex-direction:column;gap:14px">
        <div class="form-group">
          <label class="form-label">JAX DCA MAT</label>
          <input class="form-input" type="text" id="dca-val-mat" value="${UI.escapeAttr(matAbsPath)}" readonly style="opacity:0.7;cursor:not-allowed;font-family:monospace;font-size:12px" />
        </div>
        <div class="form-group">
          <label class="form-label">对应 ODB 文件</label>
          <div style="display:flex;gap:8px;align-items:center">
            <input class="form-input" type="text" id="dca-val-odb" placeholder="正在搜索同名 ODB..." style="flex:1;font-family:monospace;font-size:12px" />
            <button class="btn btn-secondary" style="flex-shrink:0;padding:8px 14px" onclick="App._browseDcaValOdb()">浏览...</button>
          </div>
          <div id="dca-val-odb-hint" style="font-size:12px;color:var(--text-secondary);margin-top:4px"></div>
        </div>
        <div class="form-group">
          <label class="form-label">时间步策略</label>
          <div style="display:flex;flex-direction:column;gap:10px;margin-top:8px">
            <label style="display:flex;align-items:flex-start;gap:10px;cursor:pointer">
              <input type="radio" name="dca-val-align-mode" value="interpolate" ${supportsInterpolation ? 'checked' : 'disabled'} style="margin-top:3px;accent-color:var(--accent)">
              <span style="font-size:13px;line-height:1.6">
                <b>否，直接插值现有 JAX cycle history</b><br>
                <span style="color:var(--text-secondary)">不回到 INP 重跑 DCA，只把现有 JAX DCA frame 历史插值到 ODB frame_time 后 compare。</span>
              </span>
            </label>
            <label style="display:flex;align-items:flex-start;gap:10px;cursor:pointer">
              <input type="radio" name="dca-val-align-mode" value="align_odb" ${supportsInterpolation ? '' : 'checked'} style="margin-top:3px;accent-color:var(--accent)">
              <span style="font-size:13px;line-height:1.6">
                <b>是，对齐到 ODB frame_time</b><br>
                <span style="color:var(--text-secondary)">使用 JAX DCA 的周期历史 / Fourier 表达把结果重采样到 ODB 输出时间后 compare；这一步不依赖重新求解 DCA。</span>
              </span>
            </label>
          </div>
          <div class="form-hint" style="margin-top:8px">${supportsInterpolation ? `当前 DCA MAT 检测到 ${matMeta?.frame_count || 0} 个 JAX cycle frame，可直接做插值或 ODB 对齐。` : '当前 DCA MAT 缺少完整 cycle frame 历史，已强制切换到“对齐到 ODB frame_time”。'}</div>
        </div>
        <div style="font-size:12px;color:var(--text-secondary);line-height:1.7">DCA validation 现在和其它验证一样从 ODB 起步。页面会先提取 ABAQUS ODB，再把 JAX DCA 周期历史对齐到相同的时间采样后生成 summary、steady boxplot 和逐变量 frame boxplot。</div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px">
          <div class="form-group" style="margin:0">
            <label class="form-label">validation 输出目录</label>
            <input class="form-input" type="text" id="dca-val-output-dir" value="${UI.escapeAttr(validationDir)}" readonly style="opacity:0.7;cursor:not-allowed;font-family:monospace;font-size:12px" />
            <input type="hidden" id="dca-val-cfg" value="${UI.escapeAttr(run?.cfgWsl || '')}" />
            <input type="hidden" id="dca-val-output-dir-wsl" value="${UI.escapeAttr(run?.runDirWsl || '')}" />
            <input type="hidden" id="dca-val-run-dir-win" value="${UI.escapeAttr(run?.runDirWin || '')}" />
          </div>
        </div>
        <div style="display:flex;gap:12px">
          <button class="btn btn-primary" onclick="App.runDirectMethodValidation()">▶ 运行 DCA ODB 验证</button>
        </div>
      </div>
    </div>`;

    const odbHint = document.getElementById('dca-val-odb-hint');
    if (odbHint) odbHint.textContent = '正在搜索同名 ODB...';
    const findRes = await this.apiPost('/api/models/find-odb', {
      mat_path: matAbsPath,
      model_path: model?.path || '',
    }).catch(() => null);
    const odbInput = document.getElementById('dca-val-odb');
    if (findRes?.found && odbInput) {
      odbInput.value = findRes.odb_path;
      if (odbHint) {
        odbHint.style.color = '#34d399';
        odbHint.textContent = '✓ 已自动定位 DCA 对应 ODB 文件。';
      }
    } else if (odbHint) {
      odbHint.style.color = '#fbbf24';
      odbHint.textContent = '⚠ 未找到同名 ODB，请点击「浏览...」手动选择。';
    }

    formArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
  },

  async runDirectMethodValidation() {
    const model = this._validation?.model || this._directMethods?.model;
    const matPath = document.getElementById('dca-val-mat')?.value?.trim();
    const odbPath = document.getElementById('dca-val-odb')?.value?.trim();
    const outputDir = document.getElementById('dca-val-output-dir')?.value?.trim();
    const outputDirWsl = document.getElementById('dca-val-output-dir-wsl')?.value?.trim() || this.winToWsl(outputDir);
    const cfgWsl = document.getElementById('dca-val-cfg')?.value?.trim();
    const runDirWin = document.getElementById('dca-val-run-dir-win')?.value?.trim() || outputDir;
    const alignMode = document.querySelector('input[name="dca-val-align-mode"]:checked')?.value || 'interpolate';
    if (!model || !matPath) {
      alert('请先选择一个 DCA MAT。');
      return;
    }
    if (!odbPath) {
      alert('请先选择一个 ODB 文件。');
      return;
    }

    if (runDirWin && cfgWsl) {
      const cfgLines = [
        '# validation configuration (auto-generated by jaxmech web DCA validation form)',
        '',
        'mat_files = [',
        `    "${this.winToWsl(matPath)}"`,
        ']',
        '',
        'odb_files = [',
        `    "${this.winToWsl(odbPath)}"`,
        ']',
        '',
        'copy_mat = 1',
        `validation_dir = ${outputDirWsl}`,
        `align_mode = ${alignMode}`,
      ];
      await this.apiPut(`/api/config/validation?model_path=${encodeURIComponent(runDirWin)}`, {
        content: cfgLines.join('\n'),
      });
    }

    const args = [
      ...(cfgWsl ? ['--config', cfgWsl] : []),
      '--mat', this.winToWsl(matPath),
      '--odb', this.winToWsl(odbPath),
      '--model-root', this.winToWsl(model.path),
      '--validation-dir', outputDirWsl,
      '--copy-mat',
    ];
    if (alignMode === 'align_odb') {
      args.push('--align-time-steps');
    }

    const result = await this.apiPost('/api/run', {
      module: 'jaxmech.modules.validation.nonlinear.run',
      args,
    });
    this.viewTask(result.id);
  },


  // ──────────── Validation ────────────
  async loadValidation() {
    const container = document.getElementById('validation-content');
    if (!this.selectedModel) {
      container.innerHTML = UI.hint('warning', '请先在模型浏览中选择一个工作模型。', '前往模型浏览', "App.requestModelFor('validation')");
      return;
    }
    const { source, name } = this.selectedModel;
    container.innerHTML = '<div class="empty-state"><div class="empty-icon">⏳</div><div class="empty-text">扫描 MAT 文件...</div></div>';

    const model = await this.api(`/api/models/${source}/${name}`).catch(() => null);
    if (!model) { container.innerHTML = UI.hint('error', '无法获取模型状态。'); return; }

    // Scan MATs and build a source/result mapping:
    // source list only comes from inc_analysis/, while validation/
    // contributes status for same-named files.
    const scanRes = await this.apiPost('/api/models/scan-mats', { model_path: model.path }).catch(() => null);
    const allScanMats = scanRes?.mat_files || [];
    const subdirOf = m => String(m.subdir || '').toLowerCase().replace(/\\/g, '/');
    const underWorkflow = (m, root) => {
      const subdir = subdirOf(m);
      return subdir === root || subdir.startsWith(`${root}/`);
    };
    const latestByName = (items) => {
      const out = new Map();
      [...items].sort((a, b) => Number(a.mtime || 0) - Number(b.mtime || 0)).forEach(item => out.set(item.name, item));
      return out;
    };
    const incMats = allScanMats.filter(m => underWorkflow(m, 'inc_analysis'));
    const dcaMats = [];
    const validationMats = allScanMats.filter(m => underWorkflow(m, 'validation'));
    const validationByName = latestByName(validationMats);

    if (!incMats.length) {
      container.innerHTML = UI.hint('error', '当前模型下没有 inc_analysis MAT。demo 版验证模块需要已有实体弹性结果文件。', '前往增量分析', "App.navigate('elastic')");
      return;
    }

    // Read metadata only from validation results, because validation status is
    // determined by same-named files under validation/ rather than the source
    // inc_analysis MAT itself.
    let validationDetails = [];
    if (validationMats.length > 0) {
      const metaRes = await this.apiPost('/api/models/mat-meta', {
        paths: validationMats.map(m => m.abs_path)
      }).catch(() => null);
      validationDetails = metaRes?.details || [];
      const mtimeByPath = new Map(validationMats.map(m => [m.abs_path, Number(m.mtime || 0)]));
      validationDetails.forEach(item => { item.mtime = mtimeByPath.get(item.path) || 0; });
    }
    const validationMetaByName = latestByName(validationDetails);
    let dcaMetaByName = new Map();
    if (dcaMats.length > 0) {
      const dcaMetaRes = await this.apiPost('/api/models/mat-meta', {
        paths: dcaMats.map(m => m.abs_path),
      }).catch(() => null);
      dcaMetaByName = latestByName(dcaMetaRes?.details || []);
    }

    // ── Render MAT status table ──
    const matRows = incMats.map((m) => {
      const validationMat = validationByName.get(m.name);
      const validationMeta = validationMetaByName.get(m.name);
      let badge = `<span style="color:var(--text-secondary);font-size:11px;font-weight:600;white-space:nowrap">○ validation 目录无同名 MAT</span>`;
      if (validationMeta?.is_validated) {
        badge = `<span style="color:#34d399;font-size:11px;font-weight:600;white-space:nowrap">✓ 已验证</span>`;
      } else if (validationMat && validationMeta?.has_validation_flag) {
        badge = `<span style="color:#f59e0b;font-size:11px;font-weight:600;white-space:nowrap">⚠ validation MAT 标记未通过</span>`;
      } else if (validationMat) {
        badge = `<span style="color:#60a5fa;font-size:11px;font-weight:600;white-space:nowrap">📄 validation 目录已有同名 MAT</span>`;
      }
      // Use data-matpath attribute to avoid backslash escaping in onclick strings
      return `<div class="val-mat-row" data-matpath="${UI.escapeAttr(m.abs_path)}"
          style="display:flex;align-items:center;gap:10px;padding:10px 14px;
          background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid rgba(255,255,255,0.07);cursor:pointer">
        <span style="font-size:13px;font-family:monospace;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
          title="${UI.escapeHtml(m.abs_path)}">📦 ${UI.escapeHtml(m.name)}</span>
        ${badge}
        <span style="font-size:12px;color:var(--accent)">选择 →</span>
      </div>`;
    }).join('');

    let html = `<div style="font-size:13px;color:var(--text-secondary);margin-bottom:12px;padding:10px 14px;
        background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid rgba(255,255,255,0.06)">
      模型: <b style="color:var(--text-primary)">${UI.escapeHtml(model.name)}</b>
      <span style="color:rgba(255,255,255,0.25);margin:0 8px">|</span>
      <code style="font-size:11px;color:var(--accent)">${UI.escapeHtml(model.path)}</code>
    </div>`;

    if (incMats.length) {
      html += `<div class="card" style="margin-bottom:16px">
        <div class="card-title">分析 MAT 文件 — 点击选择进行 ODB 验证</div>
        <div style="margin-top:12px;display:flex;flex-direction:column;gap:8px">
          ${matRows}
        </div>
        <div style="font-size:12px;color:var(--text-secondary);margin-top:12px">
          状态依据：这里只检查 validation 文件夹中是否存在同名 MAT；这不是在判断 inc_analysis 结果 MAT 是否存在。若 validation 目录中的同名 MAT 可解析出 validated_with_ODB 等字段，则显示“已验证”，否则仅显示“validation 目录已有同名 MAT”。demo 版仅开放实体单元弹性 ODB 验证。
        </div>
      </div>`;
    }

    if (dcaMats.length) {
      const dcaRows = dcaMats.map((m) => {
        const meta = dcaMetaByName.get(m.name) || {};
        const ref = meta?.supports_time_alignment
          ? `<span style="color:#60a5fa;font-size:11px;font-weight:600;white-space:nowrap">支持 ODB 时间对齐</span>`
          : `<span style="color:var(--text-secondary);font-size:11px;font-weight:600;white-space:nowrap">缺少 cycle frame history</span>`;
        return `<div class="dca-val-mat-row" data-matpath="${UI.escapeAttr(m.abs_path)}"
            style="display:flex;align-items:center;gap:10px;padding:10px 14px;
            background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid rgba(255,255,255,0.07);cursor:pointer">
          <span style="font-size:13px;font-family:monospace;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
            title="${UI.escapeHtml(m.abs_path)}">📦 ${UI.escapeHtml(m.name)}</span>
          ${ref}
          <span style="font-size:12px;color:var(--accent)">选择 →</span>
        </div>`;
      }).join('');

      html += `<div class="card" style="margin-bottom:16px">
        <div class="card-title">DCA 结果 MAT 文件 — 点击选择进行 ODB 验证</div>
        <div style="margin-top:12px;display:flex;flex-direction:column;gap:8px">${dcaRows}</div>
        <div style="font-size:12px;color:var(--text-secondary);margin-top:12px">DCA validation 现在同样从 ODB 起步。页面会先提取 ABAQUS ODB，再根据时间采样是否对齐，在“现有 cycle history 插值”和“对齐到 ODB frame_time 的 Fourier 重采样”之间切换。</div>
      </div>`;
    }

    html += `<div id="validation-form-area"></div>`;
    container.innerHTML = html;

    // Bind click events AFTER innerHTML is set (avoids onclick backslash-escaping issue)
    container.querySelectorAll('.val-mat-row').forEach(row => {
      row.addEventListener('click', () => {
        App._startValidation(row.dataset.matpath);
      });
    });
    container.querySelectorAll('.dca-val-mat-row').forEach(row => {
      row.addEventListener('click', () => {
        App._startDirectMethodValidation(row.dataset.matpath);
      });
    });

    // Save model ref for later use
    this._valModel = model;
    this._validation = { model, currentMatMeta: null, currentDcaMatMeta: null };
  },

  // Called when user clicks a MAT in the validation list
  async _startValidation(matAbsPath) {
    const formArea = document.getElementById('validation-form-area');
    if (!formArea) return;
    const model = this._valModel;
    const metaRes = await this.apiPost('/api/models/mat-meta', { paths: [matAbsPath] }).catch(() => null);
    const matMeta = metaRes?.details?.[0] || null;
    this._validation.currentMatMeta = matMeta;
    const family = String(matMeta?.family || '').trim().toLowerCase();
    const branch = String(matMeta?.validation_branch || 'elastic').trim().toLowerCase();
    const materialModel = String(matMeta?.material_model || 'linear_elastic').trim().toLowerCase();
    if (family !== 'solid' || branch !== 'elastic' || materialModel !== 'linear_elastic') {
      formArea.innerHTML = UI.hint('warning', 'demo 版 validation 仅开放实体单元线弹性 MAT 的 ODB 对标。shell、nonlinear、DCA validation 请使用完整版。');
      formArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }

    const isNonlinear = false;
    const supportsInterpolation = Boolean(matMeta?.supports_interpolated_validation);
    const modeBadge = isNonlinear
      ? `<span style="display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;background:rgba(96,165,250,0.14);color:#60a5fa;font-size:12px;font-weight:600">多步 / nonlinear</span>`
      : `<span style="display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;background:rgba(52,211,153,0.14);color:#34d399;font-size:12px;font-weight:600">单步 / elastic</span>`;

    const alignControls = isNonlinear ? `<div class="form-group">
          <label class="form-label">时间步策略</label>
          <div style="display:flex;flex-direction:column;gap:10px;margin-top:8px">
            <label style="display:flex;align-items:flex-start;gap:10px;cursor:pointer">
              <input type="radio" name="val-align-mode" value="interpolate" ${supportsInterpolation ? 'checked' : 'disabled'} style="margin-top:3px;accent-color:var(--accent)">
              <span style="font-size:13px;line-height:1.6">
                <b>否，直接插值现有 JAX frame</b><br>
                <span style="color:var(--text-secondary)">不重算 JAX，只把现有 MAT 的 frame 历史插值到 ODB 输出时间后进行 compare。</span>
              </span>
            </label>
            <label style="display:flex;align-items:flex-start;gap:10px;cursor:pointer">
              <input type="radio" name="val-align-mode" value="align_odb" ${supportsInterpolation ? '' : 'checked'} style="margin-top:3px;accent-color:var(--accent)">
              <span style="font-size:13px;line-height:1.6">
                <b>是，对齐到 ODB frame_time</b><br>
                <span style="color:var(--text-secondary)">按 ODB 的 frame 输出时间强制重新运行 JAX nonlinear_static，再做 compare。</span>
              </span>
            </label>
          </div>
          <div class="form-hint" style="margin-top:8px">${supportsInterpolation ? `当前 MAT 检测到 ${matMeta?.frame_count || 0} 个 JAX frame，可直接做插值对比。` : '当前 MAT 缺少完整 frame 历史，已强制切换到“对齐到 ODB 重算”。'}</div>
        </div>` : `<div style="font-size:12px;color:var(--text-secondary);line-height:1.7">当前 MAT 只有单步结果，验证模块将默认走 elastic 分支。</div>`;

    const nonlinearDebugControls = isNonlinear ? `<div class="form-group">
          <label class="form-label">附加诊断图</label>
          <div style="display:flex;align-items:flex-start;gap:10px;margin-top:8px">
            <input type="checkbox" id="val-emit-peeq-plot" style="margin-top:3px;accent-color:var(--accent)">
            <label for="val-emit-peeq-plot" style="font-size:13px;line-height:1.6;cursor:pointer">
              生成 PEEQ 演化对比图（调试用）
            </label>
          </div>
          <div class="form-hint" style="margin-top:8px">PEEQ 图比较 ABAQUS 的等效塑性应变 PEEQ 与 JAX 的等效塑性应变 EQPS 在整个加载过程中的 max/mean 演化。默认关闭，因为它主要用于弹塑性调试，不是常规验证必需产物。</div>
        </div>` : '';

    formArea.innerHTML = `<div class="card">
      <div class="card-title">ODB 验证配置</div>
      <div style="margin-top:10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap">${modeBadge}</div>
      <div style="margin-top:12px;display:flex;flex-direction:column;gap:14px">
        <div class="form-group">
          <label class="form-label">已选 MAT 文件</label>
          <input class="form-input" type="text" id="val-mat" value="${UI.escapeHtml(matAbsPath)}" readonly
            style="opacity:0.7;cursor:not-allowed;font-family:monospace;font-size:12px" />
        </div>
        <div class="form-group">
          <label class="form-label">对应 ODB 文件</label>
          <div style="display:flex;gap:8px;align-items:center">
            <input class="form-input" type="text" id="val-odb" placeholder="正在搜索同名 ODB..."
              style="flex:1;font-family:monospace;font-size:12px" />
            <button class="btn btn-secondary" style="flex-shrink:0;padding:8px 14px"
              onclick="App._browseValOdb()">浏览...</button>
          </div>
          <div id="val-odb-hint" style="font-size:12px;color:var(--text-secondary);margin-top:4px"></div>
        </div>
        ${alignControls}
        ${nonlinearDebugControls}
        <div style="display:flex;gap:12px">
          <button class="btn btn-primary" onclick="App.runValidation()">▶ 运行 ODB 验证</button>
        </div>
      </div>
    </div>`;

    formArea.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Auto-search for same-named ODB
    const odbHint = document.getElementById('val-odb-hint');
    if (odbHint) odbHint.textContent = '正在搜索同名 ODB...';
    const findRes = await this.apiPost('/api/models/find-odb', {
      mat_path: matAbsPath,
      model_path: model?.path || '',
    }).catch(() => null);

    const odbInput = document.getElementById('val-odb');
    if (findRes?.found && odbInput) {
      odbInput.value = findRes.odb_path;
      if (odbHint) { odbHint.style.color = '#34d399'; odbHint.textContent = '✓ 已自动定位同名 ODB 文件。'; }
    } else {
      if (odbHint) { odbHint.style.color = '#fbbf24'; odbHint.textContent = '⚠ 未找到同名 ODB，请点击「浏览...」手动选择。'; }
    }
  },

  async _browseValOdb() {
    // Use win_stored_root from env config as initial ODB browse dir,
    // falling back to model.path (which may be inside the project), then C:\
    let initialDir = 'C:\\';
    try {
      const cfg = await this.api('/api/config/env').catch(() => null);
      const winRoot = cfg?.values?.win_stored_root?.trim();
      if (winRoot) { initialDir = winRoot; }
      else { initialDir = this._valModel?.path || 'C:\\'; }
    } catch (_) {
      initialDir = this._valModel?.path || 'C:\\';
    }
    const res = await this.apiPost('/api/browse', {
      field_type: 'odb_file',
      title: '选择 ODB 文件',
      initial_dir: initialDir,
    }).catch(() => null);
    if (res?.path) {
      const odbInput = document.getElementById('val-odb');
      if (odbInput) odbInput.value = res.path;
      const odbHint = document.getElementById('val-odb-hint');
      if (odbHint) { odbHint.style.color = 'var(--text-secondary)'; odbHint.textContent = ''; }
    }
  },

  async runValidation() {
    const matPath = document.getElementById('val-mat')?.value?.trim();
    const odbPath = document.getElementById('val-odb')?.value?.trim();
    if (!matPath) { alert('请先选择 MAT 文件。'); return; }
    if (!odbPath) { alert('请先选择 ODB 文件。'); return; }

    const model = this._valModel;
    if (!model) { alert('模型信息丢失，请重新进入验证模块。'); return; }

    const matMeta = this._validation?.currentMatMeta || {};
    const matWsl = this.winToWsl(matPath);
    const odbWsl = this.winToWsl(odbPath);
    const modelWsl = this.winToWsl(model.path);
    const matStem = this._sanitizeRunToken(this._vizBasename(matPath).replace(/\.mat$/i, ''), 'MAT');
    const branchLabel = 'Elastic';
    const run = this._analysisRunPaths(
      model,
      'validation',
      `Validation_${matStem}_${branchLabel}`,
      'validation.template.cfg',
    );
    const cfgLines = [
      '# validation configuration (auto-generated by jaxmech web validation form)',
      `# run_dir: ${run.runLabel}`,
      '',
      'mat_files = [',
      `    "${matWsl}"`,
      ']',
      '',
      'odb_files = [',
      `    "${odbWsl}"`,
      ']',
      '',
      'copy_mat = 1',
      `validation_dir = ${run.runDirWsl}`,
    ];
    await this.apiPut(`/api/config/validation?model_path=${encodeURIComponent(run.runDirWin)}`, {
      content: cfgLines.join('\n'),
    });

    let module = 'jaxmech.modules.validation.elastic.run';
    let args = [
      '--config', run.cfgWsl,
      '--mat', matWsl,
      '--odb', odbWsl,
      '--model-root', modelWsl,
      '--validation-dir', run.runDirWsl,
      '--copy-mat',
    ];

    const result = await this.apiPost('/api/run', { module, args });
    this.viewTask(result.id);
  },



  // ──────────── Shakedown ────────────

  // Internal wizard state for shakedown
  _sd: {
    model: null,
    orderedMats: [],   // ordered absolute paths (Windows), max 3
    meta: null,
    allScanMats: [],   // from /api/models/scan-mats
  },

  async loadShakedown() {
    const container = document.getElementById('shakedown-content');
    if (!this.selectedModel) {
      container.innerHTML = UI.hint('warning', '请先在模型浏览中选择一个工作模型。', '前往模型浏览', "App.requestModelFor('shakedown')");
      return;
    }
    const { source, name } = this.selectedModel;
    container.innerHTML = '<div class="empty-state"><div class="empty-icon">⏳</div><div class="empty-text">加载中...</div></div>';

    const model = await this.api(`/api/models/${source}/${name}`).catch(() => null);
    if (!model) { container.innerHTML = UI.hint('error', '无法获取模型状态。'); return; }
    if (!model.inc_analysis.has_mat) {
      container.innerHTML = UI.hint('error', '未找到弹性分析结果（.mat 文件）。安定分析需要已有的 .mat 文件。', '前往弹性分析', "App.navigate('elastic')");
      return;
    }

    // Reset wizard state
    this._sd.model = model;
    this._sd.orderedMats = [];
    this._sd.meta = null;

    // Pre-scan all .mat files in model directory
    const scanRes = await this.apiPost('/api/models/scan-mats', { model_path: model.path }).catch(() => null);
    this._sd.allScanMats = scanRes?.mat_files || [];

    this._renderShakedownStep1(model);
  },

  _renderShakedownStep1(model) {
    const container = document.getElementById('shakedown-content');

    let html = '';
    // 轻量模型信息条（仅显示模型根目录）
    html += `<div style="font-size:13px;color:var(--text-secondary);margin-bottom:12px;padding:10px 14px;background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid rgba(255,255,255,0.06)">
      模型: <b style="color:var(--text-primary)">${UI.escapeHtml(model.name)}</b>
      <span style="color:rgba(255,255,255,0.25);margin:0 8px">|</span>
      <code style="font-size:11px;color:var(--accent)">${UI.escapeHtml(model.path)}</code>
    </div>`;

    html += `<div class="card" style="margin-bottom:16px">
      <div class="card-title">Step 1 &mdash; 选择 MAT 文件（有序，最多 3 个）</div>
      <div style="font-size:13px;color:var(--text-secondary);margin:10px 0 14px">
        按荷载工况顺序逐个添加 .mat 文件（MAT 排列顺序对分析结果有影响）。<b>必须选择当前模型目录内的正式增量/弹性结果 MAT</b>，最多可添加 3 个；临时 <code>inc_analysis/inputs</code> sidecar 不可用于安定分析。
      </div>
      <div id="sd-ordered-mat-list" style="display:flex;flex-direction:column;gap:8px;margin-bottom:12px">
      </div>
      <div style="display:flex;gap:10px;align-items:center">
        <button class="btn btn-secondary" style="padding:6px 14px;font-size:13px"
          id="sd-add-mat-btn" onclick="App._sdAddMat()">＋ 浏览添加 MAT 文件</button>
        <span style="font-size:12px;color:var(--text-secondary)" id="sd-mat-count-info">已选 0 个（最多 3 个）</span>
      </div>
      <div id="sd-mat-error" style="margin-top:10px"></div>
      <div style="margin-top:14px">
        <button class="btn btn-primary" onclick="App._confirmShakedownMats()">&#10003; 确认选择，展开配置</button>
      </div>
    </div>`;

    container.innerHTML = html;
    this._sdRenderOrderedMats();
  },

  _sdRenderOrderedMats() {
    const listEl = document.getElementById('sd-ordered-mat-list');
    const addBtn = document.getElementById('sd-add-mat-btn');
    const countInfo = document.getElementById('sd-mat-count-info');
    if (!listEl) return;
    const mats = this._sd.orderedMats;
    listEl.innerHTML = mats.map((absPath, i) => {
      const fname = absPath.split(/[\/\\]/).pop();
      const parts = absPath.split(/[\/\\]/);
      const pdir = parts.length >= 3 ? parts.slice(-3, -1).join('/') : parts.slice(0, -1).join('/');
      return `<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;background:rgba(255,255,255,0.04);border-radius:8px;border:1px solid rgba(255,255,255,0.07)">
        <span style="min-width:22px;font-size:13px;font-weight:700;color:var(--accent)">${i+1}.</span>
        <div style="flex:1;min-width:0">
          <div style="font-size:13px;font-family:monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${UI.escapeHtml(absPath)}">📦 ${UI.escapeHtml(fname)}</div>
          <div style="font-size:11px;color:var(--text-secondary);margin-top:2px">${UI.escapeHtml(pdir)}</div>
        </div>
        <button class="btn btn-secondary" style="padding:2px 10px;font-size:12px;flex-shrink:0" onclick="App._sdRemoveMat(${i})">移除</button>
      </div>`;
    }).join('');
    if (countInfo) countInfo.textContent = `已选 ${mats.length} 个（最多 3 个）`;
    if (addBtn) addBtn.disabled = mats.length >= 3;
  },

  async _sdAddMat() {
    if (this._sd.orderedMats.length >= 3) return;
    const modelPath = this._sd.model?.path || 'C:\\';
    const index = this._sd.orderedMats.length + 1;
    const errDiv = document.getElementById('sd-mat-error');
    const res = await this.apiPost('/api/browse', {
      field_type: 'mat_file',
      title: `选择第 ${index} 个 MAT 文件（模型目录: ${this._sd.model?.name}）`,
      initial_dir: modelPath,
    }).catch(() => null);
    if (!res || !res.path) return;

    // —— 校验是否属于当前模型目录 ——
    // Windows 路径比较：统一转为小写并标准化分隔符
    const normPath = (p) => p.replace(/\\/g, '/').toLowerCase();
    const selectedNorm = normPath(res.path);
    const modelNorm = normPath(modelPath);
    if (selectedNorm.includes('/inc_analysis/inputs/')) {
      if (errDiv) errDiv.innerHTML = UI.hint('error',
        '所选文件属于临时解析 sidecar，不是正式增量分析结果 MAT。<br>请改选 inc_analysis 根目录下的结果 MAT 文件。');
      return;
    }
    if (!selectedNorm.startsWith(modelNorm + '/') && selectedNorm !== modelNorm) {
      if (errDiv) errDiv.innerHTML = UI.hint('error',
        `所选文件不属于当前模型目录。` +
        `<br>当前模型: <code>${UI.escapeHtml(modelPath)}</code>` +
        `<br>所选文件: <code>${UI.escapeHtml(res.path)}</code>` +
        `<br>请仅选择当前模型目录下的 .mat 文件。`);
      return;
    }

    if (this._sd.orderedMats.includes(res.path)) {
      if (errDiv) errDiv.innerHTML = UI.hint('warning', `已添加该文件：${res.path.split(/[\/\\]/).pop()}`);
      return;
    }
    this._sd.orderedMats.push(res.path);
    this._sdRenderOrderedMats();
    if (errDiv) errDiv.innerHTML = '';
  },

  _sdRemoveMat(index) {
    this._sd.orderedMats.splice(index, 1);
    this._sdRenderOrderedMats();
  },

  async _confirmShakedownMats() {
    const errDiv = document.getElementById('sd-mat-error');
    if (this._sd.orderedMats.length === 0) {
      if (errDiv) errDiv.innerHTML = UI.hint('error', '请至少添加一个 MAT 文件。');
      return;
    }
    const paths = this._sd.orderedMats;

    if (errDiv) errDiv.innerHTML = UI.hint('info', '正在检查 MAT 字段...');
    const meta = await this.apiPost('/api/models/mat-meta', { paths }).catch(() => null);
    if (!meta) {
      if (errDiv) errDiv.innerHTML = UI.hint('error', 'MAT 元数据读取失败，请检查文件。');
      return;
    }
    if (!meta.has_required_fields) {
      const missing = meta.missing_fields.join(', ');
      if (errDiv) errDiv.innerHTML = UI.hint('error',
        `所选 MAT 缺少必要字段：${missing}。这些 MAT 可能不是完整的弹性分析结果，请重新选择。`);
      return;
    }
    // 验证 family 是否为 shell 或 solid
    const fam = (meta.family || '').trim().toLowerCase();
    if (fam !== 'shell' && fam !== 'solid') {
      if (errDiv) errDiv.innerHTML = UI.hint('error',
        `不支持的单元族 (family="${meta.family}")。当前仅支持 shell 和 solid，请重新选择。`);
      return;
    }
    if (fam !== 'solid') {
      if (errDiv) errDiv.innerHTML = UI.hint('warning', 'demo 版 shakedown 仅开放实体单元 C formulation / CVXPY 流程，请选择 solid elastic MAT。');
      return;
    }
    // Shell shakedown requires the personal edition (native_shell.py)
    if (fam === 'shell' && this._moduleAvail && this._moduleAvail.shakedown_shell === false) {
      if (errDiv) errDiv.innerHTML = UI.hint('error',
        '当前安装不包含壳单元安定分析模块（native_shell.py）。' +
        '壳单元安定分析仅在个人版（personal edition）中可用。' +
        '<br>实体单元安定分析可正常使用。');
      return;
    }
    // 检测 validation 同名 MAT 状态（非阻塞，仅警告）
    const allScanMats = Array.isArray(this._sd.allScanMats) ? this._sd.allScanMats : [];
    const validationMats = allScanMats.filter(m => {
      const subdir = String(m.subdir || '').toLowerCase().replace(/\\/g, '/');
      return subdir === 'validation' || subdir.startsWith('validation/');
    });
    const validationByName = new Map([...validationMats].sort((a, b) => Number(a.mtime || 0) - Number(b.mtime || 0)).map(m => [m.name, m]));
    let validationMetaByName = new Map();
    if (validationMats.length > 0) {
      const validationMetaRes = await this.apiPost('/api/models/mat-meta', {
        paths: validationMats.map(m => m.abs_path)
      }).catch(() => null);
      const mtimeByPath = new Map(validationMats.map(m => [m.abs_path, Number(m.mtime || 0)]));
      const details = validationMetaRes?.details || [];
      details.forEach(item => { item.mtime = mtimeByPath.get(item.path) || 0; });
      validationMetaByName = new Map([...details].sort((a, b) => Number(a.mtime || 0) - Number(b.mtime || 0)).map(d => [d.name, d]));
    }

    const validationDetails = paths.map((path) => {
      const name = path.split(/[\/\\]/).pop();
      const validationMat = validationByName.get(name);
      const validationMeta = validationMetaByName.get(name);
      let status = 'missing';
      if (validationMeta?.is_validated) status = 'validated';
      else if (validationMat && validationMeta?.has_validation_flag) status = 'flagged';
      else if (validationMat) status = 'exists';
      return { name, status };
    });

    meta.details = validationDetails;
    meta.is_validated = validationDetails.length > 0 && validationDetails.every(d => d.status === 'validated');

    if (errDiv && validationDetails.length > 0) {
      const statusLines = validationDetails.map(d => {
        const icon = d.status === 'validated' ? '✅' : d.status === 'exists' ? '📄' : d.status === 'flagged' ? '⚠️' : '○';
        const color = d.status === 'validated'
          ? 'var(--success,#22c55e)'
          : d.status === 'exists'
            ? 'var(--accent,#60a5fa)'
            : 'var(--warning,#f59e0b)';
        const label = d.status === 'validated'
          ? '已通过 ODB 验证'
          : d.status === 'exists'
            ? 'validation 目录已有同名 MAT'
            : d.status === 'flagged'
              ? 'validation MAT 标记未通过'
              : 'validation 目录无同名 MAT';
        return `<div style="display:flex;align-items:center;gap:8px;padding:4px 0">
          <span>${icon}</span>
          <span style="font-size:13px;flex:1">${UI.escapeHtml(d.name)}</span>
          <span style="font-size:12px;color:${color};font-weight:600">${label}</span>
        </div>`;
      }).join('');
      const allValidated = validationDetails.every(d => d.status === 'validated');
      const anyUnvalidated = validationDetails.some(d => d.status !== 'validated');
      errDiv.innerHTML = `<div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:12px 16px;
border:1px solid rgba(255,255,255,0.08);margin-bottom:8px">
        <div style="font-size:12px;font-weight:600;color:var(--text-secondary);margin-bottom:8px">MAT 验证状态</div>
        ${statusLines}
      </div>`;
      if (anyUnvalidated) {
        errDiv.innerHTML += UI.hint('warning',
          '部分 MAT 尚未做 ODB 对标。validation 仅用于结果检查，不是安定分析前置条件；当前仍可继续。',
          '前往验证', "App.navigate('validation')");
      }
    }
    this._sd.meta = meta;
    // 使用 setTimeout 使提示短暂可见后再切换到 Step2
    // 已验证: 300ms 快速过渡; 未验证: 1500ms 让用户看到警告
    setTimeout(() => this._renderShakedownStep2(), meta.is_validated ? 300 : 1500);
  },



  _renderShakedownStep2() {
    const container = document.getElementById('shakedown-content');
    const meta = this._sd.meta;
    const matCount = this._sd.orderedMats.length;
    const family = (meta.family || '').trim().toLowerCase();
    const isShell = family === 'shell';

    // NVert 上限：1mat→2, 2mat→4, 3mat→8
    const nvertMaxMap = { 1: 2, 2: 4, 3: 8 };
    const nvertMax = nvertMaxMap[matCount] || 2;
    const nvertOptions = [1, 2, 4, 8].filter(v => v <= nvertMax);

    const hint = (t) => `<span class="form-hint" style="margin-top:4px">${t}</span>`;

    let html = '';

    // ── Step 1 已完成摘要 ──
    const details = meta.details || [];
    const allValidated = details.length > 0 && details.every(d => d.status === 'validated');
    const anyUnvalidated = details.some(d => d.status !== 'validated');
    const globalBadge = allValidated
      ? `<span style="background:rgba(52,211,153,0.15);color:#34d399;border:1px solid rgba(52,211,153,0.3);border-radius:4px;padding:1px 8px;font-size:11px;font-weight:600">✓ 全部已验证</span>`
      : anyUnvalidated
        ? `<span style="background:rgba(251,191,36,0.15);color:#fbbf24;border:1px solid rgba(251,191,36,0.3);border-radius:4px;padding:1px 8px;font-size:11px;font-weight:600">⚠ 部分未验证</span>`
        : `<span style="background:rgba(251,191,36,0.15);color:#fbbf24;border:1px solid rgba(251,191,36,0.3);border-radius:4px;padding:1px 8px;font-size:11px;font-weight:600">⚠ 未验证</span>`;

    // Per-MAT validation rows
    const matRows = this._sd.orderedMats.map((p, i) => {
      const fname = p.split(/[\/\\]/).pop();
      const d = details[i];
      let badge = '';
      if (d) {
        badge = d.status === 'validated'
          ? `<span style="color:#34d399;font-size:11px;font-weight:600;white-space:nowrap">✓ 已验证</span>`
          : d.status === 'exists'
            ? `<span style="color:#60a5fa;font-size:11px;font-weight:600;white-space:nowrap">📄 validation 目录已有同名 MAT</span>`
            : d.status === 'flagged'
              ? `<span style="color:#f59e0b;font-size:11px;font-weight:600;white-space:nowrap">⚠ validation MAT 标记未通过</span>`
              : `<span style="color:var(--text-secondary);font-size:11px;font-weight:600;white-space:nowrap">○ validation 目录无同名 MAT</span>`;
      }
      return `<div style="display:flex;align-items:center;gap:8px">
        <span style="min-width:18px;font-size:12px;color:var(--accent);font-weight:700">${i+1}.</span>
        <span style="font-size:12px;font-family:monospace;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${UI.escapeHtml(p)}">📦 ${UI.escapeHtml(fname)}</span>
        ${badge}
      </div>`;
    }).join('');

    html += `<div class="card" style="margin-bottom:16px">
      <div class="card-title">Step 1 已完成 &mdash; MAT 选择</div>
      <div style="font-size:13px;color:var(--text-secondary);margin-top:8px;display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <span>已选 ${matCount} 个 MAT</span>
        <span style="color:var(--text-secondary)">|</span>
        <span>单元族: <b>${UI.escapeHtml(family.toUpperCase())}</b></span>
        <span style="color:var(--text-secondary)">|</span>
        ${globalBadge}
        <span style="color:var(--text-secondary)">|</span>
        <a href="#" style="color:var(--accent);text-decoration:none"
          onclick="App._renderShakedownStep1(App._sd.model);return false">重新选择</a>
      </div>
      <div style="margin-top:10px;display:flex;flex-direction:column;gap:4px">${matRows}</div>
      ${anyUnvalidated ? `<div style="font-size:12px;color:#fbbf24;margin-top:10px">⚠ 部分 MAT 未检测到 ODB 验证标记，建议先完成 <a href="#" onclick="App.navigate('validation');return false" style="color:#fbbf24">ODB 验证</a> 再进行安定分析。</div>` : ''}
    </div>`;


    // ── Step 2 主配置卡 ──
    html += `<div class="card" style="margin-bottom:16px">
      <div class="card-title">Step 2 &mdash; 配置安定分析参数</div>
      <div style="margin-top:16px;display:flex;flex-direction:column;gap:18px">`;

    // ── (1) Formulation ──
    if (isShell) {
      // Shell currently only supports VersionC — lock UI to C
      html += `<div class="form-group">
        <label class="form-label">单元公式 (formulation)</label>
        <input class="form-input" type="text" id="sd-formulation" value="C" readonly
          style="opacity:0.55;cursor:not-allowed" />
        ${hint('Shell 安定分析当前仅支持 VersionC（残余应力 ρ 变量），已自动锁定。')}
      </div>`;
    } else {
      const formulationOptions = [['C', 'C（残余应力 ρ 变量）']];
      html += `<div class="form-group">
        <label class="form-label">单元公式 (formulation)</label>
        <select class="form-input" id="sd-formulation">
          ${formulationOptions.map(([v,l]) => `<option value="${v}">${l}</option>`).join('')}
        </select>
        ${hint('C = 直接以残余应力 ρ 为变量；MN = 采用 legacy M/N 变量。')}
      </div>`;
    }


    // ── (2) Shell yield（仅 shell）──
    if (isShell) {
      html += `<div class="form-group">
        <label class="form-label">Shell 屈服面 (shell_yield)</label>
        <select class="form-input" id="sd-shell-yield" onchange="App._onShellYieldChange()">
          <option value="Ilyushin">Ilyushin（广义应力屈服面）</option>
          <option value="layer">Layer（层应力路线）</option>
        </select>
        ${hint('Ilyushin：广义应力屈服面；layer：对每层应力分别检验。')}
      </div>`;
      html += `<div class="form-group" id="sd-ilyushin-c-group">
        <label class="form-label">Ilyushin interaction parameter (ilyushin_c)</label>
        <input class="form-input" type="number" id="sd-ilyushin-c" value="0.5" min="0" max="1" step="0.0001" />
        ${hint('\\(\\varphi = \\varphi_N + \\varphi_M + 2c|\\varphi_{NM}|\\)，c=0.9999 近乎纯 Ilyushin。')}
      </div>`;
    }

    // ── (3) Solver ──
    if (isShell) {
      html += `<div class="form-group">
        <label class="form-label">求解器 (solver)</label>
        <input type="hidden" id="sd-solver" value="gurobi" />
        <input class="form-input" type="text" value="Gurobi（Windows 侧）" readonly
          style="opacity:0.72;cursor:not-allowed" />
        ${hint('Shell shakedown 当前发布入口固定使用 Gurobi；CVXPY 仅保留为手写 cfg / CLI 对照路径。')}
      </div>`;
    } else {
      html += `<div class="form-group">
        <label class="form-label">求解器 (solver)</label>
        <input type="hidden" id="sd-solver" value="cvxpy" />
        <input class="form-input" type="text" value="CVXPY" readonly
          style="opacity:0.72;cursor:not-allowed" />
        ${hint('demo 版固定使用 CVXPY backend。')}
      </div>`;
    }

    // (3b) Gurobi 配置
    html += `<div id="sd-gurobi-extra" style="display:${isShell ? 'flex' : 'none'};flex-direction:column;gap:12px;
        padding:14px 16px;background:rgba(255,255,255,0.04);border-radius:10px;border:1px solid rgba(255,255,255,0.08)">
      <div style="font-size:12px;font-weight:600;color:var(--text-secondary)">Gurobi 配置</div>
      <div class="form-group" style="margin:0">
        <label class="form-label">Backend（调用方式）</label>
        <select class="form-input" id="sd-gurobi-backend">
          <option value="python" selected>Python（推荐，通过 gurobipy 调用）</option>
          <option value="matlab">MATLAB（通过 MATLAB Gurobi 接口调用）</option>
        </select>
        ${hint('python = 使用 gurobipy API（需已安装 gurobipy）；matlab = 通过 MATLAB 调用 Gurobi。')}
      </div>
      <div class="form-group" style="margin:0">
        <label class="form-label">Gurobi 安装根目录 (windows_gurobi_root)</label>
        <input class="form-input" id="sd-gurobi-root" value="${UI.escapeAttr(this._sd.gurobiRoot || '')}" placeholder="正在自动检测 C:\\gurobi*..." />
        ${hint('留空时会自动扫描 C:\\ 根目录下的 gurobi* 并选择版本号最小的安装目录；也可直接填写具体安装根目录。程序会自动补 \\win64，并扫描 win64\\pythonXY\\lib\\gurobipy。')}
      </div>
      <div class="form-group" style="margin:0">
        <label class="form-label">Gurobi Python (windows_gurobi_python_exe)</label>
        <input class="form-input" id="sd-gurobi-python-exe" placeholder="C:\\Users\\...\\.virtualenvs\\jaxmech-gurobi311\\Scripts\\python.exe" />
        ${hint('可选。用于指定与 gurobipy binding 兼容的 Windows Python；留空时会优先使用 Gurobi 安装目录中扫描到的最高版本 pythonXY binding。')}
      </div>
      <div class="form-group" style="margin:0">
        <label class="form-label">用户名 (windows_gurobi_user)</label>
        <input class="form-input" id="sd-gurobi-user" placeholder="Academic 许可证用户名（可选）" />
      </div>
      <div class="form-group" style="margin:0">
        <label class="form-label">密码 (windows_gurobi_password)</label>
        <input class="form-input" type="password" id="sd-gurobi-pass" placeholder="Academic 许可证密码（可选）" />
      </div>
      <div style="font-size:12px;font-weight:600;color:var(--text-secondary);margin-top:4px">Gurobi 求解参数</div>
      <div style="display:flex;gap:12px;flex-wrap:wrap">
        <div class="form-group" style="margin:0;flex:1;min-width:140px">
          <label class="form-label">TimeLimit（秒）</label>
          <input class="form-input" type="number" id="sd-gurobi-timelimit" value="3600" min="1" />
        </div>
        <div class="form-group" style="margin:0;flex:1;min-width:180px">
          <label class="form-label">FeasibilityTol</label>
          <input class="form-input" type="number" id="sd-gurobi-feastol" value="0.000001" min="1e-9" max="1e-2" step="any" />
        </div>
      </div>
      <label style="display:flex;align-items:flex-start;gap:8px;font-size:13px;color:var(--text-secondary);line-height:1.45">
        <input type="checkbox" id="sd-gurobi-reconstruct-duals" checked
          onchange="App._onGurobiDualReconstructionChange()"
          style="width:16px;height:16px;accent-color:var(--accent-blue);margin-top:2px" />
        <span>尝试重建 Gurobi 未返回的 dual variables。默认开启；关闭后只快速写出 primal result MAT。</span>
      </label>
      <label id="sd-gurobi-unlimited-duals-row" style="display:flex;align-items:flex-start;gap:8px;font-size:13px;color:var(--text-secondary);line-height:1.45">
        <input type="checkbox" id="sd-gurobi-unlimited-duals"
          style="width:16px;height:16px;accent-color:var(--accent-blue);margin-top:2px" />
        <span>Dual reconstruction 不限时。默认会尝试重建 Gurobi 未返回的 dual variables，但若耗时超过原 Gurobi solve runtime，则跳过 dual 并继续写出 result MAT。</span>
      </label>
    </div>`;

    // ── (4) NVert ──
    html += `<div class="form-group">
      <label class="form-label">载荷角点数 (num_vert)
        <span style="color:var(--text-secondary);font-size:12px">（已选 ${matCount} 个 MAT，最多可选 ${nvertMax}）</span>
      </label>
      <select class="form-input" id="sd-nvert" onchange="App._onNVertChange()">
        ${nvertOptions.map(v => `<option value="${v}">${v}</option>`).join('')}
      </select>
      ${hint(`1 MAT: NVert=1 或 2，无载荷组合；多 MAT: \\(N_{\\rm vert}\\) 个顶点均为多 MAT 的线性组合。<br>1=只取最大顶点；2=最大/最小；4=2载荷全角点；8=3载荷全角点。`)}
    </div>`;

    // ── (5) R_ratios ──
    const rRatioInputs = this._sd.orderedMats.map((p, i) => {
      const fname = p.split(/[\/\\]/).pop();
      return `<div class="form-group" style="margin:0;flex:1;min-width:160px">
        <label class="form-label">\\(\\sigma^E_{${i+1}}\\)
          <span style="font-size:11px;color:var(--text-secondary);margin-left:4px">(${UI.escapeHtml(fname)})</span>
        </label>
        <label class="form-label" style="font-size:11px;color:var(--text-secondary);margin-top:2px">\\(R_{${i+1}}\\) 应力比</label>
        <input class="form-input" type="number" id="sd-rratio-${i}" value="0" min="-1" max="1" step="0.01"
          placeholder="0" title="${UI.escapeHtml(fname)}" />
      </div>`;
    }).join('');
    html += `<div class="form-group">
      <label class="form-label">\\(R_n\\) 应力比（最小/最大弹性应力比）</label>
      <div style="display:flex;gap:10px;flex-wrap:wrap">${rRatioInputs}</div>
      ${hint('\\(R_n = \\sigma^E_{\\min} / \\sigma^E_{\\max}\\)。默认 0 表示无反向加载；-1 表示完全反向。与 MAT 文件一一对应。')}
    </div>`;

    // ── (6) yield_values ──
    html += `<div class="form-group">
      <label class="form-label">屈服强度 (yield_values)</label>
      <div style="font-size:12px;color:var(--text-secondary);margin-bottom:8px">每行输入一种材料的屈服应力（MPa），材料名需与 INP 定义一致。</div>
      <div id="sd-yield-rows" style="display:flex;flex-direction:column;gap:6px;margin-bottom:8px">
        <div class="sd-yield-row" style="display:flex;gap:8px;align-items:center">
          <input class="form-input sd-yield-name" style="flex:1;min-width:120px" value="MATERIAL-1" placeholder="材料名" />
          <input class="form-input sd-yield-val" type="number" style="width:100px" value="280" min="0" step="1" placeholder="σ_y (MPa)" />
          <button class="btn btn-secondary" style="padding:2px 8px;font-size:13px" onclick="this.closest('.sd-yield-row').remove()">－</button>
        </div>
      </div>
      <button class="btn btn-secondary" style="padding:4px 12px;font-size:12px" onclick="App._addYieldRow()">＋ 添加材料</button>
      ${hint("填写各材料名与屈服应力（MPa），最终生成 {'材料名': σ_y, ...} 格式。")}
    </div>`;

    // ── (7) 载荷组合定义（仅 matCount >= 2 时显示）──
    if (matCount >= 2) {
      html += `<div class="form-group">
        <label class="form-label">载荷顶点定义方式 (use_angles)</label>
        <select class="form-input" id="sd-use-angle" onchange="App._onUseAngleChange()">
          <option value="yes" selected>是（按方位角自动生成顶点）</option>
          <option value="no">否（手动填写 load_factor_set 矩阵）</option>
        </select>
        ${hint('选"是"时按方位角生成各顶点的荷载系数；选"否"时手工填写各顶点的荷载系数矩阵。')}
      </div>`;

      // 角度区（use_angles=yes）—— theta/phi 纵向列布局
      const hasPhi = matCount >= 3;
      html += `<div id="sd-angles-section" class="sd-angles-section" style="display:none">
        <div style="font-size:12px;font-weight:600;color:var(--text-secondary)">角度列表（单位：degree °）</div>
        <div class="sd-angle-grid">
          ${hasPhi ? `
          <div id="sd-phi-group" class="form-group sd-angle-group">
            <label class="form-label">\\(\\varphi\\)（仰角）</label>
            <div id="sd-phi-list" class="sd-angle-list">
              <div class="sd-angle-item">
                <span class="sd-angle-index">1.</span>
                <input class="form-input sd-angle-input sd-phi-item" type="number" step="any" value="0" placeholder="° (deg)"/>
              </div>
            </div>
            <button class="btn btn-secondary sd-angle-add"
              onclick="App._addAngle('sd-phi-list','sd-phi-item')">＋ 添加 \\(\\varphi\\)</button>
            ${hint('\\(a_1=\\cos\\varphi\\cos\\theta,\\; a_2=\\cos\\varphi\\sin\\theta,\\; a_3=\\sin\\varphi\\)。单位 degree。')}
          </div>` : ''}
          <div id="sd-theta-group" class="form-group sd-angle-group">
            <label class="form-label">\\(\\theta\\)（方位角）</label>
            <div id="sd-theta-list" class="sd-angle-list">
              <div class="sd-angle-item">
                <span class="sd-angle-index">1.</span>
                <input class="form-input sd-angle-input sd-theta-item" type="number" step="any" value="0" placeholder="° (deg)"/>
              </div>
            </div>
            <button class="btn btn-secondary sd-angle-add"
              onclick="App._addAngle('sd-theta-list','sd-theta-item')">＋ 添加 \\(\\theta\\)</button>
            ${hint('1 载荷：\\(a_1=\\cos\\theta,\\; a_2=\\sin\\theta\\)。单位 degree。')}
          </div>
        </div>
      </div>`;

      // load_factor_set 表格（use_angles=no）
      const colCount = matCount;
      const lfsColHeaders = this._sd.orderedMats.slice(0, colCount).map((p, i) => {
        const fname = p.split(/[\/\\]/).pop();
        return `\\(\\sigma^E_{${i+1}}\\) <span style="font-size:10px;color:var(--text-secondary)">(${UI.escapeHtml(fname)})</span>`;
      });
      html += `<div id="sd-lfs-section" style="display:none;flex-direction:column;gap:12px;
          padding:14px 16px;background:rgba(255,255,255,0.04);border-radius:10px;border:1px solid rgba(255,255,255,0.08)">
        <div style="font-size:12px;font-weight:600;color:var(--text-secondary)">load_factor_set（荷载比例系数矩阵）</div>
        ${hint(`每行对应一个分析顶点，填写各工况的荷载系数（${colCount} 列对应 ${colCount} 个 MAT）。建议每行平方和 = 1。`)}
        <table id="sd-lfs-table" style="border-collapse:collapse;width:100%;margin-top:4px">
          <thead>
            <tr>
              ${lfsColHeaders.map(h => `<th style="font-size:12px;color:var(--text-secondary);padding:4px 8px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.1)">${h}</th>`).join('')}
              <th style="width:36px"></th>
            </tr>
          </thead>
          <tbody id="sd-lfs-body">
            <tr class="sd-lfs-row">
              ${Array(colCount).fill(0).map(() => `<td style="padding:4px 6px"><input type="number" step="any" class="form-input sd-lfs-cell" style="width:100%;padding:6px 8px" value="1.0" /></td>`).join('')}
              <td style="padding:4px 6px"><button class="btn btn-secondary" style="padding:0 8px;height:30px;font-size:14px" onclick="this.closest('tr').remove()">－</button></td>
            </tr>
          </tbody>
        </table>
        <div style="display:flex;gap:8px;margin-top:4px">
          <button class="btn btn-secondary" style="padding:6px 14px;font-size:13px" onclick="App._addLfsRow(${colCount})">＋ 添加行</button>
          <button class="btn btn-secondary" style="padding:6px 14px;font-size:13px" onclick="App._normalizeLfs()">⊙ 归一化</button>
        </div>
      </div>`;
    }

    // ── (8) 执行步骤 ──
    html += `<div class="form-group" id="sd-step-group" style="display:${isShell ? 'flex' : 'none'};flex-direction:column">
      <label class="form-label">执行步骤</label>
      <select class="form-input" id="shakedown-step">
        <option value="all">all — 完整执行</option>
        <option value="prepare">prepare — 仅准备（Gurobi 模式）</option>
        <option value="collect">collect — 仅收集（Gurobi 模式）</option>
      </select>
    </div>`;

    html += `</div></div>`; // end config card

    html += `<div style="display:flex;gap:12px;margin-top:4px">
      <button class="btn btn-primary" onclick="App.saveAndRunShakedown()">&#9654; 保存配置并运行安定分析</button>
    </div>`;

    container.innerHTML = html;

    // Initialize form linkages.
    this._onNVertChange();
    this._onSolverChange();
    this._onShellYieldChange();
    this._onGurobiDualReconstructionChange();
    this._autofillShakedownGurobiRoot();

    this._typesetMath(container);
  },




  _onShellYieldChange() {
    const val = document.getElementById('sd-shell-yield')?.value;
    const cGroup = document.getElementById('sd-ilyushin-c-group');
    if (cGroup) cGroup.style.display = val === 'Ilyushin' ? 'flex' : 'none';
    if (cGroup) cGroup.style.flexDirection = 'column';
  },

  _onSolverChange() {
    const solver = document.getElementById('sd-solver')?.value;
    const extra = document.getElementById('sd-gurobi-extra');
    if (extra) { extra.style.display = solver === 'gurobi' ? 'flex' : 'none'; extra.style.flexDirection = 'column'; }
    const stepGroup = document.getElementById('sd-step-group');
    if (stepGroup) { stepGroup.style.display = solver === 'gurobi' ? 'flex' : 'none'; stepGroup.style.flexDirection = 'column'; }
    this._onGurobiDualReconstructionChange();
    if (solver === 'gurobi') this._autofillShakedownGurobiRoot();
  },

  _onGurobiDualReconstructionChange() {
    const reconstruct = document.getElementById('sd-gurobi-reconstruct-duals');
    const unlimited = document.getElementById('sd-gurobi-unlimited-duals');
    const row = document.getElementById('sd-gurobi-unlimited-duals-row');
    const enabled = reconstruct ? reconstruct.checked : true;
    if (unlimited) {
      unlimited.disabled = !enabled;
      if (!enabled) unlimited.checked = false;
    }
    if (row) {
      row.style.opacity = enabled ? '1' : '0.48';
      row.style.pointerEvents = enabled ? '' : 'none';
    }
  },

  async _autofillShakedownGurobiRoot() {
    const input = document.getElementById('sd-gurobi-root');
    if (!input || input.value.trim()) return;
    const res = await this.api('/api/status/detect-gurobi-root').catch(() => null);
    if (!res?.found || !res.root) return;
    if (!input.value.trim()) {
      input.value = res.root;
      this._sd.gurobiRoot = res.root;
    }
  },

  _onNVertChange() {
    const matCount = this._sd.orderedMats.length;
    const useAngleSel = document.getElementById('sd-use-angle');
    const anglesSection = document.getElementById('sd-angles-section');
    const lfsSection = document.getElementById('sd-lfs-section');

    if (matCount <= 1) {
      // 单 MAT：无载荷组合，隐藏整个载荷定义区
      if (anglesSection) anglesSection.style.display = 'none';
      if (lfsSection) lfsSection.style.display = 'none';
    } else {
      // 多 MAT：根据 use_angles 切换
      const useAngle = useAngleSel?.value === 'yes';
      if (!useAngle) {
        if (anglesSection) anglesSection.style.display = 'none';
        if (lfsSection) { lfsSection.style.display = 'flex'; lfsSection.style.flexDirection = 'column'; }
      } else {
        if (anglesSection) { anglesSection.style.display = 'flex'; anglesSection.style.flexDirection = 'column'; }
        if (lfsSection) lfsSection.style.display = 'none';
      }
    }
  },

  _onUseAngleChange() {
    this._onNVertChange();
  },



  _addAngle(listId, itemClass) {
    const list = document.getElementById(listId);
    if (!list) return;
    const count = list.querySelectorAll('.' + itemClass).length + 1;
    const wrapper = document.createElement('div');
    wrapper.className = 'sd-angle-item';
    const lbl = document.createElement('span');
    lbl.className = 'sd-angle-index';
    lbl.textContent = count + '.';
    const inp = document.createElement('input');
    inp.type = 'number'; inp.step = 'any'; inp.value = '0';
    inp.placeholder = '° (deg)';
    inp.className = 'form-input sd-angle-input ' + itemClass;
    wrapper.appendChild(lbl); wrapper.appendChild(inp);
    list.appendChild(wrapper);
  },

  _addLfsRow(colCount) {
    const tbody = document.getElementById('sd-lfs-body');
    if (!tbody) return;
    const tr = document.createElement('tr');
    tr.className = 'sd-lfs-row';
    tr.innerHTML = Array(colCount).fill(0).map(() =>
      `<td style="padding:4px 6px"><input type="number" step="any" class="form-input sd-lfs-cell" style="width:100%;padding:6px 8px" value="0.0" /></td>`
    ).join('') + `<td style="padding:4px 6px"><button class="btn btn-secondary" style="padding:0 8px;height:30px;font-size:14px" onclick="this.closest('tr').remove()">－</button></td>`;
    tbody.appendChild(tr);
  },

  _normalizeLfs() {
    const rows = document.querySelectorAll('#sd-lfs-body .sd-lfs-row');
    rows.forEach(row => {
      const cells = row.querySelectorAll('.sd-lfs-cell');
      const vals = [...cells].map(c => parseFloat(c.value) || 0);
      const norm = Math.sqrt(vals.reduce((s, v) => s + v * v, 0));
      if (norm > 1e-12) {
        cells.forEach((c, i) => { c.value = (vals[i] / norm).toFixed(8); });
      }
    });
  },

  _addYieldRow() {
    const container = document.getElementById('sd-yield-rows');
    if (!container) return;
    const row = document.createElement('div');
    row.className = 'sd-yield-row';
    row.style.cssText = 'display:flex;gap:8px;align-items:center';
    row.innerHTML = '<input class="form-input sd-yield-name" style="flex:1;min-width:120px" placeholder="材料名" />' +
      '<input class="form-input sd-yield-val" type="number" style="width:100px" value="280" min="0" step="1" placeholder="σ_y (MPa)" />' +
      '<button class="btn btn-secondary" style="padding:2px 8px;font-size:13px" onclick="this.closest(\'.sd-yield-row\').remove()">－</button>';
    container.appendChild(row);
  },

  _collectYieldValues() {
    // 从表单行收集 yield_values，生成 {'MAT1': 280, ...} 字符串
    const rows = document.querySelectorAll('#sd-yield-rows .sd-yield-row');
    const pairs = [...rows].map(row => {
      const name = row.querySelector('.sd-yield-name')?.value?.trim() || '';
      const val = parseFloat(row.querySelector('.sd-yield-val')?.value) || 0;
      return name ? `'${name}': ${val}` : null;
    }).filter(Boolean);
    return '{' + pairs.join(', ') + '}';
  },

  async saveAndRunShakedown() {
    const model = this._sd.model;
    if (!model) return;

    const winToWsl = (p) => p.replace(/\\/g, '/').replace(/^([A-Za-z]):\//,  (_, d) => `/mnt/${d.toLowerCase()}/`);
    const modelRoot = winToWsl(model.path);
    // 使用有序 absolutePaths，直接转WSL路径
    const matPaths = this._sd.orderedMats.map(absPath => `"${winToWsl(absPath)}"`);

    const formulation = document.getElementById('sd-formulation')?.value || 'C';
    const shellYield = document.getElementById('sd-shell-yield')?.value || 'Ilyushin';
    const ilyushinC = document.getElementById('sd-ilyushin-c')?.value || '0.5';
    const solver = document.getElementById('sd-solver')?.value || 'cvxpy';
    const nvert = parseInt(document.getElementById('sd-nvert')?.value || '4');
    const useAngle = document.getElementById('sd-use-angle')?.value === 'yes';
    const step = document.getElementById('shakedown-step')?.value || 'all';
    const yieldValues = this._collectYieldValues() || "{'MATERIAL-1': 280}";

    // R_ratios
    const matCount = this._sd.orderedMats.length;
    const rRatios = Array.from({length: matCount}, (_, i) =>
      parseFloat(document.getElementById(`sd-rratio-${i}`)?.value || '0')
    );

    // 生成时间戳子目录名: yyyyMMDD_HHmmss
    const now = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    const tsDir = `${now.getFullYear()}${pad(now.getMonth()+1)}${pad(now.getDate())}` +
                  `_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    // 生成 solver variant 后缀：包含 family/yield/formulation/solver
    const isShellFamily = (this._sd.meta?.family || '').trim().toLowerCase() === 'shell';
    let solverVariant;
    if (isShellFamily) {
      // Shell：包含 Ilyushin 或 Layer
      const yieldLabel = shellYield === 'Ilyushin' ? 'Ilyushin' : 'Layer';
      const fmtLabel = formulation === 'Ext' ? 'VersionCExt' : 'VersionC';
      const gurobiSuffix = solver === 'gurobi' ? '_Gurobi' : '';
      solverVariant = `${yieldLabel}_${fmtLabel}${gurobiSuffix}`;
    } else {
      // Solid
      const solverVariantMap = {
        'cvxpy':  formulation === 'MN' ? 'VersionMN' : 'VersionC',
        'gurobi': formulation === 'MN' ? 'VersionMN_Gurobi' : 'VersionC_Gurobi',
      };
      solverVariant = solverVariantMap[solver] || solver;
    }
    const runLabel = `${tsDir}_${solverVariant}_${nvert}P`;
    // Windows 路径（用于保存配置操作）
    const sdSubWin = `${model.path}\\shakedown\\${runLabel}`;
    // WSL 路径（用于传递给 run 命令）
    const sdSubWsl = `${modelRoot}/shakedown/${runLabel}`;

    const lines = [
      '# shakedown configuration (auto-generated by jaxmech web)',
      `# run_dir: ${tsDir}`,
      '',
      `mat_files = [\n  ${matPaths.join(',\n  ')}\n]`,
      '',
      `formulation = ${formulation}`,
    ];

    // shell_yield + ilyushin_c（只在 shell 时写入）
    const family = (this._sd.meta?.family || '').trim().toLowerCase();
    if (family === 'shell') {
      lines.push(`shell_yield = ${shellYield}`);
      if (shellYield === 'Ilyushin') lines.push(`ilyushin_c = ${ilyushinC}`);
    }
    lines.push('');

    // solver & backend
    lines.push(`solver = ${solver}`);
    if (solver === 'gurobi') {
      const gurobiRoot = document.getElementById('sd-gurobi-root')?.value || '';
      const gurobiPythonExe = document.getElementById('sd-gurobi-python-exe')?.value || '';
      const gurobiUser = document.getElementById('sd-gurobi-user')?.value || '';
      const gurobiPass = document.getElementById('sd-gurobi-pass')?.value || '';
      const timeLimit = document.getElementById('sd-gurobi-timelimit')?.value || '3600';
      const feasTol = document.getElementById('sd-gurobi-feastol')?.value || '1e-6';
      const gurobiBackend = document.getElementById('sd-gurobi-backend')?.value || 'python';
      const reconstructDuals = document.getElementById('sd-gurobi-reconstruct-duals')?.checked ?? true;
      const unlimitedDuals = reconstructDuals && document.getElementById('sd-gurobi-unlimited-duals')?.checked;
      lines.push(`gurobi_backend = ${gurobiBackend}`);
      if (gurobiRoot) lines.push(`windows_gurobi_root = ${gurobiRoot}`);
      if (gurobiPythonExe) lines.push(`windows_gurobi_python_exe = ${gurobiPythonExe}`);
      if (gurobiUser) lines.push(`windows_gurobi_user = ${gurobiUser}`);
      if (gurobiPass) lines.push(`windows_gurobi_password = ${gurobiPass}`);
      lines.push(`gurobi_params = {'TimeLimit': ${timeLimit}, 'FeasibilityTol': ${feasTol}}`);
      lines.push(`reconstruct_gurobi_duals = ${reconstructDuals ? 'yes' : 'no'}`);
      lines.push(`unlimited_gurobi_dual_reconstruction = ${unlimitedDuals ? 'yes' : 'no'}`);
    }
    lines.push('');

    // R_ratios & yield_values
    lines.push(`R_ratios = [${rRatios.join(', ')}]`);
    lines.push(`yield_values = ${yieldValues}`);
    lines.push('');

    // num_vert & load definition
    lines.push(`num_vert = ${nvert}`);
    if (matCount <= 1) {
      // 单 MAT：无载荷组合，不写 use_angles / load_factor_set（后端默认 use_angles=yes）
    } else if (useAngle) {
      lines.push('use_angles = yes');
      // phi: matCount >= 3 时（3个载荷）使用仰角
      if (matCount >= 3) {
        const phis = [...document.querySelectorAll('.sd-phi-item')].map(i => parseFloat(i.value) || 0);
        if (phis.length) lines.push(`phi_deg = [${phis.join(', ')}]`);
      }
      // theta: 多 MAT 始终用方位角
      const thetas = [...document.querySelectorAll('.sd-theta-item')].map(i => parseFloat(i.value) || 0);
      if (thetas.length) lines.push(`theta_deg = [${thetas.join(', ')}]`);
    } else {
      // 多 MAT 手动填矩阵
      lines.push('use_angles = no');
      const rows = document.querySelectorAll('#sd-lfs-body .sd-lfs-row');
      const lfsRows = [...rows].map(row => {
        const cells = [...row.querySelectorAll('.sd-lfs-cell')].map(c => parseFloat(c.value) || 0);
        return `    [${cells.join(', ')}]`;
      });
      if (lfsRows.length) {
        lines.push(`load_factor_set = [\n${lfsRows.join(',\n')}\n]`);
      } else {
        // 保护：没有行时补一个全1行，避免后端报错
        lines.push(`load_factor_set = [[${Array(matCount).fill('1.0').join(', ')}]]`);
      }
    }
    lines.push('');




    const content = lines.join('\n');
    // 将配置写入子目录（需要后端支持自动创建该子目录）
    await this.apiPut(`/api/config/shakedown?module=shakedown&model_path=${encodeURIComponent(sdSubWin)}`, { content });

    // parse_shakedown_config 需要 .cfg 文件路径，而非目录
    const cfgFileName = 'shakedown_analysis.template.cfg';
    const args = ['--config', `${sdSubWsl}/${cfgFileName}`];
    if (step !== 'all') args.push('--step', step);

    const result = await this.apiPost('/api/run', {
      module: 'jaxmech.modules.shakedown.run',
      args,
    });
    this.viewTask(result.id);
  },



  // ──────────── Tasks ────────────
  _taskArtifactUrl(path) {
    return `/api/files?path=${encodeURIComponent(path)}`;
  },

  async openTaskFolder(taskId) {
    const res = await this.apiPost(`/api/tasks/${encodeURIComponent(taskId)}/open-folder`, {}).catch((err) => ({ error: err?.message || '打开失败' }));
    if (!res || res.error || res.detail) {
      alert(`无法打开任务文件夹：${res?.detail || res?.error || '未知错误'}`);
    }
  },

  async stopTask(taskId) {
    const ok = window.confirm('确定停止这个任务吗？已写出的 MAT/artifact 会保留，但不是所有 solver 都能从中间迭代状态生成新的 MAT。');
    if (!ok) return;
    const res = await this.apiPost(`/api/tasks/${encodeURIComponent(taskId)}/stop`, {}).catch((err) => ({ error: err?.message || '停止失败' }));
    if (!res || res.error || res.detail) {
      alert(`停止任务失败：${res?.detail || res?.error || '未知错误'}`);
      return;
    }
    const btn = document.getElementById(`task-stop-${taskId}`);
    if (btn) {
      btn.disabled = true;
      btn.textContent = '停止中...';
    }
  },

  async openArtifactVisualization(path) {
    const target = String(path || '').trim();
    if (!target) return;
    const exists = await this.api(`/api/files/exists?path=${encodeURIComponent(target)}`).catch((err) => ({ exists: false, detail: err?.message || '' }));
    if (!exists?.exists) {
      alert(`找不到可视化 MAT 文件，可能已移动到其它计算机或被手动删除。\n\n${target}`);
      return;
    }
    this._vizSelectedPath = target;
    this.navigate('visualization');
  },

  async openArtifactFolder(path) {
    const target = String(path || '').trim();
    if (!target) return;
    const res = await this.apiPost(`/api/files/open-folder?path=${encodeURIComponent(target)}`, {}).catch((err) => ({ error: err?.message || '打开失败' }));
    if (!res || res.error || res.detail) {
      alert(`无法打开结果文件夹：${res?.detail || res?.error || '未知错误'}\n\n${target}`);
    }
  },

  _renderTaskArtifacts(artifacts) {
    if (!Array.isArray(artifacts) || artifacts.length === 0) return '';
    const cards = artifacts.map((artifact) => {
      const label = UI.escapeHtml(artifact.label || artifact.path || 'artifact');
      const path = UI.escapeHtml(artifact.path || '');
      const url = this._taskArtifactUrl(artifact.path || '');
      const statusFlag = String(artifact.status_flag || '').toLowerCase();
      const statusBadge = statusFlag
        ? `<span style="font-size:11px;padding:2px 8px;border-radius:99px;background:${statusFlag === 'completed' ? 'var(--accent-green-glow)' : 'var(--accent-amber-glow)'};color:${statusFlag === 'completed' ? 'var(--accent-green)' : 'var(--accent-amber)'}">${statusFlag}</span>`
        : '';
      if ((artifact.kind || '').toLowerCase() === 'image') {
        const rawLabel = String(artifact.label || artifact.path || '');
        const showImageFolderButton = !/validation\s+boxplot/i.test(rawLabel);
        return `<div style="display:flex;flex-direction:column;gap:10px;padding:14px;border-radius:10px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08)">
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><span style="font-size:13px;font-weight:600">${label}</span>${statusBadge}</div>
          <img src="${url}" alt="${label}" style="width:100%;max-height:480px;object-fit:contain;border-radius:8px;background:rgba(15,23,42,0.35);border:1px solid rgba(255,255,255,0.06)" />
          ${showImageFolderButton ? `<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
            <button class="btn btn-secondary" style="padding:6px 10px;font-size:12px" onclick="App.openArtifactFolder('${UI.escapeAttr(artifact.path || '')}')">打开所在文件夹</button>
          </div>` : ''}
          <div style="font-size:11px;color:var(--text-secondary);font-family:monospace;word-break:break-all">${path}</div>
        </div>`;
      }
      const lowerKind = (artifact.kind || '').toLowerCase();
      const lowerPath = String(artifact.path || '').toLowerCase();
      const isMat = lowerPath.endsWith('.mat') || lowerKind.includes('mat');
      const vizButton = isMat
        ? `<button class="btn btn-secondary" style="padding:6px 10px;font-size:12px" onclick="App.openArtifactVisualization('${UI.escapeAttr(artifact.path || '')}')">打开 visualization</button>`
        : '';
      return `<div style="padding:14px;border-radius:10px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08)">
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px"><span style="font-size:13px;font-weight:600">${label}</span>${statusBadge}</div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <button class="btn btn-secondary" style="padding:6px 10px;font-size:12px" onclick="App.openArtifactFolder('${UI.escapeAttr(artifact.path || '')}')">打开所在文件夹</button>
          ${vizButton}
        </div>
        <div style="font-size:11px;color:var(--text-secondary);font-family:monospace;word-break:break-all;margin-top:8px">${path}</div>
      </div>`;
    }).join('');

    return `<div class="card" style="margin-bottom:16px">
      <div class="card-title" style="margin-bottom:12px">结果产物</div>
      <div style="display:flex;flex-direction:column;gap:12px">${cards}</div>
    </div>`;
  },

  _renderTaskConfig(config) {
    const cfg = config || {};
    const path = String(cfg.path || '').trim();
    const content = String(cfg.content || '');
    if (!path) return '';
    const exists = String(cfg.exists || '').toLowerCase() === 'true';
    const body = exists && content
      ? UI.escapeHtml(content)
      : UI.escapeHtml('未找到配置文件或配置文件为空。');
    return `<div class="card" style="margin-bottom:16px">
      <div class="card-header" style="margin-bottom:12px">
        <span class="card-title">运行配置</span>
        <span style="font-size:11px;padding:2px 8px;border-radius:99px;background:${exists ? 'var(--accent-green-glow)' : 'var(--accent-red-glow)'};color:${exists ? 'var(--accent-green)' : 'var(--accent-red)'}">${exists ? 'cfg' : 'missing'}</span>
      </div>
      <div style="font-size:11px;color:var(--text-secondary);font-family:monospace;word-break:break-all;margin-bottom:8px">${UI.escapeHtml(path)}</div>
      <pre class="log-console" style="max-height:260px;margin:0">${body}</pre>
    </div>`;
  },

  async loadTasks() {
    const container = document.getElementById('tasks-content');
    const tasks = await this.api('/api/tasks?limit=500').catch(() => []);
    this.tasks = tasks;

    if (tasks.length === 0) {
      container.innerHTML = UI.empty('📋', '暂无任务记录');
      return;
    }
    container.innerHTML = '<div class="card" style="margin-bottom:12px"><div class="card-title">任务记录</div>' +
      '<div style="font-size:12px;color:var(--text-secondary);margin-top:6px">任务从 Cache/WebTaskRuns/ 目录恢复；只要对应文件夹未被手动清除，就可以继续查看日志和结果。</div></div>' +
      '<div class="task-selection-toolbar">' +
      '<label class="task-select-all"><input type="checkbox" id="task-select-all" onchange="App.toggleAllTaskSelection(this.checked)" /> 全选</label>' +
      '<span id="task-selection-count">已选择 0 个任务</span>' +
      '<button class="btn btn-danger" id="task-delete-selected" onclick="App.deleteSelectedTasks()" disabled>删除所选</button>' +
      '</div>' +
      '<div class="task-list">' +
      tasks.map(t => UI.taskItem(t, { selectable: true })).join('') + '</div>';
    this.updateTaskSelectionToolbar();
  },

  toggleAllTaskSelection(checked) {
    document.querySelectorAll('.task-select-checkbox').forEach(input => {
      input.checked = !!checked;
    });
    this.updateTaskSelectionToolbar();
  },

  updateTaskSelectionToolbar() {
    const boxes = Array.from(document.querySelectorAll('.task-select-checkbox'));
    const selected = boxes.filter(input => input.checked);
    const count = document.getElementById('task-selection-count');
    if (count) count.textContent = `已选择 ${selected.length} 个任务`;
    const deleteBtn = document.getElementById('task-delete-selected');
    if (deleteBtn) deleteBtn.disabled = selected.length === 0;
    const all = document.getElementById('task-select-all');
    if (all) {
      all.checked = boxes.length > 0 && selected.length === boxes.length;
      all.indeterminate = selected.length > 0 && selected.length < boxes.length;
    }
  },

  async deleteSelectedTasks() {
    const ids = Array.from(document.querySelectorAll('.task-select-checkbox'))
      .filter(input => input.checked)
      .map(input => input.value)
      .filter(Boolean);
    if (!ids.length) return;
    const ok = window.confirm(`将删除 ${ids.length} 个任务的任务信息、运行日志和任务记录文件夹。\n\n结果 MAT 文件不会被删除，如需清理 MAT 请手动删除。\n\n确认删除吗？`);
    if (!ok) return;
    const res = await this.apiPost('/api/tasks/delete', { ids }).catch((err) => ({ error: err?.message || '删除失败' }));
    if (!res || res.error || res.detail) {
      alert(`删除任务失败：${res?.detail || res?.error || '未知错误'}`);
      return;
    }
    const skipped = Array.isArray(res.skipped) ? res.skipped : [];
    if (skipped.length) {
      const detail = skipped.map(item => `${item.id}: ${item.reason}`).join('\n');
      alert(`部分任务未删除，通常是因为仍在运行或记录已不存在：\n\n${detail}`);
    }
    await this.loadTasks();
  },

  async viewTask(taskId) {
    // Activate tasks page WITHOUT calling loadTasks() to avoid content overwrite race condition.
    this.currentPage = 'tasks';
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const active = document.querySelector('.nav-item[data-page="tasks"]');
    if (active) active.classList.add('active');
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const pageEl = document.getElementById('page-tasks');
    if (pageEl) pageEl.classList.add('active');

    const container = document.getElementById('tasks-content');
    container.innerHTML = '<div class="empty-state"><div class="empty-icon">⏳</div><div class="empty-text">加载任务详情...</div></div>';

    const task = await this.api(`/api/tasks/${taskId}`).catch(() => null);
    if (!task || task.error) {
      container.innerHTML = UI.hint('error', '任务未找到');
      return;
    }

    const statusMap = {
      running: '运行中', success: '完结', failed: '失败', pending: '等待中', stopped: '已停止'
    };
    const taskArtifacts = Array.isArray(task.artifacts) ? [...task.artifacts] : [];
    const moduleLabel = UI.taskModuleLabel ? UI.taskModuleLabel(task) : task.module;

    let html = `
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:16px">
        <button class="btn btn-secondary" onclick="App.loadTasks()">← 返回任务列表</button>
        ${task.task_dir ? `<button class="btn btn-secondary" onclick="App.openTaskFolder('${UI.escapeAttr(task.id)}')">打开文件夹</button>` : ''}
        ${(task.status === 'running' || task.status === 'pending') ? `<button class="btn btn-danger" id="task-stop-${UI.escapeAttr(task.id)}" onclick="App.stopTask('${UI.escapeAttr(task.id)}')">停止</button>` : ''}
      </div>
      <div class="card" style="margin-bottom:16px">
        <div class="card-header">
          <span class="card-title">${UI.escapeHtml(moduleLabel)}</span>
          <span class="task-status-badge ${task.status}">${statusMap[task.status]}</span>
        </div>
        <div style="font-size:13px;color:var(--text-secondary)">
          <div>任务 ID: ${task.id}</div>
          <div>模块: ${UI.escapeHtml(task.module || '')}</div>
          <div>参数: ${task.args.join(' ')}</div>
          <div>创建: ${task.created_at}</div>
          ${task.started_at ? `<div>开始: ${task.started_at}</div>` : ''}
          ${task.finished_at ? `<div>完成: ${task.finished_at}</div>` : ''}
          ${task.task_dir ? `<div>日志目录: <code>${UI.escapeHtml(task.task_dir)}</code></div>` : ''}
        </div>
      </div>
    `;

    if (task.error_summary) {
      html += UI.hint('error', '错误摘要: ' + task.error_summary);
    }

    html += this._renderTaskConfig(task.config);

    html += `<div id="task-artifacts-section">${this._renderTaskArtifacts(taskArtifacts)}</div>`;

    html += `<div class="card"><div class="card-header" style="margin-bottom:12px">
      <span class="card-title">运行日志</span>
      <a class="btn btn-secondary" style="padding:6px 10px;font-size:12px;text-decoration:none"
        href="/api/tasks/${encodeURIComponent(task.id)}/logs.txt" download="${UI.escapeAttr(task.id)}_run.log.txt">保存日志</a>
    </div>` +
      '<div class="log-console" id="task-log-console">' +
      '<div class="log-line log-stdout" style="color:var(--text-secondary);font-style:italic">' +
      '等待输出中...</div>' +
      '</div></div>';

    container.innerHTML = html;

    const logConsole = document.getElementById('task-log-console');

    // Show existing logs
    if (task.logs && task.logs.length > 0 && task.status !== 'running' && task.status !== 'pending') {
      // Only render from REST for finished tasks; running tasks get logs via WebSocket replay.
      logConsole.innerHTML = task.logs.map(e => UI.logLine(e)).join('');
      logConsole.scrollTop = logConsole.scrollHeight;
    }

    // Connect WebSocket for live updates
    if (task.status === 'running' || task.status === 'pending') {
      if (this.activeWs) this.activeWs.close();
      this.activeWs = WS.connect(taskId, {
        onMessage: (msg) => {
          if (msg.type === 'stdout' || msg.type === 'stderr') {
            // Clear the placeholder hint on first real log line.
            const placeholder = logConsole.querySelector('.log-line[style*="italic"]');
            if (placeholder) placeholder.remove();
            logConsole.innerHTML += UI.logLine(msg);
            logConsole.scrollTop = logConsole.scrollHeight;
          }
          if (msg.type === 'artifact' && msg.artifact) {
            if (!taskArtifacts.some(a => a.path === msg.artifact.path)) {
              taskArtifacts.push(msg.artifact);
            }
            const section = document.getElementById('task-artifacts-section');
            if (section) section.innerHTML = this._renderTaskArtifacts(taskArtifacts);
          }
          if (msg.type === 'status') {
            // Update status badge
            const badge = container.querySelector('.task-status-badge');
            if (badge) {
              badge.className = `task-status-badge ${msg.status}`;
              badge.textContent = statusMap[msg.status] || msg.status;
            }
            const stopBtn = document.getElementById(`task-stop-${task.id}`);
            if (stopBtn && (msg.status === 'success' || msg.status === 'failed' || msg.status === 'stopped')) {
              stopBtn.remove();
            }
            if (Array.isArray(msg.artifacts)) {
              taskArtifacts.splice(0, taskArtifacts.length, ...msg.artifacts);
              const section = document.getElementById('task-artifacts-section');
              if (section) section.innerHTML = this._renderTaskArtifacts(taskArtifacts);
            }
            if (msg.error_summary) {
              const hintHtml = UI.hint('error', '错误摘要: ' + msg.error_summary);
              logConsole.parentElement.insertAdjacentHTML('beforebegin', hintHtml);
            }
          }
        },
        onClose: () => { this.activeWs = null; }
      });
    }
  },

  // ──────────── Documents ────────────
  async loadDocs() {
    const container = document.getElementById('docs-content');
    const docs = await this.api('/api/docs').catch(() => []);
    this.docs = docs;

    if (docs.length === 0) {
      container.innerHTML = UI.empty('📄', '未找到文档');
      return;
    }

    // Group docs by category
    const groups = {};
    docs.forEach(d => {
      const key = d.category || 'Other';
      if (!groups[key]) groups[key] = { label: d.category_label || key, icon: d.category_icon || '📄', items: [] };
      groups[key].items.push(d);
    });

    const catOrder = ['General', 'Modules', 'Theory', 'Development'];
    const sortedKeys = [...Object.keys(groups)].sort((a, b) => {
      const ia = catOrder.indexOf(a); const ib = catOrder.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });

    let html = '<div class="doc-list">';
    sortedKeys.forEach(key => {
      const g = groups[key];
      html += `<div style="margin-bottom:6px;padding:8px 12px;font-size:12px;font-weight:600;
        color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.5px;
        border-bottom:1px solid var(--border-color)">${g.icon} ${g.label}</div>`;
      html += g.items.map(d => UI.docItem(d)).join('');
    });
    html += '</div><div id="doc-viewer" style="margin-top:20px"></div>';
    container.innerHTML = html;
  },

  async viewDoc(relPath) {
    const doc = await this.api(`/api/docs/${relPath}`).catch(() => null);
    const viewer = document.getElementById('doc-viewer');
    if (!doc || !viewer) return;

    if (typeof marked !== 'undefined') {
      viewer.innerHTML = `<div class="doc-content">${this._renderMarkdownWithMath(doc.content)}</div>`;
    } else {
      viewer.innerHTML = `<div class="doc-content"><pre>${UI.escapeHtml(doc.content)}</pre></div>`;
    }
    this._typesetMath(viewer);
  },

  _renderMarkdownWithMath(markdown) {
    const mathBlocks = [];
    let protectedMarkdown = String(markdown || '');
    const stashMath = (match) => {
      const token = `@@JAXMECH_MATH_${mathBlocks.length}@@`;
      mathBlocks.push({ token, value: match });
      return token;
    };

    protectedMarkdown = protectedMarkdown
      .replace(/\$\$[\s\S]*?\$\$/g, stashMath)
      .replace(/\\\[[\s\S]*?\\\]/g, stashMath)
      .replace(/\\\([\s\S]*?\\\)/g, stashMath)
      .replace(/\$(?!\$)(?:\\.|[^$\\\n])+\$/g, stashMath);

    let html = marked.parse(protectedMarkdown);
    mathBlocks.forEach(({ token, value }) => {
      html = html.split(token).join(UI.escapeHtml(value));
    });
    return html;
  },

  _typesetMath(el, attempt = 0) {
    if (!el) return;
    if (window.MathJax?.typesetPromise) {
      MathJax.typesetPromise([el]).catch(e => console.warn('MathJax typeset:', e));
    } else if (window.MathJax?.startup?.promise) {
      MathJax.startup.promise
        .then(() => MathJax.typesetPromise([el]))
        .catch(e => console.warn('MathJax:', e));
    } else if (attempt < 40) {
      window.setTimeout(() => this._typesetMath(el, attempt + 1), 150);
    }
  },

  // ──────────── Settings ────────────
  async loadSettings() {
    const container = document.getElementById('settings-content');
    container.innerHTML = '<div class="empty-state"><div class="empty-icon">⏳</div><div class="empty-text">加载中...</div></div>';

    const cfg = await this.api('/api/config/env').catch(() => ({ values: {} }));
    const vals = { ...(cfg.values || {}) };
    if (!vals.wsl_stored_root?.trim() && vals.win_stored_root?.trim()) {
      vals.wsl_stored_root = this.winToWsl(vals.win_stored_root.trim());
    }
    if (!vals.win_stored_root?.trim() && vals.wsl_stored_root?.trim()) {
      vals.win_stored_root = this.wslToWin(vals.wsl_stored_root.trim());
    }

    // 首次配置检测：关键字段为空时显示引导横幅
    const missingCritical = !vals.windows_python_exe?.trim() || !vals.wsl_python?.trim();

    // field_type 与后端 browse.py 的 _FIELD_CONFIG 对应
    const fields = [
      {
        key: 'wsl_python', label: 'WSL Python 路径', fieldType: 'wsl_python', isFile: true,
        autoDetect: 'App.autoDetectWslPython()',
        hint: 'WSL 侧 Python 可执行文件的完整绝对路径，例如：/home/username/miniconda3/envs/jax-fem-env/bin/python'
      },
      {
        key: 'windows_python_exe', label: 'Windows Python 路径', fieldType: 'windows_python', isFile: true,
        autoDetect: 'App.autoDetectWinPython()',
        hint: 'Windows 侧 Python 可执行文件的完整路径，例如：C:\\Users\\username\\miniconda3\\python.exe'
      },
      {
        key: 'win_stored_root', label: 'Windows 模型根目录', fieldType: 'win_folder', isFile: false,
        hint: 'Windows 侧工作模型目录。默认与 WSL 模型根目录指向同一物理目录，例如：E:\\ProjectLocal\\Models'
      },
      {
        key: 'wsl_stored_root', label: 'WSL 模型根目录', fieldType: 'wsl_folder', isFile: false,
        hint: 'WSL 侧工作模型目录（/mnt/... 格式）。默认与 Windows 模型根目录指向同一物理目录，例如：/mnt/e/ProjectLocal/Models'
      },
      {
        key: 'windows_abaqus_cmd', label: 'ABAQUS 命令 (windows_abaqus_cmd)', fieldType: null, isFile: false,
        hint: 'ODB 验证所需的 Windows 侧 ABAQUS 可执行命令，例如：abaqus 或 C:\\SIMULIA\\Commands\\abaqus'
      },
    ];

    let html = '';

    // ── 首次配置横幅 ──
    if (missingCritical) {
      html += `<div style="margin-bottom:16px;padding:14px 18px;border-radius:10px;
          background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3)">
        <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
          <div style="flex:1;min-width:200px">
            <div style="font-size:14px;font-weight:600;color:#fbbf24;margin-bottom:4px">⚙ 检测到未填写的关键配置</div>
            <div style="font-size:12px;color:var(--text-secondary)">
              首次使用请点击「⚡ 一键自动检测并保存」，程序将自动扫描本机 Python 路径并写入 env.cfg。
            </div>
          </div>
          <button class="btn btn-primary" style="flex-shrink:0;white-space:nowrap"
            onclick="App.autoDetectAll()">⚡ 一键自动检测并保存</button>
          <button class="btn btn-secondary" style="flex-shrink:0;white-space:nowrap"
            onclick="App.installMissingLibraries()">📦 一键安装所有支持库</button>
        </div>
      </div>`;
    }

    html += '<div class="card"><div class="card-title" style="margin-bottom:20px">环境配置 (env.cfg)</div>' +
      '<div class="settings-form">';
    fields.forEach(f => {
      const val = UI.escapeHtml(vals[f.key] || '');
      const isStoredRootField = f.key === 'win_stored_root' || f.key === 'wsl_stored_root';
      const groupId = isStoredRootField ? ` id="settings-group-${f.key}"` : '';
      if (f.fieldType) {
        // Field with browse + optional auto-detect button
        const autoBtn = f.autoDetect
          ? `<button class="btn btn-secondary" style="flex-shrink:0;padding:0 10px;height:36px;font-size:12px"
               title="自动检测并填入" onclick="${f.autoDetect}">⚡</button>`
          : '';
        html += `<div class="form-group"${groupId}>
          <label class="form-label">${f.label}</label>
          <div style="display:flex;gap:6px;align-items:center">
            <input class="form-input" id="cfg-${f.key}" value="${val}" style="flex:1" ${isStoredRootField ? `oninput="App._onStoredRootInputChange('${f.key}')" onchange="App._onStoredRootInputChange('${f.key}')"` : ''} />
            ${autoBtn}
            <button class="btn btn-secondary" style="flex-shrink:0;padding:0 12px;height:36px;font-size:13px"
              onclick="App.browseForPath('${f.key}','${f.fieldType}')">${f.isFile ? '📄' : '📂'} 浏览</button>
          </div>
          <span class="form-hint">${f.hint}</span>
        </div>`;
      } else {
        html += `<div class="form-group">
          <label class="form-label">${f.label}</label>
          <input class="form-input" id="cfg-${f.key}" value="${val}" />
          <span class="form-hint">${f.hint}</span>
        </div>`;
      }
    });
    html += '</div>' +
      '<div id="settings-stored-root-note" style="margin-top:16px"></div>' +
      '<div style="margin-top:20px;display:flex;gap:12px;flex-wrap:wrap">' +
      '<button class="btn btn-primary" onclick="App.saveSettings()">💾 保存配置</button>' +
      '<button class="btn btn-secondary" onclick="App.checkEnv()">🔍 环境检测</button>' +
      (missingCritical ? '' : '<button class="btn btn-secondary" onclick="App.autoDetectAll()">⚡ 一键自动检测</button>') +
      '<button class="btn btn-secondary" onclick="App.installMissingLibraries()">📦 一键安装所有支持库</button>' +
      '</div></div>';

    html += '<div id="settings-check" style="margin-top:20px"></div>';
    container.innerHTML = html;
    this._updateStoredRootHighlight();
  },

  /**
   * 调用后端弹出 Windows 原生文件/目录选择对话框，将选中路径填入对应输入框。
   * @param {string} fieldKey  - cfg-* input 的 key
   * @param {string} fieldType - browse.py 的 _FIELD_CONFIG key
   */
  async browseForPath(fieldKey, fieldType) {
    const res = await this.apiPost('/api/browse', { field_type: fieldType }).catch(() => null);
    if (!res || !res.path) return;  // 用户取消或出错

    let chosen = res.path;
    // WSL 字段且得到的是 Windows 路径：两种情况
    //  1. \\wsl$\Ubuntu\home\...  → 直接保留（wsl_python 与 wsl_folder 需转 WSL格式）
    //  2. E:\foo  → 对 Windows 字段直接保留
    const isWslField = (fieldType === 'wsl_python' || fieldType === 'wsl_folder');
    if (isWslField) {
      if (/^\\\\wsl\$/i.test(chosen) || /^\\\\wsl\.localhost/i.test(chosen)) {
        // \\wsl$\Ubuntu\home\... → /home/...
        chosen = chosen.replace(/^\\\\wsl\$\\[^\\]+/i, '')
                       .replace(/^\\\\wsl\.localhost\\[^\\]+/i, '')
                       .replace(/\\/g, '/');
        if (!chosen.startsWith('/')) chosen = '/' + chosen;
      } else if (/^[A-Za-z]:[\\/]/.test(chosen)) {
        // 如果用户导航到了 Windows 盘，转换为 /mnt/...
        chosen = chosen.replace(/\\/g, '/').replace(/^([A-Za-z]):\//, (_, d) => `/mnt/${d.toLowerCase()}/`);
      }
    }

    const input = document.getElementById(`cfg-${fieldKey}`);
    if (input) {
      input.value = chosen;
      // 双向同步模型根目录
      if (fieldKey === 'win_stored_root') {
        const wslInp = document.getElementById('cfg-wsl_stored_root');
        if (wslInp && !wslInp.value)
          wslInp.value = chosen.replace(/\\/g, '/').replace(/^([A-Za-z]):\//, (_, d) => `/mnt/${d.toLowerCase()}/`);
      } else if (fieldKey === 'wsl_stored_root') {
        const winInp = document.getElementById('cfg-win_stored_root');
        if (winInp && !winInp.value)
          winInp.value = chosen.replace(/^\/mnt\/([a-z])\//, (_, d) => `${d.toUpperCase()}:\\`).replace(/\//g, '\\');
      }
    }
    if (fieldKey === 'win_stored_root' || fieldKey === 'wsl_stored_root') {
      this._updateStoredRootHighlight();
    }
  },


  async saveSettings() {
    const keys = ['wsl_python', 'windows_python_exe', 'win_stored_root', 'wsl_stored_root', 'windows_abaqus_cmd'];

    const values = {};
    keys.forEach(k => {
      const el = document.getElementById(`cfg-${k}`);
      if (el) values[k] = el.value;
    });
    if (!values.wsl_stored_root?.trim() && values.win_stored_root?.trim()) {
      values.wsl_stored_root = this.winToWsl(values.win_stored_root.trim());
      const wslInput = document.getElementById('cfg-wsl_stored_root');
      if (wslInput) wslInput.value = values.wsl_stored_root;
    }
    if (!values.win_stored_root?.trim() && values.wsl_stored_root?.trim()) {
      values.win_stored_root = this.wslToWin(values.wsl_stored_root.trim());
      const winInput = document.getElementById('cfg-win_stored_root');
      if (winInput) winInput.value = values.win_stored_root;
    }
    await this.apiPut('/api/config/env', { values });
    const checkDiv = document.getElementById('settings-check');
    if (checkDiv) checkDiv.innerHTML = UI.hint('success', '配置已保存');
    this._updateStoredRootHighlight();
  },

  async checkEnv() {
    const checkDiv = document.getElementById('settings-check');
    if (checkDiv) checkDiv.innerHTML = UI.hint('info', '正在检测环境，请稍候...');

    // 并发触发所有检测（耗时操作）
    const [status, winPyRes, wslPyRes, abaqusRes] = await Promise.all([
      this.api('/api/status').catch(() => null),
      this.api('/api/status/detect-windows-python').catch(() => null),
      this.api('/api/status/detect-wsl-python').catch(() => null),
      this.api('/api/status/detect-abaqus').catch(() => null),
    ]);

    if (!status) {
      checkDiv.innerHTML = UI.hint('error', '环境检测失败，请确认服务正常运行');
      return;
    }

    let html = '';
    const autoFilled = {};  // key → value，用于最终批量保存

    // ── WSL 安装状态 ──
    html += status.wsl?.available
      ? UI.hint('success', 'WSL：已安装')
      : UI.hint('error', 'WSL：未安装 — ' + (status.wsl?.detail || '未检测到'));

    // ── WSL Python / JAX ──
    if (wslPyRes?.found) {
      const best = wslPyRes.best;
      html += UI.hint('success',
        `JAX ${best.jax_version} — WSL Python ${best.py_version} (${best.source})<br>` +
        `<code style="font-size:11px">${UI.escapeHtml(best.path)}</code>`);
      autoFilled['wsl_python'] = best.path;
      const inp = document.getElementById('cfg-wsl_python');
      if (inp) inp.value = best.path;

      // 多候选选择器
      if ((wslPyRes.candidates || []).length > 1) {
        html += this._renderCandidatePicker(
          'wsl_python', wslPyRes.candidates,
          c => `${c.path} — Python ${c.py_version} / JAX ${c.jax_version} [${c.source}]`
        );
      }
    } else {
      // JAX 未找到 — 降级显示 /api/status 里的 jax 结果
      const jaxPyPath = status.jax?.python_path || '';
      const errMsg = status.jax?.error ? ` (${status.jax.error})` : '';
      if (status.jax?.available) {
        html += UI.hint('success',
          `JAX ${status.jax.version} — 使用 WSL Python: ${UI.escapeHtml(jaxPyPath)}`);
        autoFilled['wsl_python'] = jaxPyPath;
        const inp = document.getElementById('cfg-wsl_python');
        if (inp && jaxPyPath) inp.value = jaxPyPath;
      } else {
        html += UI.hint('warning',
          `JAX：未检测到${errMsg}。` +
          `请确认 WSL Python 环境已安装 JAX，或手动填写 WSL Python 路径。`);
      }
    }

    // ── Windows Python ──
    if (winPyRes?.found) {
      const best = winPyRes.best;
      html += UI.hint('success',
        `Windows Python ${best.version} [${best.source}]<br>` +
        `<code style="font-size:11px">${UI.escapeHtml(best.executable)}</code>`);
      autoFilled['windows_python_exe'] = best.executable;
      const inp = document.getElementById('cfg-windows_python_exe');
      if (inp) inp.value = best.executable;

      // 多候选选择器
      if ((winPyRes.candidates || []).length > 1) {
        html += this._renderCandidatePicker(
          'windows_python_exe', winPyRes.candidates,
          c => `${c.executable} — Python ${c.version} [${c.source}]`
        );
      }
    } else {
      const winPyExe = status.windows_python?.executable || '';
      const winPyVer = status.windows_python?.version || '?';
      html += UI.hint('success', `Windows Python ${winPyVer} — ${UI.escapeHtml(winPyExe)}`);
      autoFilled['windows_python_exe'] = winPyExe;
      const inp = document.getElementById('cfg-windows_python_exe');
      if (inp && winPyExe) inp.value = winPyExe;
    }

    // ── ABAQUS ──
    if (abaqusRes?.found) {
      html += UI.hint('success',
        `ABAQUS：已检测到 <code>${UI.escapeHtml(abaqusRes.abaqus_cmd)}</code>（${UI.escapeHtml(abaqusRes.source)}）`);
      const inp = document.getElementById('cfg-windows_abaqus_cmd');
      if (inp && !inp.value.trim()) {
        inp.value = abaqusRes.abaqus_cmd;
        autoFilled['windows_abaqus_cmd'] = abaqusRes.abaqus_cmd;
      }
    } else {
      html += UI.hint('warning', 'ABAQUS：未检测到，如需 ODB 验证请手动填写命令。');
    }

    // ── Python 库检测 ──
    const libs = status.python_libraries;
    if (libs) {
      if (libs.all_available) {
        html += UI.hint('success', `Windows Python 必需库：全部已安装（${Object.keys(libs.libraries).length} 个）`);
      } else {
        html += UI.hint('warning', `缺少 ${libs.missing_count} 个必需库`);
        html += `<div style="margin-top:8px;display:flex;gap:8px">
          <button class="btn btn-primary" onclick="App.installMissingLibraries()">📦 一键安装缺失库</button>
          <button class="btn btn-secondary" onclick="App.showLibraryDetails()">📋 查看详情</button>
        </div>`;
      }
    }

    // ── 自动保存检测结果 ──
    if (Object.keys(autoFilled).length) {
      await this.apiPut('/api/config/env', { values: autoFilled }).catch(() => null);
      html += `<div style="margin-top:12px;padding:10px 14px;border-radius:8px;
          background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.25);
          font-size:12px;color:#34d399">
        ✓ 以上检测结果已自动写入 Config/env.cfg
      </div>`;
    }

    checkDiv.innerHTML = html;
  },

  /**
   * 渲染多候选 Python 路径选择器（内联 HTML 片段）。
   * @param {string} fieldKey   - cfg-* input 的 key
   * @param {Array}  candidates - [{executable|path, version|py_version, source}, ...]
   * @param {Function} labelFn  - (c) => 显示文本
   */
  _renderCandidatePicker(fieldKey, candidates, labelFn) {
    const options = candidates.map((c, i) => {
      const val = c.executable || c.path || '';
      const label = labelFn(c);
      return `<option value="${UI.escapeAttr(val)}" ${i === 0 ? 'selected' : ''}>${UI.escapeHtml(label)}</option>`;
    }).join('');
    return `<div style="margin-top:6px;display:flex;gap:8px;align-items:center">
      <select class="form-input" style="flex:1;font-size:12px;font-family:monospace"
        onchange="App._pickCandidate('${fieldKey}', this.value)">${options}</select>
      <button class="btn btn-secondary" style="padding:0 12px;height:34px;font-size:12px;flex-shrink:0"
        onclick="App._pickCandidate('${fieldKey}', this.previousElementSibling.value)">选用</button>
    </div>`;
  },

  /** 将候选值填入对应输入框（不自动保存）。 */
  _pickCandidate(fieldKey, value) {
    const inp = document.getElementById(`cfg-${fieldKey}`);
    if (inp) inp.value = value;
  },

  /** 一键自动检测 Windows Python 并单独填入。 */
  async autoDetectWinPython() {
    const inp = document.getElementById('cfg-windows_python_exe');
    if (inp) inp.value = '检测中...';
    const res = await this.api('/api/status/detect-windows-python').catch(() => null);
    if (res?.found && res.best) {
      if (inp) inp.value = res.best.executable;
      const checkDiv = document.getElementById('settings-check');
      if (checkDiv) {
        let info = UI.hint('success',
          `Windows Python ${res.best.version} [${res.best.source}]: ${UI.escapeHtml(res.best.executable)}`);
        if ((res.candidates || []).length > 1) {
          info += this._renderCandidatePicker(
            'windows_python_exe', res.candidates,
            c => `${c.executable} — Python ${c.version} [${c.source}]`
          );
        }
        checkDiv.innerHTML = info;
      }
    } else {
      if (inp) inp.value = '';
      const checkDiv = document.getElementById('settings-check');
      if (checkDiv) checkDiv.innerHTML = UI.hint('error', 'Windows Python 未检测到，请手动填写路径。');
    }
  },

  /** 一键扫描 WSL Python（含 JAX）并单独填入。 */
  async autoDetectWslPython() {
    const inp = document.getElementById('cfg-wsl_python');
    if (inp) inp.value = '扫描 WSL 中...';
    const checkDiv = document.getElementById('settings-check');
    if (checkDiv) checkDiv.innerHTML = UI.hint('info', '正在扫描 WSL 环境中的 Python / JAX，可能需要 20s...');
    const res = await this.api('/api/status/detect-wsl-python').catch(() => null);
    if (res?.found && res.best) {
      if (inp) inp.value = res.best.path;
      let info = UI.hint('success',
        `JAX ${res.best.jax_version} — WSL Python ${res.best.py_version} [${res.best.source}]: ` +
        `<code>${UI.escapeHtml(res.best.path)}</code>`);
      if ((res.candidates || []).length > 1) {
        info += this._renderCandidatePicker(
          'wsl_python', res.candidates,
          c => `${c.path} — Python ${c.py_version} / JAX ${c.jax_version} [${c.source}]`
        );
      }
      if (checkDiv) checkDiv.innerHTML = info;
    } else {
      if (inp) inp.value = '';
      if (checkDiv) checkDiv.innerHTML = UI.hint('warning',
        'WSL 中未找到包含 JAX 的 Python 环境。<br>请先在 WSL 中安装 JAX，或手动填写路径。');
    }
  },

  /** 一键自动检测所有可检测字段并保存。 */
  async autoDetectAll() {
    const checkDiv = document.getElementById('settings-check');
    if (checkDiv) checkDiv.innerHTML = UI.hint('info', '正在自动检测所有环境路径，请稍候（WSL 扫描可能需要 20–30s）...');

    const [winPyRes, wslPyRes, abaqusRes] = await Promise.all([
      this.api('/api/status/detect-windows-python').catch(() => null),
      this.api('/api/status/detect-wsl-python').catch(() => null),
      this.api('/api/status/detect-abaqus').catch(() => null),
    ]);

    const saved = {};
    let html = '';

    // Windows Python
    if (winPyRes?.found && winPyRes.best) {
      const inp = document.getElementById('cfg-windows_python_exe');
      if (inp) inp.value = winPyRes.best.executable;
      saved['windows_python_exe'] = winPyRes.best.executable;
      html += UI.hint('success',
        `Windows Python ${winPyRes.best.version} [${winPyRes.best.source}]<br>` +
        `<code style="font-size:11px">${UI.escapeHtml(winPyRes.best.executable)}</code>`);
      if ((winPyRes.candidates || []).length > 1) {
        html += this._renderCandidatePicker(
          'windows_python_exe', winPyRes.candidates,
          c => `${c.executable} — Python ${c.version} [${c.source}]`
        );
      }
    } else {
      html += UI.hint('warning', 'Windows Python：未检测到，请手动填写。');
    }

    // WSL Python
    if (wslPyRes?.found && wslPyRes.best) {
      const inp = document.getElementById('cfg-wsl_python');
      if (inp) inp.value = wslPyRes.best.path;
      saved['wsl_python'] = wslPyRes.best.path;
      html += UI.hint('success',
        `JAX ${wslPyRes.best.jax_version} — WSL Python ${wslPyRes.best.py_version} [${wslPyRes.best.source}]<br>` +
        `<code style="font-size:11px">${UI.escapeHtml(wslPyRes.best.path)}</code>`);
      if ((wslPyRes.candidates || []).length > 1) {
        html += this._renderCandidatePicker(
          'wsl_python', wslPyRes.candidates,
          c => `${c.path} — Python ${c.py_version} / JAX ${c.jax_version} [${c.source}]`
        );
      }
    } else {
      html += UI.hint('warning', 'WSL Python / JAX：未扫描到。如 JAX 未安装，请先在 WSL 中安装。');
    }

    // ABAQUS
    if (abaqusRes?.found) {
      const inp = document.getElementById('cfg-windows_abaqus_cmd');
      if (inp && !inp.value.trim()) {
        inp.value = abaqusRes.abaqus_cmd;
        saved['windows_abaqus_cmd'] = abaqusRes.abaqus_cmd;
      }
      html += UI.hint('success',
        `ABAQUS：<code>${UI.escapeHtml(abaqusRes.abaqus_cmd)}</code> (${UI.escapeHtml(abaqusRes.source)})`);
    }

    // 批量保存
    if (Object.keys(saved).length) {
      await this.apiPut('/api/config/env', { values: saved }).catch(() => null);
      html += `<div style="margin-top:12px;padding:10px 14px;border-radius:8px;
          background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.25);
          font-size:12px;color:#34d399">
        ✓ 已将 ${Object.keys(saved).length} 个检测结果写入 Config/env.cfg
        <span style="color:var(--text-secondary);margin-left:8px">（其余字段请手动填写后保存）</span>
      </div>`;
      // 重新加载页面以刷新横幅状态
      setTimeout(() => this.loadSettings(), 800);
    } else {
      html += UI.hint('warning', '未检测到任何可自动填写的路径，请手动填写后点击「保存配置」。');
    }

    if (checkDiv) checkDiv.innerHTML = html;
  },



  /**
   * 一键安装所有缺失的Python库
   */
  async installMissingLibraries() {
    const checkDiv = document.getElementById('settings-check');
    if (checkDiv) {
      checkDiv.innerHTML += UI.hint('info', '正在安装缺失的Python库，请稍候...');
    }

    try {
      const result = await this.apiPost('/api/status/install-missing-libraries', {});

      let html = '';
      if (result.success) {
        html += UI.hint('success', `✅ ${result.message}`);
        if (result.results && result.results.length > 0) {
          html += '<div class="library-details">';
          result.results.forEach(r => {
            if (r.success) {
              html += `<div class="library-success">✓ ${r.library}: ${r.message}</div>`;
            } else {
              html += `<div class="library-error">✗ ${r.library}: ${r.error}</div>`;
            }
          });
          html += '</div>';
        }

        // 安装完成后重新检测环境
        setTimeout(() => this.checkEnv(), 2000);

      } else {
        html += UI.hint('error', `❌ 安装失败: ${result.message || '未知错误'}`);
        if (result.results && result.results.length > 0) {
          html += '<div class="library-details">';
          result.results.forEach(r => {
            if (r.success) {
              html += `<div class="library-success">✓ ${r.library}: 安装成功</div>`;
            } else {
              html += `<div class="library-error">✗ ${r.library}: ${r.error}</div>`;
            }
          });
          html += '</div>';
        }
      }

      checkDiv.innerHTML += html;

    } catch (error) {
      checkDiv.innerHTML += UI.hint('error', `安装出错: ${error.message || error}`);
    }
  },

  /**
   * 显示Python库的详细状态
   */
  async showLibraryDetails() {
    const status = await this.api('/api/status').catch(() => null);
    if (!status || !status.python_libraries) {
      alert('无法获取库状态信息');
      return;
    }

    const libs = status.python_libraries.libraries;
    let details = 'Python 必需库状态详情:\n\n';

    Object.entries(libs).forEach(([name, info]) => {
      const status_text = info.available ? `✓ 已安装 (${info.version})` : '✗ 未安装';
      details += `${name}: ${status_text}\n`;
      details += `  描述: ${info.description}\n`;
      details += `  用途: ${info.required_for}\n\n`;
    });

    alert(details);
  },

  // ════════════════════════════════════════════════════════════════════
  //  Visualization page
  // ════════════════════════════════════════════════════════════════════

  _viz: {
    ws_a: null,
    ws_b: null,
    info_a: null,
    info_b: null,
    compareMode: false,
    compareLocked: false,
    syncDisplay: false,
    dragging: false,
    dragSlot: null,
    lastX: 0,
    lastY: 0,
    dragButton: 0,
    absoluteDiff: false,
    diffPayload: null,
    fieldModalSlot: 'a',
    fieldSortKey: 'default',
    fieldSortDir: 'asc',
    models: [],
    modelByKey: {},
    matCache: {},
    playTimers: {},
    playInFlight: {},
    slotState: {
      a: { modelKey: '', matPath: '', mats: [], family: '', quantityValue: '', inspect: null, selectedFields: [], sourceLocked: false },
      b: { modelKey: '', matPath: '', mats: [], family: '', quantityValue: '', inspect: null, selectedFields: [], sourceLocked: false },
    },
    // ---- Probe (Abaqus-style hover/click picking) -----------------
    // Two distinct modes: 'node' (only valid for node-located fields)
    // and 'element' (Gauss / element-located fields; lists every GP
    // value on the picked element). Mode is toggled in the A
    // toolbar; both A and B canvases share the same mode.
    probeMode: false,
    probeKind: 'node',
    // Per-slot last-issued req_id. The WS reply echoes ``req_id``;
    // we drop late replies to keep hover responsive on big meshes
    // ("last-write-wins").
    probePending: { a: 0, b: 0 },
    probeNextReqId: 1,
    // Transient hover card (single DOM node, follows the cursor).
    probeHoverEl: null,
    probeHoverMarkerEl: null,
    probeHoverThrottleAt: 0,
    // Pinned cards (max 20), each entry frozen at click time with its
    // mode/field/frame/data so later UI changes never mutate it.
    probeCards: [],
    probeNextCardId: 1,
    probeMarkers: [],
    probeCompareSettings: null,
    probeNextChartId: 1,
    // Saved field key per slot to restore when probe is toggled off
    // (probe mode temporarily constrains the field dropdown).
    probeSavedFieldKey: { a: '', b: '' },
    probeSavedFieldComponent: { a: null, b: null },
    elementSelectMode: false,
    elementSelectTab: 'box',
    elementBoxDrag: null,
    elementRightClickCandidate: null,
    elementLastConfirmAt: 0,
    elementSelection: {
      a: { selected: [], hidden_count: 0, can_undo: false, markers: [] },
      b: { selected: [], hidden_count: 0, can_undo: false, markers: [] },
    },
    elementSelectionMarkerEls: { a: [], b: [] },
  },

  async loadVisualization() {
    const el = document.getElementById('viz-content');
    if (!el) return;
    el.innerHTML = this._vizBuildHTML();
    this._vizBindEvents();
    await this._vizEnsureModels();
    await this._vizInitModelSelectors();
    this._vizSetSlotLoadedState('a', false);
    this._vizSetSlotLoadedState('b', false);

    if (this._vizSelectedPath) {
      const targetPath = this._vizSelectedPath;
      this._vizSelectedPath = '';
      await this._vizHydrateSlotFromSource('a', targetPath);
      await this._vizInspectSelectedMat('a', targetPath);
      this._vizRenderSourceSummary();
      return;
    }

    const info = await this.api('/api/viz/info').catch(() => null);
    if (info && info.scene_a) {
      this._viz.info_a = info.scene_a;
      await this._vizHydrateSlotFromSource('a', info.scene_a.source);
      this._vizUpdateControls('a');
      this._vizConnectWs('a');
      this._vizSetSlotLoadedState('a', true);
    }
    if (info && info.compare_mode && info.scene_b) {
      this._viz.info_b = info.scene_b;
      this._viz.compareMode = true;
      this._viz.compareLocked = !!info.compare_locked;
      await this._vizHydrateSlotFromSource('b', info.scene_b.source);
      this._vizUpdateControls('b');
      this._vizConnectWs('b');
      this._vizSetSlotLoadedState('b', true);
    } else if (info) {
      this._viz.compareLocked = !!info.compare_locked;
    }
    this._vizShowCompare(this._viz.compareMode);
    this._vizRenderSourceSummary();
  },

  _vizBuildHTML() {
    return `
      <div class="viz-layout">
        <div class="viz-toolbar" id="viz-toolbar-a">
          <div class="viz-group viz-source-group">
            <span class="viz-group-label">模型 A</span>
            <div class="viz-source-picker" id="viz-source-picker-a">
              <select id="viz-model-a" class="viz-model-select">
                <option value="">加载中...</option>
              </select>
              <select id="viz-mat-a" class="viz-mat-select" disabled>
                <option value="">选择 MAT 结果</option>
              </select>
              <button class="viz-btn" id="viz-mat-refresh-a" title="重新扫描 MAT 结果">刷新</button>
              <button class="viz-btn" id="viz-browse-a" title="浏览 MAT 结果文件">浏览</button>
              <button class="viz-btn primary" id="viz-load-a" title="检查当前 MAT 的场变量">选择MAT</button>
            </div>
            <div class="viz-source-current" id="viz-source-current-a" style="display:none">
              <span class="viz-source-current-text" id="viz-source-current-text-a"></span>
              <button class="viz-btn" id="viz-reselect-a" title="重新选择 MAT 结果">重选MAT</button>
            </div>
          </div>
          <div class="viz-sep"></div>
          <div class="viz-group viz-field-group">
            <span class="viz-group-label">结果量</span>
            <select id="viz-family-a" class="viz-field-family" disabled>
              <option value="">类别</option>
            </select>
            <select id="viz-quantity-a" class="viz-field-select" disabled title="">
              <option value="">结果量</option>
            </select>
            <span class="viz-field-map" id="viz-field-map-a" style="display:none"></span>
          </div>
          <div class="viz-sep"></div>
          <div class="viz-group viz-frame-group">
            <span class="viz-group-label">帧</span>
            <input type="range" id="viz-frame-slider-a" min="0" max="0" value="0" disabled>
            <button class="viz-btn viz-frame-step" id="viz-frame-prev-a" title="上一帧" aria-label="上一帧"><span class="viz-icon viz-icon-prev" aria-hidden="true"></span></button>
            <button class="viz-btn viz-frame-play" id="viz-frame-play-a" title="连续播放" aria-label="连续播放"><span class="viz-icon viz-icon-play" aria-hidden="true"></span></button>
            <button class="viz-btn viz-frame-step" id="viz-frame-next-a" title="下一帧" aria-label="下一帧"><span class="viz-icon viz-icon-next" aria-hidden="true"></span></button>
            <span class="viz-frame-display" id="viz-frame-label-a" title="">1/1</span>
            <span class="viz-frame-detail-host" id="viz-frame-detail-a" style="display:none"></span>
          </div>
          <div class="viz-sep"></div>
          <div class="viz-group viz-view-group">
            <span class="viz-group-label">视图</span>
            <button class="viz-btn viz-view-btn" data-view="+x" data-slot="a">+X</button>
            <button class="viz-btn viz-view-btn" data-view="-x" data-slot="a">-X</button>
            <button class="viz-btn viz-view-btn" data-view="+y" data-slot="a">+Y</button>
            <button class="viz-btn viz-view-btn" data-view="-y" data-slot="a">-Y</button>
            <button class="viz-btn viz-view-btn" data-view="+z" data-slot="a">+Z</button>
            <button class="viz-btn viz-view-btn" data-view="-z" data-slot="a">-Z</button>
            <button class="viz-btn viz-view-btn" data-view="iso" data-slot="a">ISO</button>
          </div>
          <div class="viz-sep"></div>
          <div class="viz-group viz-display-actions">
            <button class="viz-btn" id="viz-edges-a" title="显示或隐藏网格边线">网格</button>
            <button class="viz-btn" id="viz-overlay-a" title="显示或隐藏文件信息与坐标轴">信息</button>
            <button class="viz-btn" id="viz-screenshot-a" title="导出 300 DPI PNG">截图</button>
          </div>
          <div class="viz-sep viz-row-break"></div>
          <div class="viz-group viz-legend-group">
            <span class="viz-group-label">图例</span>
            <input type="number" id="viz-clim-min-a" placeholder="min" step="any">
            <span style="color:var(--text-muted)">~</span>
            <input type="number" id="viz-clim-max-a" placeholder="max" step="any">
            <button class="viz-btn" id="viz-clim-apply-a" title="应用色标范围">应用</button>
            <button class="viz-btn active" id="viz-clim-auto-a" title="锁定当前自动范围或恢复自动范围">自动</button>
            <select id="viz-legend-font-a" class="viz-legend-font" title="图例字体大小（pt）" aria-label="图例字体大小">
              <option value="8">8pt</option>
              <option value="10" selected>10pt</option>
              <option value="12">12pt</option>
              <option value="14">14pt</option>
              <option value="16">16pt</option>
              <option value="18">18pt</option>
              <option value="20">20pt</option>
              <option value="22">22pt</option>
              <option value="24">24pt</option>
              <option value="28">28pt</option>
            </select>
          </div>
          <div class="viz-sep viz-sep-strong"></div>
          <div class="viz-group viz-clip-group">
            <button class="viz-btn viz-clip-toggle" id="viz-clip-toggle-a" data-slot="a" title="开启或关闭轴对齐剖切，观察实体内部场分布">剖切</button>
            <button class="viz-btn viz-clip-axis" data-axis="x" data-slot="a" title="沿 X 法向剖切" disabled>X</button>
            <button class="viz-btn viz-clip-axis" data-axis="y" data-slot="a" title="沿 Y 法向剖切" disabled>Y</button>
            <button class="viz-btn viz-clip-axis active" data-axis="z" data-slot="a" title="沿 Z 法向剖切" disabled>Z</button>
            <input type="range" class="viz-clip-slider" id="viz-clip-slider-a" min="0" max="1000" value="500" step="1" disabled title="拖动剖切平面位置">
            <span class="viz-clip-display" id="viz-clip-label-a">Z 50%</span>
            <button class="viz-btn viz-clip-flip" id="viz-clip-flip-a" data-slot="a" title="翻转保留的一侧（类似 ABAQUS view-cut 的 Above/Below）" disabled>翻转</button>
          </div>
        </div>

        <div class="viz-toolbar" id="viz-toolbar-b" style="display:none">
          <div class="viz-group viz-source-group">
            <span class="viz-group-label">模型 B</span>
            <div class="viz-source-picker" id="viz-source-picker-b">
              <select id="viz-model-b" class="viz-model-select">
                <option value="">加载中...</option>
              </select>
              <select id="viz-mat-b" class="viz-mat-select" disabled>
                <option value="">选择 MAT 结果</option>
              </select>
              <button class="viz-btn" id="viz-mat-refresh-b" title="重新扫描 MAT 结果">刷新</button>
              <button class="viz-btn" id="viz-browse-b" title="浏览 MAT 结果文件">浏览</button>
              <button class="viz-btn primary" id="viz-load-b" title="检查当前 MAT 的场变量">选择MAT</button>
            </div>
            <div class="viz-source-current" id="viz-source-current-b" style="display:none">
              <span class="viz-source-current-text" id="viz-source-current-text-b"></span>
              <button class="viz-btn" id="viz-reselect-b" title="重新选择 MAT 结果">重选MAT</button>
            </div>
          </div>
          <div class="viz-sep"></div>
          <div class="viz-group viz-field-group">
            <span class="viz-group-label">结果量</span>
            <select id="viz-family-b" class="viz-field-family" disabled>
              <option value="">类别</option>
            </select>
            <select id="viz-quantity-b" class="viz-field-select" disabled title="">
              <option value="">结果量</option>
            </select>
            <span class="viz-field-map" id="viz-field-map-b" style="display:none"></span>
          </div>
          <div class="viz-sep"></div>
          <div class="viz-group viz-frame-group">
            <span class="viz-group-label">帧</span>
            <input type="range" id="viz-frame-slider-b" min="0" max="0" value="0" disabled>
            <button class="viz-btn viz-frame-step" id="viz-frame-prev-b" title="上一帧" aria-label="上一帧"><span class="viz-icon viz-icon-prev" aria-hidden="true"></span></button>
            <button class="viz-btn viz-frame-play" id="viz-frame-play-b" title="连续播放" aria-label="连续播放"><span class="viz-icon viz-icon-play" aria-hidden="true"></span></button>
            <button class="viz-btn viz-frame-step" id="viz-frame-next-b" title="下一帧" aria-label="下一帧"><span class="viz-icon viz-icon-next" aria-hidden="true"></span></button>
            <span class="viz-frame-display" id="viz-frame-label-b" title="">1/1</span>
            <span class="viz-frame-detail-host" id="viz-frame-detail-b" style="display:none"></span>
          </div>
          <div class="viz-sep"></div>
          <div class="viz-group viz-view-group">
            <span class="viz-group-label">视图</span>
            <button class="viz-btn viz-view-btn" data-view="+x" data-slot="b">+X</button>
            <button class="viz-btn viz-view-btn" data-view="-x" data-slot="b">-X</button>
            <button class="viz-btn viz-view-btn" data-view="+y" data-slot="b">+Y</button>
            <button class="viz-btn viz-view-btn" data-view="-y" data-slot="b">-Y</button>
            <button class="viz-btn viz-view-btn" data-view="+z" data-slot="b">+Z</button>
            <button class="viz-btn viz-view-btn" data-view="-z" data-slot="b">-Z</button>
            <button class="viz-btn viz-view-btn" data-view="iso" data-slot="b">ISO</button>
          </div>
          <div class="viz-sep"></div>
          <div class="viz-group viz-display-actions">
            <button class="viz-btn" id="viz-edges-b" title="显示或隐藏网格边线">网格</button>
            <button class="viz-btn" id="viz-overlay-b" title="显示或隐藏文件信息与坐标轴">信息</button>
            <button class="viz-btn" id="viz-screenshot-b" title="导出 300 DPI PNG">截图</button>
          </div>
          <div class="viz-sep viz-row-break"></div>
          <div class="viz-group viz-legend-group">
            <span class="viz-group-label">图例</span>
            <input type="number" id="viz-clim-min-b" placeholder="min" step="any">
            <span style="color:var(--text-muted)">~</span>
            <input type="number" id="viz-clim-max-b" placeholder="max" step="any">
            <button class="viz-btn" id="viz-clim-apply-b" title="应用色标范围">应用</button>
            <button class="viz-btn active" id="viz-clim-auto-b" title="锁定当前自动范围或恢复自动范围">自动</button>
            <select id="viz-legend-font-b" class="viz-legend-font" title="图例字体大小（pt）" aria-label="图例字体大小">
              <option value="8">8pt</option>
              <option value="10" selected>10pt</option>
              <option value="12">12pt</option>
              <option value="14">14pt</option>
              <option value="16">16pt</option>
              <option value="18">18pt</option>
              <option value="20">20pt</option>
              <option value="22">22pt</option>
              <option value="24">24pt</option>
              <option value="28">28pt</option>
            </select>
          </div>
          <div class="viz-sep viz-sep-strong"></div>
          <div class="viz-group viz-clip-group">
            <button class="viz-btn viz-clip-toggle" id="viz-clip-toggle-b" data-slot="b" title="开启或关闭轴对齐剖切，观察实体内部场分布">剖切</button>
            <button class="viz-btn viz-clip-axis" data-axis="x" data-slot="b" title="沿 X 法向剖切" disabled>X</button>
            <button class="viz-btn viz-clip-axis" data-axis="y" data-slot="b" title="沿 Y 法向剖切" disabled>Y</button>
            <button class="viz-btn viz-clip-axis active" data-axis="z" data-slot="b" title="沿 Z 法向剖切" disabled>Z</button>
            <input type="range" class="viz-clip-slider" id="viz-clip-slider-b" min="0" max="1000" value="500" step="1" disabled title="拖动剖切平面位置">
            <span class="viz-clip-display" id="viz-clip-label-b">Z 50%</span>
            <button class="viz-btn viz-clip-flip" id="viz-clip-flip-b" data-slot="b" title="翻转保留的一侧（类似 ABAQUS view-cut 的 Above/Below）" disabled>翻转</button>
          </div>
        </div>

        <div class="viz-toolbar viz-common-toolbar" id="viz-common-toolbar" style="display:none">
          <div class="viz-group viz-common-label-group">
            <span class="viz-group-label">通用功能</span>
          </div>
          <div class="viz-sep viz-common-divider"></div>
          <div class="viz-group viz-element-select-group" id="viz-element-select-group">
            <button class="viz-btn viz-element-select-toggle" id="viz-element-select-toggle" title="单元选取：左键追加，右键单击撤销上一步，空格确认">单元选取</button>
          </div>
          <div class="viz-sep viz-common-divider"></div>
          <div class="viz-group viz-probe-group" id="viz-probe-group-a">
            <button class="viz-btn" id="viz-probe-toggle" title="探针：移动鼠标预览，左键固定，Shift+左键旋转">探针</button>
            <div class="viz-probe-mode" id="viz-probe-mode" role="tablist" aria-label="探针模式">
              <button class="viz-btn viz-probe-mode-btn active" data-probe-mode="node" title="节点模式：只对节点字段 (U/NFORC) 做探针" disabled>节点</button>
              <button class="viz-btn viz-probe-mode-btn" data-probe-mode="element" title="单元模式：列出击中单元的所有 Gauss 点值 (S/E/PEEQ)" disabled>单元</button>
            </div>
            <span class="viz-probe-counter" id="viz-probe-counter" title="已固定的探针卡数 / 上限">0/20</span>
            <button class="viz-btn" id="viz-probe-compare" title="比较已固定探针的 frame history" disabled>探针比较</button>
            <button class="viz-btn viz-probe-export-btn" id="viz-probe-export" title="导出所有探针卡到 CSV" disabled>探针导出</button>
            <label class="viz-inline-check viz-probe-frames-wrap" id="viz-probe-all-frames-wrap" title="导出选项：勾选后导出每张探针卡在全部 frame 上的值">
              <span>（</span>
              <input type="checkbox" id="viz-probe-all-frames" disabled>
              <span>全部frames）</span>
            </label>
            <div class="viz-probe-extrema-wrap" title="极值探针：基于当前显示场变量自动添加探针">
              <button class="viz-btn" id="viz-probe-extrema" title="自动选取当前场变量的极值并添加探针：若全部有限值均大于 0，选取 n 个最大值；若同时有正值和负值，选取 round(n/2) 个最大正值和剩余数量的最小负值；若只有非正值，选取 n 个最小值。未勾选可见单元时从全部单元/节点中选，勾选后只从当前可见单元/节点中选。" disabled>极值探针</button>
              <select id="viz-probe-extrema-count" class="viz-probe-extrema-count" aria-label="极值探针数量" disabled>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3" selected>3</option>
                <option value="4">4</option>
                <option value="5">5</option>
                <option value="6">6</option>
                <option value="7">7</option>
                <option value="8">8</option>
                <option value="9">9</option>
                <option value="10">10</option>
              </select>
              <label class="viz-inline-check viz-probe-visible-wrap" id="viz-probe-visible-wrap" title="极值探针选项：勾选后只在当前可见单元/节点中寻找极值">
                <span>（</span>
                <input type="checkbox" id="viz-probe-visible-only" disabled>
                <span>可见单元）</span>
              </label>
            </div>
          </div>
          <div class="viz-sep viz-common-divider"></div>
          <div class="viz-group viz-compare-toolbar" id="viz-compare-toolbar">
            <button class="viz-btn viz-compare-btn" id="viz-compare-toggle" title="切换 A/B 双栏对比">双栏对比</button>
            <button class="viz-btn" id="viz-sync-display" title="将 B 的视角、缩放和剖切锁定到 A；B 的帧与结果量仍可独立选择" disabled>同步显示</button>
            <button class="viz-btn" id="viz-diff-run" title="计算当前 A/B 显示标量的 Diff" disabled>对比</button>
            <label class="viz-inline-check" id="viz-diff-absolute-wrap" title="勾选后计算 |B-A|，未勾选时计算 B-A">
              <input type="checkbox" id="viz-diff-absolute">
              <span>|B-A|</span>
            </label>
            <button class="viz-btn" id="viz-diff-boxplot" title="显示当前 Diff 的箱线图" disabled>箱线图</button>
            <button class="viz-btn" id="viz-diff-save" title="将当前 Diff 下载为 MAT 文件" disabled>保存Diff</button>
          </div>
        </div>

        <div class="viz-canvas-wrap">
          <div class="viz-canvas-panel" id="viz-panel-a">
            <span class="viz-panel-label">A</span>
            <canvas id="viz-canvas-a" width="800" height="600"></canvas>
            <div class="viz-math-overlay" id="viz-math-overlay-a" style="display:none"></div>
            <div class="viz-probe-pin-host" id="viz-probe-pin-host-a"></div>
            <div class="viz-element-box" id="viz-element-box-a" style="display:none"></div>
            <div class="viz-empty" id="viz-empty-a"></div>
          </div>
          <div class="viz-canvas-panel" id="viz-panel-b" style="display:none">
            <span class="viz-panel-label">B</span>
            <canvas id="viz-canvas-b" width="800" height="600"></canvas>
            <div class="viz-math-overlay" id="viz-math-overlay-b" style="display:none"></div>
            <div class="viz-probe-pin-host" id="viz-probe-pin-host-b"></div>
            <div class="viz-element-box" id="viz-element-box-b" style="display:none"></div>
            <div class="viz-empty" id="viz-empty-b"></div>
          </div>
          <div class="viz-probe-hover-card" id="viz-probe-hover-card" style="display:none"></div>
          <div class="viz-element-select-panel" id="viz-element-select-panel" style="display:none">
            <div class="viz-element-select-panel-head">
              <div>
                <div class="viz-element-select-panel-title">单元选取</div>
                <div class="viz-element-select-panel-subtitle">框选或按编号追加选择；移动鼠标可预览当前结果量探针；空格隐藏选中</div>
              </div>
              <button class="viz-btn" id="viz-element-panel-close" title="关闭单元选取">关闭</button>
            </div>
            <div class="viz-element-select-tabs" role="tablist" aria-label="单元选取方式">
              <button class="viz-btn viz-element-tab active" data-element-tab="box">框选</button>
              <button class="viz-btn viz-element-tab" data-element-tab="ids">编号选</button>
            </div>
            <div class="viz-element-tab-body" data-element-tab-body="box">
              <div class="viz-element-select-help">左键单击或拖框追加；Shift+左键转动视角；右键单击撤销上一步，右键拖动仍平移模型。</div>
              <div class="viz-element-select-panel-actions">
                <button class="viz-btn" data-element-action="hide" title="隐藏当前选中的单元" disabled>隐藏选中</button>
                <button class="viz-btn" data-element-action="show-only" title="隐藏未选中的可见单元，只保留当前选中单元" disabled>只显示选中</button>
                <button class="viz-btn" data-element-action="undo" title="撤销最近一次隐藏" disabled>撤销隐藏</button>
                <button class="viz-btn" data-element-action="restore" title="恢复所有隐藏单元" disabled>恢复全部</button>
              </div>
            </div>
            <div class="viz-element-tab-body" data-element-tab-body="ids" style="display:none">
              <div class="viz-element-id-row">
                <label>起始单元
                  <input type="text" id="viz-element-id-start" inputmode="numeric" placeholder="如 18">
                </label>
                <label>结束单元
                  <input type="text" id="viz-element-id-end" inputmode="numeric" placeholder="留空则单选">
                </label>
                <button class="viz-btn" id="viz-element-id-add" title="按输入编号追加选择">选择</button>
              </div>
              <div class="viz-element-select-help">只输入一个编号时只选择该单元；非数字不选；超出范围会按当前模型最大范围截断。</div>
              <div class="viz-element-select-panel-actions">
                <button class="viz-btn" data-element-action="hide" title="隐藏当前选中的单元" disabled>隐藏选中</button>
                <button class="viz-btn" data-element-action="show-only" title="隐藏未选中的可见单元，只保留当前选中单元" disabled>只显示选中</button>
                <button class="viz-btn" data-element-action="undo" title="撤销最近一次隐藏" disabled>撤销隐藏</button>
                <button class="viz-btn" data-element-action="restore" title="恢复所有隐藏单元" disabled>恢复全部</button>
              </div>
            </div>
            <div class="viz-element-select-status" id="viz-element-select-status">选中 0 / 隐藏 0</div>
          </div>
        </div>

        <div class="viz-status-bar" id="viz-status-bar">
          <span id="viz-status-text">就绪</span>
          <span class="viz-status-chip" id="viz-source-a" style="margin-left:auto;display:none"></span>
          <span class="viz-status-chip" id="viz-source-b" style="display:none"></span>
        </div>

        <div class="viz-modal" id="viz-boxplot-modal" style="display:none">
          <div class="viz-modal-backdrop" id="viz-boxplot-backdrop"></div>
          <div class="viz-modal-card">
            <div class="viz-modal-header">
              <div>
                <div class="viz-modal-title">差值箱线图</div>
                <div class="viz-modal-subtitle" id="viz-boxplot-subtitle"></div>
              </div>
              <div class="viz-modal-actions">
                <button class="viz-btn" id="viz-boxplot-screenshot">截图</button>
                <button class="viz-btn" id="viz-boxplot-close">关闭</button>
              </div>
            </div>
            <div class="viz-boxplot-wrap">
              <svg id="viz-boxplot-svg" viewBox="0 0 820 240" preserveAspectRatio="xMidYMid meet"></svg>
            </div>
            <div class="viz-boxplot-stats" id="viz-boxplot-stats"></div>
          </div>
        </div>

        <div class="viz-modal" id="viz-probe-compare-modal" style="display:none">
          <div class="viz-modal-backdrop" id="viz-probe-compare-backdrop"></div>
          <div class="viz-modal-card viz-probe-compare-modal-card">
            <div class="viz-modal-header">
              <div>
                <div class="viz-modal-title">探针比较</div>
                <div class="viz-modal-subtitle">选择两个探针的 frame history 进行图形化比较；每次重绘会关闭旧图。</div>
              </div>
              <div class="viz-modal-actions">
                <button class="viz-btn primary" id="viz-probe-compare-run">确定</button>
                <button class="viz-btn" id="viz-probe-compare-close">关闭</button>
              </div>
            </div>
            <div class="viz-probe-compare-list" id="viz-probe-compare-list"></div>
          </div>
        </div>

        <div class="viz-modal" id="viz-field-modal" style="display:none">
          <div class="viz-modal-backdrop" id="viz-field-backdrop"></div>
          <div class="viz-modal-card viz-field-modal-card">
            <div class="viz-modal-header">
              <div>
                <div class="viz-modal-title">选择场变量</div>
                <div class="viz-modal-subtitle" id="viz-field-subtitle"></div>
              </div>
              <button class="viz-btn" id="viz-field-close">关闭</button>
            </div>
            <div class="viz-field-toolbar">
              <button class="viz-btn" id="viz-field-defaults">默认变量</button>
              <button class="viz-btn" id="viz-field-all">全选</button>
              <button class="viz-btn" id="viz-field-clear">清空</button>
              <span class="viz-field-count" id="viz-field-count"></span>
            </div>
            <div class="viz-field-list" id="viz-field-list"></div>
            <div class="viz-field-actions">
              <button class="viz-btn" id="viz-field-cancel">取消</button>
              <button class="viz-btn primary" id="viz-field-confirm">开始可视化</button>
            </div>
          </div>
        </div>
      </div>`;
  },

  _vizBindEvents() {
    ['a', 'b'].forEach(slot => {
      document.getElementById(`viz-model-${slot}`)?.addEventListener('change', async (e) => {
        const state = this._vizGetSlotState(slot);
        state.modelKey = e.target.value || '';
        state.matPath = '';
        state.inspect = null;
        state.selectedFields = [];
        state.sourceLocked = false;
        this._vizSetSlotLoadedState(slot, false);
        await this._vizRefreshMatOptions(slot, { force: false });
        this._vizRenderSourceSummary();
      });

      document.getElementById(`viz-mat-${slot}`)?.addEventListener('change', (e) => {
        const state = this._vizGetSlotState(slot);
        state.matPath = e.target.value || '';
        state.inspect = null;
        state.selectedFields = [];
      });

      document.getElementById(`viz-mat-refresh-${slot}`)?.addEventListener('click', () => {
        this._vizRefreshMatOptions(slot, { force: true });
      });

      document.getElementById(`viz-browse-${slot}`)?.addEventListener('click', () => {
        this._vizBrowseMat(slot);
      });

      document.getElementById(`viz-load-${slot}`)?.addEventListener('click', () => {
        this._vizInspectSelectedMat(slot);
      });

      document.getElementById(`viz-reselect-${slot}`)?.addEventListener('click', () => {
        const state = this._vizGetSlotState(slot);
        state.sourceLocked = false;
        this._vizSetSlotLoadedState(slot, false);
      });

      document.getElementById(`viz-family-${slot}`)?.addEventListener('change', (e) => {
        const state = this._vizGetSlotState(slot);
        state.family = e.target.value || '';
        state.quantityValue = '';
        this._vizPopulateFieldSelectors(slot, { preferState: true });
        const qtySel = document.getElementById(`viz-quantity-${slot}`);
        if (qtySel && qtySel.value) {
          this._vizApplyFieldSelection(slot, qtySel.value);
        }
      });

      document.getElementById(`viz-quantity-${slot}`)?.addEventListener('change', (e) => {
        const value = e.target.value || '';
        const state = this._vizGetSlotState(slot);
        state.quantityValue = value;
        this._vizApplyFieldSelection(slot, value);
      });

      document.getElementById(`viz-frame-slider-${slot}`)?.addEventListener('input', e => {
        const frame = parseInt(e.target.value);
        this._vizSetFrame(slot, frame);
      });

      document.getElementById(`viz-clim-apply-${slot}`)?.addEventListener('click', () => {
        const vmin = parseFloat(document.getElementById(`viz-clim-min-${slot}`).value);
        const vmax = parseFloat(document.getElementById(`viz-clim-max-${slot}`).value);
        document.getElementById(`viz-clim-auto-${slot}`)?.classList.remove('active');
        this._vizSendWs(slot, {
          action: 'set_legend',
          vmin: isNaN(vmin) ? null : vmin,
          vmax: isNaN(vmax) ? null : vmax,
        });
      });

      document.getElementById(`viz-clim-auto-${slot}`)?.addEventListener('click', () => {
        const btn = document.getElementById(`viz-clim-auto-${slot}`);
        const info = slot === 'a' ? this._viz.info_a : this._viz.info_b;
        const ac = info?.auto_clim;
        const minEl = document.getElementById(`viz-clim-min-${slot}`);
        const maxEl = document.getElementById(`viz-clim-max-${slot}`);
        const wasAuto = !!btn?.classList.contains('active');
        if (wasAuto) {
          const vmin = parseFloat(minEl?.value || (Array.isArray(ac) ? ac[0] : ''));
          const vmax = parseFloat(maxEl?.value || (Array.isArray(ac) ? ac[1] : ''));
          btn?.classList.remove('active');
          this._vizSendWs(slot, {
            action: 'set_legend',
            vmin: isNaN(vmin) ? null : vmin,
            vmax: isNaN(vmax) ? null : vmax,
          });
          return;
        }
        btn?.classList.add('active');
        this._vizSendWs(slot, {action: 'set_legend', vmin: null, vmax: null});
        if (ac && Array.isArray(ac) && minEl && maxEl) {
          minEl.value = ac[0];
          maxEl.value = ac[1];
        } else if (minEl && maxEl) {
          minEl.value = '';
          maxEl.value = '';
        }
      });

      document.getElementById(`viz-legend-font-${slot}`)?.addEventListener('change', (e) => {
        const pt = parseInt(e.target.value, 10);
        if (Number.isFinite(pt)) {
          this._vizSendWs(slot, {action: 'set_legend', font_pt: pt});
        }
      });

      document.getElementById(`viz-frame-prev-${slot}`)?.addEventListener('click', () => {
        this._vizStepFrame(slot, -1);
      });

      document.getElementById(`viz-frame-next-${slot}`)?.addEventListener('click', () => {
        this._vizStepFrame(slot, 1);
      });

      document.getElementById(`viz-frame-play-${slot}`)?.addEventListener('click', () => {
        this._vizTogglePlayback(slot);
      });

      document.getElementById(`viz-frame-detail-${slot}`)?.addEventListener('click', () => {
        this._vizShowFrameDetail(slot);
      });

      const edgeBtn = document.getElementById(`viz-edges-${slot}`);
      if (edgeBtn) {
        edgeBtn._on = false;
        edgeBtn.addEventListener('click', () => {
          edgeBtn._on = !edgeBtn._on;
          edgeBtn.classList.toggle('active', edgeBtn._on);
          this._vizSendWs(slot, {action: 'show_edges', show: edgeBtn._on});
        });
      }

      const overlayBtn = document.getElementById(`viz-overlay-${slot}`);
      if (overlayBtn) {
        overlayBtn._on = false;
        overlayBtn.addEventListener('click', () => {
          overlayBtn._on = !overlayBtn._on;
          overlayBtn.classList.toggle('active', overlayBtn._on);
          const info = this._vizSlotInfo(slot);
          if (info) {
            info.show_overlay = overlayBtn._on;
            this._vizSetSlotInfo(slot, info);
            this._vizUpdateMathOverlay(slot);
          }
          this._vizSendWs(slot, {action: 'show_overlay', show: overlayBtn._on});
        });
      }

      document.getElementById(`viz-screenshot-${slot}`)?.addEventListener('click', () => {
        this._vizDownloadScreenshot(slot);
      });
    });

    document.querySelectorAll('.viz-view-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const slot = btn.dataset.slot || 'a';
        this._vizSendWs(slot, {action: 'set_view', preset: btn.dataset.view});
      });
    });

    document.querySelectorAll('.viz-clip-axis').forEach(btn => {
      btn.addEventListener('click', () => {
        if (btn.disabled) return;
        const slot = btn.dataset.slot || 'a';
        document.querySelectorAll(`.viz-clip-axis[data-slot="${slot}"]`).forEach(el => {
          el.classList.toggle('active', el === btn);
        });
        this._vizUpdateClipLabel(slot, this._vizCurrentClipPosition(slot));
        this._vizQueueClip(slot, {axis: btn.dataset.axis});
      });
    });

    document.querySelectorAll('.viz-clip-flip').forEach(btn => {
      btn.addEventListener('click', () => {
        if (btn.disabled) return;
        const slot = btn.dataset.slot || 'a';
        const currentInvert = !!btn.classList.contains('active');
        const next = !currentInvert;
        btn.classList.toggle('active', next);
        this._vizQueueClip(slot, {invert: next});
      });
    });

    // 剖切 is now a single toggle button (replaced the old label + 启用
    // checkbox). Clicking it flips the clip on/off; the axis / slider /
    // 翻转 controls only become enabled while clipping is on.
    document.querySelectorAll('.viz-clip-toggle').forEach(btn => {
      btn.addEventListener('click', () => {
        const slot = btn.dataset.slot || 'a';
        const next = !btn.classList.contains('active');
        this._vizSetClipEnabledUI(slot, next);
        this._vizQueueClip(slot, {enabled: next});
      });
    });

    ['a', 'b'].forEach(slot => {
      const slider = document.getElementById(`viz-clip-slider-${slot}`);
      if (slider) {
        slider.addEventListener('input', (e) => {
          const pos = parseInt(e.target.value, 10) / 1000;
          this._vizUpdateClipLabel(slot, pos);
          this._vizQueueClip(slot, {position: pos});
        });
      }
    });

    document.getElementById('viz-compare-toggle')?.addEventListener('click', async () => {
      if (this._viz.compareLocked) {
        this._viz.compareMode = true;
        this._vizShowCompare(true);
        this._vizSetStatus('Validation 结果固定使用 JAX / ABAQUS 双栏对比。');
        return;
      }
      const enabled = !this._viz.compareMode;
      this._viz.compareMode = enabled;
      this._vizShowCompare(enabled);
      const res = await this.apiPost('/api/viz/compare', {enabled}).catch(() => null);
      if (res?.compare_locked) {
        this._viz.compareLocked = true;
        this._viz.compareMode = true;
        this._vizShowCompare(true);
        this._vizSetStatus('Validation 结果固定使用 JAX / ABAQUS 双栏对比。');
        return;
      }
      if (!enabled) {
        this._vizTeardownSlotB();
      } else if (res && res.scene_b) {
        this._viz.info_b = res.scene_b;
        this._vizUpdateControls('b');
        this._vizConnectWs('b');
      }
      this._vizRenderSourceSummary();
    });

    const syncBtn = document.getElementById('viz-sync-display');
    if (syncBtn) {
      syncBtn._on = false;
      syncBtn.addEventListener('click', () => {
        syncBtn._on = !syncBtn._on;
        syncBtn.classList.toggle('active', syncBtn._on);
        this._viz.syncDisplay = syncBtn._on;
        this._vizApplySyncDisplayLock();
        this.apiPost('/api/viz/sync-display', {enabled: syncBtn._on}).catch(() => {});
      });
    }

    document.getElementById('viz-diff-absolute')?.addEventListener('change', (e) => {
      this._viz.absoluteDiff = !!e.target.checked;
      this._vizRefreshCompareButtons();
    });

    document.getElementById('viz-diff-boxplot')?.addEventListener('click', () => {
      this._vizOpenDiffBoxplot();
    });

    document.getElementById('viz-diff-run')?.addEventListener('click', () => {
      this._vizRunDiff();
    });

    document.getElementById('viz-diff-save')?.addEventListener('click', () => {
      this._vizSaveDiffMat();
    });
    document.getElementById('viz-boxplot-screenshot')?.addEventListener('click', () => {
      this._vizDownloadBoxplotScreenshot();
    });
    document.getElementById('viz-boxplot-close')?.addEventListener('click', () => {
      this._vizHideBoxplot();
    });
    document.getElementById('viz-boxplot-backdrop')?.addEventListener('click', () => {
      this._vizHideBoxplot();
    });

    document.getElementById('viz-field-close')?.addEventListener('click', () => this._vizShowFieldModal(false));
    document.getElementById('viz-field-cancel')?.addEventListener('click', () => this._vizShowFieldModal(false));
    document.getElementById('viz-field-backdrop')?.addEventListener('click', () => this._vizShowFieldModal(false));
    document.getElementById('viz-field-defaults')?.addEventListener('click', () => this._vizSetFieldChecks('defaults'));
    document.getElementById('viz-field-all')?.addEventListener('click', () => this._vizSetFieldChecks('all'));
    document.getElementById('viz-field-clear')?.addEventListener('click', () => this._vizSetFieldChecks('none'));
    document.getElementById('viz-field-confirm')?.addEventListener('click', () => this._vizConfirmFieldSelection());

    ['a', 'b'].forEach(slot => {
      const canvas = document.getElementById(`viz-canvas-${slot}`);
      if (!canvas) return;
      canvas.addEventListener('contextmenu', e => e.preventDefault());
      canvas.addEventListener('mousedown', e => {
        if (this._viz.elementSelectMode && this._viz.elementSelectTab === 'box' && !this._vizIsSyncLockedSlot(slot)) {
          if (e.button === 0 && !e.shiftKey) {
            e.preventDefault();
            e.stopPropagation();
            this._vizElementBoxStart(slot, e.clientX, e.clientY, 'add');
            return;
          }
          if (e.button === 2) {
            this._viz.elementRightClickCandidate = {
              slot,
              startClientX: e.clientX,
              startClientY: e.clientY,
              moved: false,
            };
            // Do not return: right-drag must keep the normal pan behavior.
          }
        }
        // In probe mode a plain left-click pins the current hover card.
        // Shift + left-click keeps the usual orbit-drag behavior so the
        // user can reframe without leaving probe mode.
        if (this._viz.probeMode && e.button === 0 && !e.shiftKey) {
          e.preventDefault();
          e.stopPropagation();
          this._vizProbePinAt(slot, e.clientX, e.clientY);
          return;
        }
        if (this._vizIsSyncLockedSlot(slot)) {
          e.preventDefault();
          return;
        }
        this._viz.dragging = true;
        this._viz.dragSlot = slot;
        this._viz.lastX = e.clientX;
        this._viz.lastY = e.clientY;
        this._viz.dragButton = e.button;
      });
      // Throttled hover-probe (Step 3b): only active while probe mode
      // is on. The handler runs alongside the document-level orbit
      // mousemove; while a right-button pan is in progress we suppress
      // probe so the user can reframe without spamming pick requests.
      canvas.addEventListener('mousemove', e => {
        if (this._viz.elementSelectMode) {
          if (this._viz.dragging || this._viz.elementBoxDrag) return;
          this._vizProbeHoverThrottled(slot, e, { mode: this._vizProbeModeForSlot(slot) });
          return;
        }
        if (!this._viz.probeMode) return;
        if (this._viz.dragging) return;
        this._vizProbeHoverThrottled(slot, e);
      });
      canvas.addEventListener('mouseleave', () => {
        if (this._viz.probeMode || this._viz.elementSelectMode) this._vizProbeHideHover();
      });
      canvas.addEventListener('wheel', e => {
        e.preventDefault();
        if (this._vizIsSyncLockedSlot(slot)) return;
        const factor = e.deltaY < 0 ? 1.1 : 0.9;
        this._vizSendWs(slot, {action: 'zoom', factor});
      }, {passive: false});
    });

    // Probe toolbar wiring (A only — B canvas reuses the same mode
    // and state; per plan we keep one global toggle).
    document.getElementById('viz-probe-toggle')?.addEventListener('click', () => {
      this._vizProbeToggle();
    });
    document.querySelectorAll('.viz-probe-mode-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const kind = btn.dataset.probeMode || 'node';
        this._vizProbeSetKind(kind);
      });
    });
    document.getElementById('viz-probe-export')?.addEventListener('click', () => {
      this._vizProbeExportCsv();
    });
    document.getElementById('viz-probe-compare')?.addEventListener('click', () => {
      this._vizProbeOpenCompare();
    });
    document.getElementById('viz-probe-extrema')?.addEventListener('click', () => {
      this._vizProbeRunExtrema();
    });
    document.getElementById('viz-probe-compare-close')?.addEventListener('click', () => this._vizProbeShowCompareModal(false));
    document.getElementById('viz-probe-compare-backdrop')?.addEventListener('click', () => this._vizProbeShowCompareModal(false));
    document.getElementById('viz-probe-compare-run')?.addEventListener('click', () => this._vizProbeRunCompare());
    document.getElementById('viz-element-select-toggle')?.addEventListener('click', () => this._vizElementSelectionToggle());
    document.getElementById('viz-element-panel-close')?.addEventListener('click', () => {
      if (this._viz.elementSelectMode) this._vizElementSelectionToggle();
    });
    document.querySelectorAll('.viz-element-tab').forEach(btn => {
      btn.addEventListener('click', () => this._vizElementSelectionSetTab(btn.dataset.elementTab || 'box'));
    });
    document.getElementById('viz-element-id-add')?.addEventListener('click', () => this._vizElementSelectionSelectIds());
    document.querySelectorAll('[data-element-action]').forEach(btn => {
      btn.addEventListener('click', () => {
        const action = btn.dataset.elementAction || '';
        if (action === 'hide') this._vizElementSelectionCommand('element_hide_selected');
        else if (action === 'show-only') this._vizElementSelectionCommand('element_show_selected_only');
        else if (action === 'undo') this._vizElementSelectionCommand('element_undo_hidden');
        else if (action === 'restore') this._vizElementSelectionCommand('element_restore_all');
      });
    });

    document.addEventListener('mousemove', e => {
      if (this._viz.elementBoxDrag) {
        this._vizElementBoxUpdate(e.clientX, e.clientY);
        return;
      }
      if (!this._viz.dragging) return;
      const rightCandidate = this._viz.elementRightClickCandidate;
      if (rightCandidate) {
        const moved = Math.abs(e.clientX - rightCandidate.startClientX) > 4
          || Math.abs(e.clientY - rightCandidate.startClientY) > 4;
        if (moved) rightCandidate.moved = true;
      }
      const dx = e.clientX - this._viz.lastX;
      const dy = e.clientY - this._viz.lastY;
      this._viz.lastX = e.clientX;
      this._viz.lastY = e.clientY;
      const slot = this._viz.dragSlot || 'a';
      if (this._vizIsSyncLockedSlot(slot)) return;
      if (this._viz.dragButton === 2) {
        this._vizSendWs(slot, {action: 'pan', dx, dy});
      } else {
        this._vizSendWs(slot, {action: 'orbit', dx, dy});
      }
    });

    document.addEventListener('mouseup', e => {
      if (this._viz.elementBoxDrag) {
        this._vizElementBoxFinish(e.clientX, e.clientY);
        return;
      }
      const rightCandidate = this._viz.elementRightClickCandidate;
      if (rightCandidate) {
        this._viz.elementRightClickCandidate = null;
        if (!rightCandidate.moved) {
          this._vizSendWs(rightCandidate.slot, { action: 'element_selection_undo' });
        }
      }
      this._viz.dragging = false;
    });

    const spaceConfirmCapture = e => {
      if (!this._viz.elementSelectMode) return;
      if (e.key !== ' ' && e.key !== 'Spacebar' && e.code !== 'Space') return;
      const target = e.target;
      const tag = String(target?.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || target?.isContentEditable) return;
      e.preventDefault();
      e.stopPropagation();
      if (typeof e.stopImmediatePropagation === 'function') e.stopImmediatePropagation();
      if (e.type === 'keydown' && !e.repeat) this._vizElementSelectionConfirm();
    };
    document.addEventListener('keydown', spaceConfirmCapture, true);
    document.addEventListener('keyup', spaceConfirmCapture, true);
  },

  async _vizEnsureModels() {
    const models = Array.isArray(this.models) && this.models.length
      ? this.models
      : await this.api('/api/models').catch(() => []);
    this.models = Array.isArray(models) ? models : [];
    this._viz.models = this.models;
    this._viz.modelByKey = {};
    this._viz.models.forEach(model => {
      this._viz.modelByKey[this._vizModelKey(model)] = model;
    });
    return this._viz.models;
  },

  _vizModelKey(model) {
    return model ? `${model.source}/${model.name}` : '';
  },

  _vizGetSlotState(slot) {
    return this._viz.slotState[slot] || this._viz.slotState.a;
  },

  _vizGetModelByKey(modelKey) {
    return this._viz.modelByKey[modelKey] || null;
  },

  _vizGetDefaultModelKey(slot) {
    if (slot === 'a' && this.selectedModel) {
      const current = this._vizModelKey(this.selectedModel);
      if (current && this._vizGetModelByKey(current)) return current;
    }
    if (slot === 'b') {
      const slotAKey = this._vizGetSlotState('a').modelKey;
      if (slotAKey && this._vizGetModelByKey(slotAKey)) return slotAKey;
    }
    return this._viz.models[0] ? this._vizModelKey(this._viz.models[0]) : '';
  },

  async _vizInitModelSelectors() {
    ['a', 'b'].forEach(slot => {
      const state = this._vizGetSlotState(slot);
      if (!state.modelKey) state.modelKey = this._vizGetDefaultModelKey(slot);
      this._vizPopulateModelSelect(slot);
    });
    await Promise.all(['a', 'b'].map(slot => this._vizRefreshMatOptions(slot, { force: false })));
  },

  _vizPopulateModelSelect(slot) {
    const sel = document.getElementById(`viz-model-${slot}`);
    if (!sel) return;
    const state = this._vizGetSlotState(slot);
    if (!this._viz.models.length) {
      sel.innerHTML = '<option value="">未找到模型</option>';
      sel.disabled = true;
      return;
    }
    const options = this._viz.models.map(model => {
      const value = this._vizModelKey(model);
      const sourceLabel = model.source === 'stored' ? 'StoredModels' : 'Examples';
      const selected = value === state.modelKey ? 'selected' : '';
      return `<option value="${UI.escapeAttr(value)}" ${selected}>${UI.escapeHtml(sourceLabel)} / ${UI.escapeHtml(model.name)}</option>`;
    }).join('');
    sel.innerHTML = `<option value="">选择模型</option>${options}`;
    sel.disabled = false;
    if (state.modelKey) sel.value = state.modelKey;
  },

  _vizPopulateMatSelect(slot) {
    const sel = document.getElementById(`viz-mat-${slot}`);
    if (!sel) return;
    const state = this._vizGetSlotState(slot);
    const mats = Array.isArray(state.mats) ? state.mats : [];
    if (!mats.length) {
      sel.innerHTML = '<option value="">未找到 MAT 结果</option>';
      sel.disabled = true;
      return;
    }
    const options = mats.map(mat => {
      const display = mat.rel_path || mat.name || mat.abs_path;
      const selected = this._normalizeWinPath(mat.abs_path) === this._normalizeWinPath(state.matPath) ? 'selected' : '';
      return `<option value="${UI.escapeAttr(mat.abs_path)}" ${selected}>${UI.escapeHtml(display)}</option>`;
    }).join('');
    sel.innerHTML = `<option value="">选择 MAT 结果</option>${options}`;
    sel.disabled = false;
    if (state.matPath) sel.value = state.matPath;
  },

  async _vizRefreshMatOptions(slot, { force = false, preferPath = '' } = {}) {
    const state = this._vizGetSlotState(slot);
    const sel = document.getElementById(`viz-mat-${slot}`);
    if (sel) {
      sel.innerHTML = '<option value="">正在扫描 MAT 结果...</option>';
      sel.disabled = true;
    }
    const model = this._vizGetModelByKey(state.modelKey);
    if (!model) {
      state.mats = [];
      state.matPath = '';
      this._vizPopulateMatSelect(slot);
      return;
    }
    if (force || !this._viz.matCache[state.modelKey]) {
      const scanRes = await this.apiPost('/api/models/scan-mats', { model_path: model.path }).catch(() => null);
      this._viz.matCache[state.modelKey] = Array.isArray(scanRes?.mat_files) ? scanRes.mat_files : [];
    }
    state.mats = this._viz.matCache[state.modelKey] || [];
    if (preferPath) {
      const matched = this._vizMatchMatPath(state.mats, preferPath);
      if (matched) state.matPath = matched.abs_path;
    }
    if (!state.matPath || !this._vizMatchMatPath(state.mats, state.matPath)) {
      state.matPath = state.mats[0]?.abs_path || '';
    }
    this._vizPopulateMatSelect(slot);
  },

  async _vizHydrateSlotFromSource(slot, sourcePath) {
    const normalizedSource = this._normalizeWinPath(sourcePath);
    if (!normalizedSource || !this._viz.models.length) return;
    const matchedModel = this._viz.models.find(model => {
      const modelRoot = this._normalizeWinPath(model.path);
      return normalizedSource === modelRoot || normalizedSource.startsWith(`${modelRoot}\\`);
    });
    if (!matchedModel) return;
    const state = this._vizGetSlotState(slot);
    state.modelKey = this._vizModelKey(matchedModel);
    this._vizPopulateModelSelect(slot);
    await this._vizRefreshMatOptions(slot, { force: false, preferPath: normalizedSource });
  },

  _vizMatchMatPath(mats, targetPath) {
    const normalizedTarget = this._normalizeWinPath(targetPath);
    return (Array.isArray(mats) ? mats : []).find(item => this._normalizeWinPath(item.abs_path) === normalizedTarget) || null;
  },

  _vizPathBelongsToModel(path, modelPath) {
    const target = this._normalizeWinPath(path);
    const root = this._normalizeWinPath(modelPath);
    return !!target && !!root && (target === root || target.startsWith(`${root}\\`));
  },

  _vizSetSlotLoadedState(slot, loaded) {
    const toolbar = document.getElementById(`viz-toolbar-${slot}`);
    const picker = document.getElementById(`viz-source-picker-${slot}`);
    const current = document.getElementById(`viz-source-current-${slot}`);
    const currentText = document.getElementById(`viz-source-current-text-${slot}`);
    const reselectBtn = document.getElementById(`viz-reselect-${slot}`);
    const state = this._vizGetSlotState(slot);
    // When a slot transitions from "loaded" to "not loaded" (e.g. user
    // clicks "重选MAT" or the MAT is unloaded), drop any pinned probe
    // cards for that slot — they reference scene data that is about
    // to disappear. ``_vizProbeClearForSlot`` is a no-op if probe is
    // not active or there are no cards.
    if (state.sourceLocked && !loaded && this._vizProbeClearForSlot) {
      this._vizProbeClearForSlot(slot);
      this._vizElementSelectionClearForSlot(slot);
    }
    state.sourceLocked = !!loaded;
    if (picker) picker.style.display = loaded ? 'none' : 'inline-flex';
    if (current) current.style.display = loaded ? 'inline-flex' : 'none';
    if (currentText) {
      const model = this._vizGetModelByKey(state.modelKey);
      const modelText = model ? model.name : 'No model';
      currentText.textContent = `${slot.toUpperCase()} · ${modelText} · ${this._vizBasename(state.matPath)}`;
      currentText.title = state.matPath || '';
    }
    // Lock slot B's "重选MAT" while a validation MAT pair is active — both
    // scenes come from the same embedded-ODB file, so reselecting B alone
    // would silently break the A/B comparison semantics.
    if (reselectBtn) {
      const pairLocked = slot === 'b' && !!this._viz.compareLocked;
      reselectBtn.disabled = pairLocked;
      reselectBtn.title = pairLocked
        ? '当前 A/B 来自同一个 validation MAT，B 不能单独重选'
        : '重新选择 MAT 结果';
    }
    if (toolbar) {
      Array.from(toolbar.children).forEach((child, index) => {
        if (index === 0) return;
        child.style.display = loaded ? '' : 'none';
      });
    }
    if (slot === 'a') {
      const commonToolbar = document.getElementById('viz-common-toolbar');
      if (commonToolbar) commonToolbar.style.display = loaded ? 'flex' : 'none';
      if (!loaded && this._viz.elementSelectMode) this._vizElementSelectionToggle();
    }
  },

  async _vizInspectSelectedMat(slot, overridePath = '') {
    const state = this._vizGetSlotState(slot);
    const path = overridePath || state.matPath;
    if (!path) {
      this._vizSetStatus('请先选择 MAT 结果文件。');
      return;
    }
    state.matPath = path;
    this._vizSetStatus(`正在读取场变量目录 ${slot.toUpperCase()}...`);
    const res = await this.apiPost('/api/viz/inspect', { mat_path: path }).catch(() => null);
    if (!res || !res.ok) {
      this._vizSetStatus(`读取 MAT 变量失败：${res?.detail || '未知错误'}`);
      return;
    }
    state.inspect = res;
    state.selectedFields = (res.fields || []).filter(item => item.default_selected).map(item => item.key);
    this._viz.fieldModalSlot = slot;
    this._vizRenderFieldModal();
    this._vizShowFieldModal(true);
    this._vizSetStatus(`已读取 ${res.fields?.length || 0} 个候选场变量`);
  },

  async _vizLoadSelectedMat(slot, overridePath = '', selectedFields = null) {
    this._vizStopPlayback(slot);
    const state = this._vizGetSlotState(slot);
    const path = overridePath || state.matPath;
    if (!path) {
      this._vizSetStatus('请先选择 MAT 结果文件。');
      return;
    }
    const fields = Array.isArray(selectedFields) ? selectedFields : state.selectedFields;
    if (!fields || !fields.length) {
      this._vizSetStatus('请至少选择一个场变量。');
      return;
    }
    this._vizSetStatus(`正在加载场景 ${slot.toUpperCase()}...`);
    const res = await this.apiPost('/api/viz/load', { mat_path: path, slot, selected_fields: fields }).catch(() => null);
    if (!res || !res.ok) {
      this._vizSetStatus(`加载失败：${res?.detail || '未知错误'}`);
      return;
    }
    state.selectedFields = fields;
    state.family = '';
    state.quantityValue = '';
    if (slot === 'a') {
      const isValidationPair = !!(res.validation_compare && res.scene_b);
      // When the backend reports that a validation lock was just released
      // (user is loading a regular MAT on top of a previous validation
      // pair), tear down scene B before installing new A info so the UI
      // matches the backend's now-single-pane state.
      const exitingValidation = !isValidationPair && (
        res.unlocked_validation || (this._viz.compareLocked && !res.compare_locked)
      );
      if (exitingValidation) {
        this._vizTeardownSlotB();
      }
      this._viz.info_a = isValidationPair ? res.scene_a : res;
      this._viz.compareLocked = !!res.compare_locked;
      this._vizUpdateControls('a');
      this._vizConnectWs('a');
      if (isValidationPair) {
        this._viz.info_b = res.scene_b;
        this._viz.compareMode = true;
        this._vizShowCompare(true);
        const stateB = this._vizGetSlotState('b');
        stateB.matPath = res.scene_b.source || path;
        stateB.selectedFields = Array.isArray(res.scene_b.fields) ? res.scene_b.fields : [];
        stateB.family = '';
        stateB.quantityValue = '';
        await this._vizHydrateSlotFromSource('b', res.scene_b.source || path);
        this._vizPopulateMatSelect('b');
        this._vizSetSlotLoadedState('b', true);
        this._vizUpdateControls('b');
        this._vizConnectWs('b');
      } else {
        this._viz.compareLocked = false;
      }
    } else {
      this._viz.info_b = res;
      this._viz.compareLocked = false;
      this._viz.compareMode = true;
      this._vizShowCompare(true);
      this._vizUpdateControls('b');
      this._vizConnectWs('b');
    }
    await this._vizHydrateSlotFromSource(slot, res.source || path);
    const refreshedState = this._vizGetSlotState(slot);
    refreshedState.matPath = res.source || path;
    this._vizPopulateMatSelect(slot);
    this._vizSetSlotLoadedState(slot, true);
    this._vizRenderSourceSummary();
    this._vizSetStatus(`场景 ${slot.toUpperCase()} 已加载`);
  },

  _vizShowFieldModal(show) {
    const modal = document.getElementById('viz-field-modal');
    if (!modal) return;
    modal.style.display = show ? 'flex' : 'none';
  },

  _vizFieldSortItems(fields) {
    const sort = this._viz.fieldSortKey || 'default';
    const dir = this._viz.fieldSortDir === 'desc' ? -1 : 1;
    const arr = [...(fields || [])];
    const text = item => String(item.display_key || item.key || '').toLowerCase();
    const orderMap = (values, value) => {
      const idx = values.indexOf(String(value || ''));
      return idx < 0 ? values.length : idx;
    };
    const cmpText = (a, b) => text(a).localeCompare(text(b));
    const cmpShape = (a, b) => {
      const size = item => (item.shape || []).reduce((acc, v) => acc * Math.max(1, Number(v) || 1), 1);
      return size(a) - size(b);
    };
    arr.sort((a, b) => {
      let result = 0;
      if (sort === 'family') {
        result = orderMap(['stress', 'strain', 'displacement', 'force', 'other'], a.family) - orderMap(['stress', 'strain', 'displacement', 'force', 'other'], b.family) || cmpText(a, b);
      } else if (sort === 'location') {
        result = orderMap(['node', 'element', 'gauss', 'unknown'], a.location) - orderMap(['node', 'element', 'gauss', 'unknown'], b.location) || cmpText(a, b);
      } else if (sort === 'frames') {
        result = orderMap(['single', 'frames'], a.frames) - orderMap(['single', 'frames'], b.frames) || cmpText(a, b);
      } else if (sort === 'tensor') {
        result = orderMap(['stress_voigt', 'strain_voigt', 'vector', 'scalar', 'components'], a.tensor_kind) - orderMap(['stress_voigt', 'strain_voigt', 'vector', 'scalar', 'components'], b.tensor_kind) || cmpText(a, b);
      } else if (sort === 'symbol') {
        result = String(a.symbol || '').localeCompare(String(b.symbol || '')) || cmpText(a, b);
      } else if (sort === 'shape') {
        result = cmpShape(a, b) || cmpText(a, b);
      } else if (sort === 'name') {
        result = cmpText(a, b);
      } else {
        result = Number(!a.default_selected) - Number(!b.default_selected)
          || orderMap(['stress', 'strain', 'displacement', 'force', 'other'], a.family) - orderMap(['stress', 'strain', 'displacement', 'force', 'other'], b.family)
          || cmpText(a, b);
      }
      return sort === 'default' ? result : result * dir;
    });
    return arr;
  },

  _vizSetFieldSort(key) {
    const next = key || 'default';
    if (this._viz.fieldSortKey === next && next !== 'default') {
      this._viz.fieldSortDir = this._viz.fieldSortDir === 'asc' ? 'desc' : 'asc';
    } else {
      this._viz.fieldSortKey = next;
      this._viz.fieldSortDir = 'asc';
    }
    this._vizRenderFieldModal();
  },

  _vizSortHeader(key, label) {
    const active = this._viz.fieldSortKey === key;
    const arrow = active ? (this._viz.fieldSortDir === 'desc' ? ' ↓' : ' ↑') : '';
    return `<button type="button" class="viz-field-sort-btn ${active ? 'active' : ''}" data-sort="${UI.escapeAttr(key)}">${UI.escapeHtml(label)}${arrow}</button>`;
  },

  _vizRenderFieldTableHeader(fields, selected) {
    const allChecked = fields.length > 0 && fields.every(item => selected.has(item.key));
    return `
      <div class="viz-field-table-head">
        <label class="viz-field-head-check">
          <input type="checkbox" id="viz-field-check-all" ${allChecked ? 'checked' : ''}>
        </label>
        <div>${this._vizSortHeader('name', '名称')}</div>
        <div>${this._vizSortHeader('location', '位置')}</div>
        <div>${this._vizSortHeader('frames', '帧')}</div>
        <div>${this._vizSortHeader('tensor', '类型')}</div>
        <div>${this._vizSortHeader('family', '物理量')}</div>
        <div>${this._vizSortHeader('symbol', '符号')}</div>
        <div>${this._vizSortHeader('shape', 'Shape')}</div>
        <div>${this._vizSortHeader('default', '默认')}</div>
      </div>`;
  },

  _vizBindFieldTableEvents() {
    document.querySelectorAll('#viz-field-list .viz-field-sort-btn').forEach(btn => {
      btn.addEventListener('click', () => this._vizSetFieldSort(btn.dataset.sort || 'default'));
    });
    document.getElementById('viz-field-check-all')?.addEventListener('change', e => {
      this._vizSetFieldChecks(e.target.checked ? 'all' : 'none');
    });
    document.querySelectorAll('#viz-field-list .viz-field-check').forEach(input => {
      input.addEventListener('change', () => this._vizStoreFieldChecks());
    });
  },

  _vizRenderFieldModal() {
    const slot = this._viz.fieldModalSlot || 'a';
    const state = this._vizGetSlotState(slot);
    const inspect = state.inspect || {};
    const fields = this._vizFieldSortItems(inspect.fields || []);
    const selected = new Set(state.selectedFields || []);
    const subtitle = document.getElementById('viz-field-subtitle');
    if (subtitle) subtitle.textContent = `${slot.toUpperCase()} · ${this._vizBasename(inspect.source || state.matPath)} · 选择需要载入 VizData 的场变量`;
    const list = document.getElementById('viz-field-list');
    if (!list) return;
    if (!fields.length) {
      list.innerHTML = '<div class="viz-field-empty">未发现可识别的数值场变量。</div>';
      this._vizUpdateFieldCount();
      return;
    }
    const rows = fields.map(item => {
      const checked = selected.has(item.key) ? 'checked' : '';
      const shape = Array.isArray(item.shape) ? item.shape.join(' × ') : '-';
      const family = this._vizFamilyLabel(item.family === 'force' ? 'reaction' : item.family);
      const scalarNote = this._vizFieldScalarNote(item);
      const symbol = String(item.symbol || '').trim();
      const formula = String(item.formula || '').trim();
      const description = String(item.description || '').trim();
      const symbolHtml = symbol ? `\\(${UI.escapeHtml(symbol)}\\)` : '-';
      const displayKey = String(item.display_key || item.key || '').trim();
      const mainTitle = [item.label || item.key, item.key && item.key !== displayKey ? `MAT key: ${item.key}` : '', description].filter(Boolean).join('\n');
      const symbolTitle = [formula || symbol, description].filter(Boolean).join('\n');
      return `
        <label class="viz-field-table-row">
          <input type="checkbox" class="viz-field-check" value="${UI.escapeAttr(item.key)}" ${checked}>
          <span class="viz-field-main" title="${UI.escapeAttr(mainTitle)}">
            <span class="viz-field-name">${UI.escapeHtml(displayKey || item.key)}</span>
            <span class="viz-field-label">${UI.escapeHtml(item.label || item.key)}</span>
          </span>
          <span class="viz-field-cell">${UI.escapeHtml(item.location || 'unknown')}</span>
          <span class="viz-field-cell">${UI.escapeHtml(item.frames || 'single')}</span>
          <span class="viz-field-cell">${UI.escapeHtml(item.tensor_kind || 'components')}</span>
          <span class="viz-field-cell">${UI.escapeHtml(family)}</span>
          <span class="viz-field-symbol" title="${UI.escapeAttr(symbolTitle)}">${symbolHtml}</span>
          <span class="viz-field-shape">${UI.escapeHtml(shape)}</span>
          <span class="viz-field-note">${UI.escapeHtml(item.default_selected ? scalarNote : '可选')}</span>
        </label>`;
    }).join('');
    list.innerHTML = this._vizRenderFieldTableHeader(fields, selected) + rows;
    this._typesetMath(list);
    this._vizBindFieldTableEvents();
    this._vizUpdateFieldCount();
  },

  _vizFieldScalarNote(item) {
    const n = Number(item.n_components || 1);
    const text = `${item.key || ''} ${item.label || ''} ${item.description || ''}`.toLowerCase();
    if (text.includes('shell generalized stress') && n === 6) return 'components + SF/SM magnitudes';
    if (text.includes('shell generalized strain') && n === 6) return 'components + GE/GK magnitudes';
    if (this._vizIsShellDisplacementRotationField(item.key, item) && n === 6) return 'components + U/UR magnitudes';
    if (this._vizIsShellForceMomentField(item.key, item) && n === 6) return 'components + F/M magnitudes';
    if (this._vizIsShellGeneralizedSubfield(item.key, item) && n === 3) return 'components + Magnitude';
    if (text.includes('plastic strain')) return n > 1 ? `components + Magnitude (${n})` : 'scalar';
    if (item.tensor_kind === 'stress_voigt') return `components + Mises (${n})`;
    if (item.tensor_kind === 'strain_voigt') return `components + Equivalent (${n})`;
    if (item.tensor_kind === 'vector') return n > 1 ? `components + Magnitude (${n})` : 'components + Magnitude';
    return n > 1 ? `components (${n})` : 'scalar';
  },

  _vizStoreFieldChecks() {
    const slot = this._viz.fieldModalSlot || 'a';
    const state = this._vizGetSlotState(slot);
    state.selectedFields = [...document.querySelectorAll('#viz-field-list .viz-field-check')]
      .filter(input => input.checked)
      .map(input => input.value);
    this._vizUpdateFieldCount();
  },

  _vizUpdateFieldCount() {
    const slot = this._viz.fieldModalSlot || 'a';
    const state = this._vizGetSlotState(slot);
    const count = state.selectedFields?.length || 0;
    const el = document.getElementById('viz-field-count');
    if (el) el.textContent = `已选 ${count} 个变量`;
    const confirm = document.getElementById('viz-field-confirm');
    if (confirm) confirm.disabled = count === 0;
  },

  _vizSetFieldChecks(mode) {
    const slot = this._viz.fieldModalSlot || 'a';
    const state = this._vizGetSlotState(slot);
    const fields = state.inspect?.fields || [];
    if (mode === 'all') {
      state.selectedFields = fields.map(item => item.key);
    } else if (mode === 'defaults') {
      state.selectedFields = fields.filter(item => item.default_selected).map(item => item.key);
    } else {
      state.selectedFields = [];
    }
    this._vizRenderFieldModal();
  },

  async _vizConfirmFieldSelection() {
    this._vizStoreFieldChecks();
    const slot = this._viz.fieldModalSlot || 'a';
    const state = this._vizGetSlotState(slot);
    this._vizShowFieldModal(false);
    await this._vizLoadSelectedMat(slot, state.matPath, state.selectedFields);
  },

  async _vizBrowseMat(slot) {
    const state = this._vizGetSlotState(slot);
    const model = this._vizGetModelByKey(state.modelKey);
    const browseRes = await this.apiPost('/api/browse', {
      field_type: 'mat_file',
      title: `为场景 ${slot.toUpperCase()} 选择 MAT 结果`,
      initial_dir: model?.path || 'C:\\',
    }).catch(() => null);
    if (!browseRes?.path) return;
    if (model && this._vizPathBelongsToModel(browseRes.path, model.path)) {
      if (!this._vizMatchMatPath(state.mats, browseRes.path)) {
        await this._vizRefreshMatOptions(slot, { force: true, preferPath: browseRes.path });
      }
      const matched = this._vizMatchMatPath(state.mats, browseRes.path);
      if (matched) {
        state.matPath = matched.abs_path;
        this._vizPopulateMatSelect(slot);
      }
    }
    await this._vizInspectSelectedMat(slot, browseRes.path);
  },

  _vizSetStatus(text, tone = '') {
    const statusEl = document.getElementById('viz-status-text');
    if (statusEl) statusEl.textContent = text;
    const bar = document.getElementById('viz-status-bar');
    if (bar) {
      bar.classList.toggle('viz-status-alert', tone === 'alert');
    }
  },

  _vizTeardownSlotB() {
    // Centralized teardown for scene B (used by the 双栏对比 toggle and by
    // reloading slot A with a non-validation MAT after a validation pair).
    this._vizStopPlayback('b');
    if (this._viz.ws_b) {
      try { this._viz.ws_b.close(); } catch (_) {}
      this._viz.ws_b = null;
    }
    this._viz.info_b = null;
    this._viz.compareMode = false;
    this._viz.compareLocked = false;
    this._vizShowCompare(false);
    this._vizSetSlotLoadedState('b', false);
    const stateB = this._vizGetSlotState('b');
    stateB.matPath = '';
    stateB.selectedFields = [];
    stateB.family = '';
    stateB.quantityValue = '';
    stateB.modelKey = '';
    const emptyEl = document.getElementById('viz-empty-b');
    if (emptyEl) {
      emptyEl.style.display = '';
      emptyEl.textContent = '';
    }
    const canvas = document.getElementById('viz-canvas-b');
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    this._vizHideBoxplot();
    this._vizRenderSourceSummary();
    this._vizRefreshCompareButtons();
  },

  _vizShowCompare(show) {
    const panelB = document.getElementById('viz-panel-b');
    const toolbarB = document.getElementById('viz-toolbar-b');
    const btn = document.getElementById('viz-compare-toggle');
    const syncBtn = document.getElementById('viz-sync-display');
    const diffBtn = document.getElementById('viz-diff-run');
    const optionWraps = [
      document.getElementById('viz-diff-absolute-wrap'),
    ];
    const boxplotBtn = document.getElementById('viz-diff-boxplot');
    const saveBtn = document.getElementById('viz-diff-save');
    const locked = !!this._viz.compareLocked;
    if (panelB) panelB.style.display = show ? '' : 'none';
    if (toolbarB) toolbarB.style.display = show ? '' : 'none';
    if (btn) {
      btn.classList.toggle('active', show);
      btn.classList.toggle('locked', locked);
      btn.disabled = locked;
      btn.title = locked ? 'Validation 结果固定使用 JAX / ABAQUS 双栏对比' : '切换 A/B 双栏对比';
    }
    optionWraps.forEach(wrap => {
      if (wrap) wrap.style.display = 'inline-flex';
    });
    if (!show) {
      this._viz.diffPayload = null;
      this._vizHideBoxplot();
      // Leaving compare mode also drops sync_display; reset the button
      // visual state and unfreeze any locked B controls.
      if (syncBtn) {
        syncBtn._on = false;
        syncBtn.classList.remove('active');
      }
      this._viz.syncDisplay = false;
      this._vizApplySyncDisplayLock();
    }
    this._vizUpdatePanelLabels();
    this._vizRefreshCompareButtons();
    this._vizRenderSourceSummary();
  },

  _vizRefreshCompareButtons() {
    const syncBtn = document.getElementById('viz-sync-display');
    const absoluteCheck = document.getElementById('viz-diff-absolute');
    const boxplotBtn = document.getElementById('viz-diff-boxplot');
    const saveBtn = document.getElementById('viz-diff-save');
    const diffBtn = document.getElementById('viz-diff-run');
    const hasBoth = !!(this._viz.info_a && this._viz.info_b && this._viz.compareMode);
    const hasDiff = this._vizHasDiffField();
    if (syncBtn) syncBtn.disabled = !this._viz.compareMode;
    if (absoluteCheck) {
      absoluteCheck.disabled = !hasBoth;
      absoluteCheck.checked = !!this._viz.absoluteDiff;
    }
    if (boxplotBtn) boxplotBtn.disabled = !hasDiff;
    if (saveBtn) saveBtn.disabled = !hasDiff;
    if (diffBtn) diffBtn.disabled = !hasBoth;
    this._vizApplySyncDisplayLock();
  },

  _vizApplySyncDisplayLock() {
    // Freeze only B controls governed by sync_display: frame, view/camera
    // and clipping. B field / legend controls remain editable so users can
    // compare unlike quantities.
    const toolbarB = document.getElementById('viz-toolbar-b');
    const panelB = document.getElementById('viz-panel-b');
    const on = !!(this._viz && this._viz.syncDisplay && this._viz.compareMode);
    if (toolbarB) toolbarB.classList.toggle('viz-sync-locked', on);
    if (panelB) panelB.classList.toggle('viz-sync-locked', on);
    if (toolbarB) {
      const controls = toolbarB.querySelectorAll(
        '.viz-view-btn, .viz-clip-group button, .viz-clip-group input',
      );
      controls.forEach((el) => {
        if (el.tagName === 'LABEL') {
          el.classList.toggle('viz-control-locked', on);
          const inner = el.querySelector('input');
          if (inner) {
            if (on) {
              if (inner.dataset.syncPrevDisabled === undefined) {
                inner.dataset.syncPrevDisabled = inner.disabled ? '1' : '0';
              }
              inner.disabled = true;
            } else if (inner.dataset.syncPrevDisabled !== undefined) {
              inner.disabled = inner.dataset.syncPrevDisabled === '1';
              delete inner.dataset.syncPrevDisabled;
            }
          }
          return;
        }
        if (on) {
          if (el.dataset.syncPrevDisabled === undefined) {
            el.dataset.syncPrevDisabled = el.disabled ? '1' : '0';
          }
          el.disabled = true;
        } else if (el.dataset.syncPrevDisabled !== undefined) {
          el.disabled = el.dataset.syncPrevDisabled === '1';
          delete el.dataset.syncPrevDisabled;
        }
      });
    }
  },

  _vizIsSyncLockedSlot(slot) {
    return slot === 'b' && !!(this._viz && this._viz.syncDisplay && this._viz.compareMode);
  },

  _vizHasDiffField() {
    const hasA = Array.isArray(this._viz.info_a?.fields) && this._viz.info_a.fields.includes('viz_diff');
    const hasB = Array.isArray(this._viz.info_b?.fields) && this._viz.info_b.fields.includes('viz_diff');
    return !!(this._viz.compareMode && hasA && hasB);
  },

  _vizActiveUsesDiff() {
    return this._viz.info_a?.active_field === 'viz_diff' || this._viz.info_b?.active_field === 'viz_diff';
  },

  _vizRenderSourceSummary() {
    ['a', 'b'].forEach(slot => {
      const chip = document.getElementById(`viz-source-${slot}`);
      if (!chip) return;
      const info = slot === 'a' ? this._viz.info_a : this._viz.info_b;
      const state = this._vizGetSlotState(slot);
      const model = this._vizGetModelByKey(state.modelKey);
      if (!info || (slot === 'b' && !this._viz.compareMode)) {
        chip.style.display = 'none';
        chip.textContent = '';
        chip.title = '';
        return;
      }
      const modelText = model ? model.name : 'No model';
      const sourcePath = info.source || state.matPath || '';
      chip.textContent = `${slot.toUpperCase()} · ${modelText} · ${this._vizBasename(sourcePath)}`;
      chip.title = sourcePath;
      chip.style.display = '';
    });
  },

  _vizBasename(path) {
    const text = String(path || '').trim();
    if (!text) return '-';
    const parts = text.split(/[\\/]/);
    return parts[parts.length - 1] || text;
  },

  _vizIsShellPeeqKey(key, meta = {}) {
    const raw = String(key || '');
    const text = `${raw} ${meta.label || ''} ${meta.description || ''}`.toLowerCase();
    if (!text.includes('peeq')) return false;
    return raw === 'gauss_peeq'
      || raw === 'frame_gauss_peeq'
      || raw.includes('shell_layer_peeq')
      || raw.includes('validation_jax_PEEQ')
      || raw.includes('validation_abaqus_PEEQ')
      || text.includes('through-thickness')
      || text.includes('sneg')
      || text.includes('spos');
  },

  _vizPeeqGroupOptionLabel(key, meta = {}) {
    const text = `${key || ''} ${meta.label || ''}`.toLowerCase();
    if (text.includes('spos') || text.includes('peeq_pos') || /\bpos\b/.test(text)) return 'Pos';
    if (text.includes('sneg') || text.includes('peeq_neg') || /\bneg\b/.test(text)) return 'Neg';
    return 'Max';
  },

  _vizPeeqGroupSort(label) {
    return { Pos: 0, Neg: 1, Max: 2 }[label] ?? 9;
  },

  _vizShellPeeqGroupMeta(fieldMeta, keys) {
    const maxKey = keys.find(key => this._vizPeeqGroupOptionLabel(key, fieldMeta[key]) === 'Max') || keys[0];
    const meta = fieldMeta[maxKey] || {};
    const baseDescription = meta.description || 'Equivalent plastic strain for shell section points.';
    const optionHint = 'Use the quantity menu to choose Pos, Neg, or Max through the exposed SNEG/SPOS surfaces.';
    return {
      ...meta,
      label: 'PEEQ',
      symbol: meta.symbol || '\\bar{\\varepsilon}^{p}',
      formula: meta.formula || '\\bar{\\varepsilon}^{p}=\\{PEEQ_{SPOS},PEEQ_{SNEG},\\max_z PEEQ\\}',
      description: baseDescription.includes('Use the quantity menu') ? baseDescription : `${baseDescription} ${optionHint}`,
    };
  },

  _vizIsShellSplitMagnitudeKey(key) {
    const raw = String(key || '').toLowerCase();
    return raw.includes('sgen_sf_magnitude')
      || raw.includes('sgen_sm_magnitude')
      || raw.includes('egen_ge_magnitude')
      || raw.includes('egen_gk_magnitude')
      || raw.includes('u_trans_magnitude')
      || raw.includes('u_rot_magnitude')
      || raw.includes('nforc_force_magnitude')
      || raw.includes('nforc_moment_magnitude')
      || raw.includes('ceq_force_magnitude')
      || raw.includes('ceq_moment_magnitude')
      || raw.includes('ferror_force_magnitude')
      || raw.includes('ferror_moment_magnitude');
  },

  _vizIsShellGeneralizedSubfield(key, meta = {}) {
    const text = `${key || ''} ${meta.label || ''} ${meta.description || ''}`.toLowerCase();
    return text.includes('shakedown_sf_')
      || text.includes('shakedown_sm_')
      || text.includes('shakedown_ge_')
      || text.includes('shakedown_gk_')
      || text.includes('rsdms_sf_')
      || text.includes('rsdms_sm_')
      || text.includes('rsdms_ge_')
      || text.includes('rsdms_gk_')
      || text.includes('generalized_residual_sf')
      || text.includes('generalized_residual_sm')
      || text.includes('generalized_total_sf')
      || text.includes('generalized_total_sm')
      || text.includes('generalized_residual_ge')
      || text.includes('generalized_residual_gk')
      || text.includes('generalized_total_ge')
      || text.includes('generalized_total_gk');
  },

  _vizIsShellDisplacementRotationField(key, meta = {}) {
    const nComp = Number(meta.n_components || 1);
    const text = `${key || ''} ${meta.label || ''} ${meta.description || ''}`.toLowerCase();
    return nComp === 6 && (
      text.includes('shell displacement/rotation')
      || text.includes('shell displacement')
      || String(key || '').toLowerCase() === 'shell_u_nodal'
      || String(key || '').toLowerCase() === 'frame_u'
      || String(key || '').toLowerCase() === 'validation_jax_u'
      || String(key || '').toLowerCase() === 'validation_abaqus_u'
    );
  },

  _vizIsShellForceMomentField(key, meta = {}) {
    const nComp = Number(meta.n_components || 1);
    const lower = String(key || '').toLowerCase();
    const text = `${lower} ${meta.label || ''} ${meta.description || ''}`.toLowerCase();
    return nComp === 6 && (
      text.includes('shell generalized nodal force')
      || text.includes('force/moment')
      || lower === 'shell_gen_internal_force'
      || lower === 'frame_nforc'
      || lower === 'frame_shell_f_drill'
      || lower === 'validation_jax_nforc'
      || lower === 'validation_abaqus_nforc'
      || lower === 'rsdms_nforc'
      || lower === 'rsdms_ceq'
      || lower === 'rsdms_ferror'
      || lower === 'shakedown_equality_violation'
    );
  },

  _vizIsShellGeneralizedField(key, meta = {}) {
    const nComp = Number(meta.n_components || 1);
    const text = `${key || ''} ${meta.label || ''} ${meta.description || ''}`.toLowerCase();
    return nComp === 6 && (
      text.includes('shell generalized stress')
      || text.includes('shell generalized strain')
      || text.includes('sgen shell generalized')
      || text.includes('egen shell generalized')
    );
  },

  _vizIsShellSplitVectorField(key, meta = {}) {
    const nComp = Number(meta.n_components || 1);
    const text = `${key || ''} ${meta.label || ''} ${meta.description || ''}`.toLowerCase();
    return nComp === 6 && (
      this._vizIsShellGeneralizedField(key, meta)
      || this._vizIsShellDisplacementRotationField(key, meta)
      || this._vizIsShellForceMomentField(key, meta)
    );
  },

  _vizShellSplitMagnitudeOptions(key, fieldMeta = {}) {
    const raw = String(key || '');
    const lower = raw.toLowerCase();
    const meta = fieldMeta[key] || {};
    const text = `${lower} ${meta.label || ''} ${meta.description || ''}`.toLowerCase();
    let pairs = [];
    if (lower === 'frame_gauss_stress' || lower === 'gauss_stress' || lower === 'shell_generalized_stress' || text.includes('shell generalized stress')) {
      let prefix = lower === 'shell_generalized_stress' ? 'shell' : (lower === 'gauss_stress' ? 'gauss_shell' : 'frame_shell');
      if (lower === 'validation_jax_s') prefix = 'validation_jax';
      if (lower === 'validation_abaqus_s') prefix = 'validation_abaqus';
      pairs = [
        [`${prefix}_sgen_sf_magnitude`, 'SF magnitude'],
        [`${prefix}_sgen_sm_magnitude`, 'SM magnitude'],
      ];
    } else if (lower === 'frame_gauss_strain' || lower === 'gauss_strain' || lower === 'shell_generalized_strain' || text.includes('shell generalized strain')) {
      let prefix = lower === 'shell_generalized_strain' ? 'shell' : (lower === 'gauss_strain' ? 'gauss_shell' : 'frame_shell');
      if (lower === 'validation_jax_e') prefix = 'validation_jax';
      if (lower === 'validation_abaqus_e') prefix = 'validation_abaqus';
      pairs = [
        [`${prefix}_egen_ge_magnitude`, 'GE magnitude'],
        [`${prefix}_egen_gk_magnitude`, 'GK magnitude'],
      ];
    } else if (this._vizIsShellDisplacementRotationField(key, meta)) {
      let prefix = lower === 'shell_u_nodal' ? 'shell' : 'frame_shell';
      if (lower === 'validation_jax_u') prefix = 'validation_jax';
      if (lower === 'validation_abaqus_u') prefix = 'validation_abaqus';
      pairs = [
        [`${prefix}_u_trans_magnitude`, 'U magnitude'],
        [`${prefix}_u_rot_magnitude`, 'UR magnitude'],
      ];
    } else if (this._vizIsShellForceMomentField(key, meta)) {
      let prefix = ({
        shell_gen_internal_force: 'shell_nforc',
        frame_nforc: 'frame_shell_nforc',
        frame_shell_f_drill: 'frame_shell_f_drill',
        validation_jax_nforc: 'validation_jax_nforc',
        validation_abaqus_nforc: 'validation_abaqus_nforc',
        rsdms_nforc: 'rsdms_nforc',
        rsdms_ceq: 'rsdms_ceq',
        rsdms_ferror: 'rsdms_ferror',
        shakedown_equality_violation: 'shakedown_ceq',
      })[lower] || lower;
      pairs = [
        [`${prefix}_force_magnitude`, 'F magnitude'],
        [`${prefix}_moment_magnitude`, 'M magnitude'],
      ];
    }
    return pairs
      .filter(([derivedKey]) => fieldMeta[derivedKey])
      .map(([derivedKey, label]) => ({ key: derivedKey, label, meta: fieldMeta[derivedKey] || {} }));
  },

  _vizClassifyFieldFamily(key, meta = {}) {
    const keyLower = String(key || '').toLowerCase();
    const text = `${keyLower} ${meta.label || ''} ${meta.description || ''}`.toLowerCase();
    if (keyLower === 'shakedown_inequality_multiplier') return 'other';
    if (this._vizIsShellGeneralizedSubfield(key, meta)) {
      if (text.includes('ge_') || text.includes('gk_') || /\bge\b/.test(text) || /\bgk\b/.test(text)) return 'strain';
      return 'stress';
    }
    // NFORC / reaction / internal-force checks must come BEFORE the
    // broad 'stress' substring check — NFORC fields like
    // ``solid_nforc_nodal`` carry a description such as "Nodal force
    // due to stress, compared against ABAQUS NFORC output", which
    // otherwise (mis-)matches the 'stress' rule first and ends up
    // showing Mises / S11 / S22 / S12 instead of NFORC1/2/3.
    if (
      keyLower.includes('nforc')
      || keyLower.includes('reaction')
      || keyLower.includes('internal_force')
      || text.includes('nodal force')
      || text.includes('reaction')
      || text.includes('internal force')
      || text.includes('nforc')
    ) {
      return 'reaction';
    }
    if (text.includes('stress')) return 'stress';
    if (text.includes('strain') || text.includes('peeq')) return 'strain';
    if (/(^u$|^frame_u$|^elastic_u$|^solid_u_nodal$|_u_|displacement)/.test(text)) return 'displacement';
    if (text.includes('force')) return 'reaction';
    return 'other';
  },

  _vizComponentLabels(family, meta = {}, key = '') {
    const nComp = Number(meta.n_components || 1);
    const text = `${key} ${meta.label || ''} ${meta.description || ''}`.toLowerCase();
    if (nComp <= 1) return [];
    if (text.includes('shell generalized stress') && nComp === 6) {
      return ['SF11', 'SF22', 'SF12', 'SM11', 'SM22', 'SM12'];
    }
    if (text.includes('shell generalized strain') && nComp === 6) {
      return ['GE11', 'GE22', 'GE12', 'GK11', 'GK22', 'GK12'];
    }
    if (this._vizIsShellDisplacementRotationField(key, meta)) {
      return ['U1', 'U2', 'U3', 'UR1', 'UR2', 'UR3'];
    }
    if (this._vizIsShellForceMomentField(key, meta)) {
      return ['F1', 'F2', 'F3', 'M1', 'M2', 'M3'];
    }
    if (text.includes('plastic strain')) {
      if (nComp === 4) return ['PE11', 'PE22', 'PE12', 'PE33'];
      if (nComp === 6) return ['PE11', 'PE22', 'PE33', 'PE12', 'PE13', 'PE23'];
      if (nComp === 3) return ['PE11', 'PE22', 'PE12'];
    }
    if (text.includes('equality_violation') || text.includes('self-equilibrium')) {
      return ['1', '2', '3'].slice(0, nComp).concat(Array.from({ length: Math.max(0, nComp - 3) }, (_, i) => `${i + 4}`));
    }
    if (text.includes('shakedown_sf_') || text.includes('rsdms_sf_') || text.includes('generalized_residual_sf') || text.includes('generalized_total_sf')) {
      return ['SF11', 'SF22', 'SF12'].slice(0, nComp);
    }
    if (text.includes('shakedown_sm_') || text.includes('rsdms_sm_') || text.includes('generalized_residual_sm') || text.includes('generalized_total_sm')) {
      return ['SM11', 'SM22', 'SM12'].slice(0, nComp);
    }
    if (text.includes('shakedown_ge_') || text.includes('rsdms_ge_') || text.includes('generalized_residual_ge') || text.includes('generalized_total_ge')) {
      return ['GE11', 'GE22', 'GE12'].slice(0, nComp);
    }
    if (text.includes('shakedown_gk_') || text.includes('rsdms_gk_') || text.includes('generalized_residual_gk') || text.includes('generalized_total_gk')) {
      return ['GK11', 'GK22', 'GK12'].slice(0, nComp);
    }
    if (family === 'stress') {
      if (nComp === 6) return ['S11', 'S22', 'S33', 'S12', 'S13', 'S23'];
      if (nComp === 3) return ['S11', 'S22', 'S12'];
    }
    if (family === 'strain') {
      if (nComp === 6) return ['E11', 'E22', 'E33', 'E12', 'E13', 'E23'];
      if (nComp === 3) return ['E11', 'E22', 'E12'];
    }
    if (family === 'displacement') {
      const labels = text.includes('validation_') ? ['U1', 'U2', 'U3'] : ['1', '2', '3'];
      return labels.slice(0, nComp).concat(Array.from({ length: Math.max(0, nComp - labels.length) }, (_, i) => `${i + labels.length + 1}`));
    }
    if (family === 'reaction') {
      // Use the canonical NFORC1/NFORC2/NFORC3 labels for every nodal-
      // force field — solid (``solid_nforc_nodal``), frame
      // (``frame_nforc``), validation (``validation_jax_NFORC``,
      // ``validation_abaqus_NFORC``) and shell (where the 6-comp
      // generalized force/moment is already handled by the dedicated
      // shell branch above). Falling back to plain ``'1','2','3'``
      // for solid models made it impossible to tell NFORC1 from a
      // stress component visually.
      const labels = ['NFORC1', 'NFORC2', 'NFORC3'];
      return labels.slice(0, nComp).concat(Array.from({ length: Math.max(0, nComp - labels.length) }, (_, i) => `${i + labels.length + 1}`));
    }
    return Array.from({ length: nComp }, (_, i) => `${i + 1}`);
  },

  _vizFamilyLabel(family) {
    return {
      stress: 'Stress',
      strain: 'Strain',
      displacement: 'Displacement',
      reaction: 'Force',
      force: 'Force',
      other: '其他',
    }[family] || '其他';
  },

  _vizVariableCode(key, meta = {}) {
    const lower = String(key || '').toLowerCase();
    const text = `${key} ${meta.label || ''} ${meta.description || ''}`.toLowerCase();
    if (lower === 'shakedown_inequality_multiplier' || text.includes('lagrange multiplier')) return 'Lambda';
    if (text.includes('viz_diff') || /\bdiff\b/.test(text)) return 'Diff';
    if (lower.includes('xs_path') || text.includes('path effective excess')) return 'XS_path';
    if (lower.includes('rsdms_xs') || lower.includes('rsdm_xs') || text.includes('excess stress')) return 'XS';
    if (text.includes('cineq') || text.includes('inequality_violation')) return 'CInEQ';
    if (text.includes('equality_violation') || text.includes('self-equilibrium')) return 'CEQ';
    if (text.includes('shell generalized stress')) return 'SGEN';
    if (text.includes('shell generalized strain')) return 'EGEN';
    const shellGenCode = this._vizShellGeneralizedResultantCode(lower, text);
    if (shellGenCode) return shellGenCode;
    if (text.includes('shakedown_phi') || text.includes('ilyushin_phi') || text.includes('yield function')) return 'Phi';
    if (text.includes('plastic strain') && !text.includes('peeq')) return 'PE';
    if (text.includes('peeq')) return 'PEEQ';
    if (text.includes('residual_stress') || text.includes('residual stress')) return 'RS';
    if (text.includes('force_error') || text.includes('ferror') || text.includes('force balance error')) return 'FERR';
    if (lower === 'frame_shell_f_drill' || text.includes('fdrill') || text.includes('f_{drill}')) return 'FDRILL';
    if (text.includes('reaction')) return 'RF';
    if (text.includes('internal force') || text.includes('internal_force') || text.includes('nforc') || text.includes('force')) return 'NFORC';
    if (text.includes('stress')) return 'S';
    if (text.includes('strain')) return 'E';
    if (/(^u$|^frame_u$|^elastic_u$|^solid_u_nodal$|_u_|displacement)/.test(text)) return 'U';
    return String(key || 'VAR').replace(/^shakedown_/, '').replace(/^frame_/, '').replace(/^gauss_/, '').toUpperCase();
  },

  _vizShellGeneralizedResultantCode(lower, text) {
    const isResidual = lower.includes('_res') || text.includes('residual') || text.includes('rho');
    const isTotal = lower.includes('_tot') || text.includes('total');
    const suffix = isTotal ? 'T' : (isResidual ? 'R' : '');
    if (text.includes('generalized_residual_sf') || text.includes('generalized_total_sf') || text.includes('shakedown_sf_') || text.includes('rsdms_sf_') || /\bsf\b/.test(text)) return suffix ? `SF${suffix}` : 'SF';
    if (text.includes('generalized_residual_sm') || text.includes('generalized_total_sm') || text.includes('shakedown_sm_') || text.includes('rsdms_sm_') || /\bsm\b/.test(text)) return suffix ? `SM${suffix}` : 'SM';
    if (text.includes('generalized_residual_ge') || text.includes('generalized_total_ge') || text.includes('shakedown_ge_') || text.includes('rsdms_ge_') || /\bge\b/.test(text)) return suffix ? `GE${suffix}` : 'GE';
    if (text.includes('generalized_residual_gk') || text.includes('generalized_total_gk') || text.includes('shakedown_gk_') || text.includes('rsdms_gk_') || /\bgk\b/.test(text)) return suffix ? `GK${suffix}` : 'GK';
    return '';
  },

  _vizShortFieldSource(key, meta = {}) {
    const raw = String(key || '');
    const text = `${raw} ${meta.label || ''} ${meta.description || ''}`.toLowerCase();
    if (text.includes('through-thickness max')) return 'through-thickness max';
    if (raw === 'frame_gauss_peeq' || raw === 'gauss_peeq') return 'through-thickness max';
    if (text.includes('sneg')) return 'SNEG surface';
    if (text.includes('spos')) return 'SPOS surface';
    if (text.includes('plastic strain')) return 'plastic strain';
    if (text.includes('generalized stress') || raw === 'frame_gauss_stress') return 'generalized [SF, SM]';
    if (text.includes('generalized strain') || raw === 'frame_gauss_strain') return 'generalized [GE, GK]';
    if (text.includes('force balance error') || raw === 'frame_force_error' || raw.toLowerCase() === 'rsdms_ferror') return 'free DOF balance';
    if (raw === 'frame_shell_f_drill') return 'drilling force';
    if (text.includes('displacement/rotation') || raw === 'frame_u') return 'translation/rotation';
    if (text.includes('generalized nodal force') || raw === 'frame_nforc') return 'force/moment';
    if (this._vizIsShellGeneralizedSubfield(raw, meta)) return 'shell generalized';
    if (this._vizIsShellForceMomentField(raw, meta)) return 'force/moment';
    return raw
      .replace(/^shakedown_/, '')
      .replace(/^frame_/, '')
      .replace(/^gauss_/, '')
      .replace(/_stress$/, '')
      .replace(/_strain$/, '')
      .replace(/_/g, ' ');
  },

  _vizVariableDisplayLabel(code, key, meta = {}, count = 1) {
    if (/^(SF|SM|GE|GK)[RT]$/.test(code)) {
      const state = code.endsWith('T') ? 'total' : 'rho';
      if (count > 1) return `${code} (${state}; ${this._vizShortFieldSource(key, meta)})`;
      return `${code} (${state})`;
    }
    return count > 1 ? `${code} (${this._vizShortFieldSource(key, meta)})` : code;
  },

  _vizVariableMapText(variable) {
    if (!variable) return '';
    const key = String(variable.key || '');
    if (variable.code === 'Lambda') {
      return `${variable.code} (${key}): yield inequality Lagrange multiplier`;
    }
    if (variable.code === 'Diff') {
      return `${variable.code} (${key}): B-A displayed scalar`;
    }
    if (variable.code === 'CEQ') {
      return `${variable.code} (${key}): residual stress self-equilibrium`;
    }
    if (variable.code === 'CInEQ') {
      return `${variable.code} (${key}): yield inequality violation`;
    }
    if (variable.code === 'SGEN') {
      return `${variable.code} (${key}): ${variable.meta?.description || 'shell generalized stress [SF, SM]'}`;
    }
    if (variable.code === 'EGEN') {
      return `${variable.code} (${key}): ${variable.meta?.description || 'shell generalized strain [GE, GK]'}`;
    }
    if (/^(SF|SM|GE|GK)[RT]$/.test(variable.code)) {
      const kind = variable.code.endsWith('T') ? 'total' : 'residual/rho';
      return `${variable.code} (${key}): ${kind} ${variable.meta?.description || 'shell generalized quantity'}`;
    }
    if (variable.meta?.description) {
      return `${variable.code} (${key}): ${variable.meta.description}`;
    }
    return `${variable.code} (${key})`;
  },

  _vizScalarOptionLabel(family, key, meta = {}) {
    const text = `${key} ${meta.label || ''} ${meta.description || ''}`.toLowerCase();
    if (String(key || '').toLowerCase() === 'shakedown_inequality_multiplier') return 'Value';
    if (String(key || '').toLowerCase().includes('_xs_') || text.includes('excess stress')) return 'von Mises';
    if (text.includes('viz_diff') || /\bdiff\b/.test(text)) return 'Value';
    if (text.includes('peeq')) return 'PEEQ';
    if (text.includes('plastic strain')) return 'Magnitude';
    if (text.includes('violation')) return 'Magnitude';
    if (this._vizIsShellGeneralizedSubfield(key, meta)) return 'Magnitude';
    if (text.includes('shell generalized stress') || text.includes('shell generalized strain')) return 'Magnitude';
    if (family === 'stress') return 'Mises';
    if (family === 'strain') return 'Equivalent';
    if (family === 'displacement') return 'Magnitude';
    if (family === 'reaction') return 'Magnitude';
    return 'Value';
  },

  _vizEquivalentLabel(family, key, meta = {}) {
    const text = `${key} ${meta.label || ''} ${meta.description || ''}`.toLowerCase();
    if (text.includes('plastic strain')) return 'Magnitude';
    if (this._vizIsShellGeneralizedSubfield(key, meta)) return 'Magnitude';
    if (text.includes('shell generalized stress') || text.includes('shell generalized strain')) return 'Magnitude';
    if (family === 'stress') return 'Mises';
    if (family === 'strain') return 'Equivalent';
    if (family === 'displacement' || family === 'reaction') return 'Magnitude';
    return this._vizScalarOptionLabel(family, key, meta);
  },

  _vizEncodeFieldChoice(key, component) {
    return `${key}::${component === null || component === undefined ? 'auto' : component}`;
  },

  _vizDecodeFieldChoice(value) {
    const text = String(value || '');
    const idx = text.lastIndexOf('::');
    if (idx < 0) return { key: text, component: null };
    const key = text.slice(0, idx);
    const raw = text.slice(idx + 2);
    return { key, component: raw === 'auto' ? null : Number(raw) };
  },

  _vizNormalizeChoice(key, component) {
    let resolvedKey = String(key || '');
    let resolvedComponent = component ?? null;
    if (resolvedKey.startsWith('derived_vm_')) {
      resolvedKey = resolvedKey.slice('derived_vm_'.length);
      resolvedComponent = null;
    } else if (resolvedKey.startsWith('derived_mag_')) {
      resolvedKey = resolvedKey.slice('derived_mag_'.length);
      resolvedComponent = null;
    }
    return { key: resolvedKey, component: resolvedComponent };
  },

  _vizBuildFieldCatalog(info) {
    const fieldMeta = info?.field_meta || {};
    const orderedKeys = Array.isArray(info?.fields) && info.fields.length ? info.fields : Object.keys(fieldMeta);
    const visibleKeys = orderedKeys.filter(key => !String(key).startsWith('derived_'));
    let keys = visibleKeys.length ? visibleKeys : orderedKeys;
    // Probe-mode field filter (Step 3a): the toolbar dropdown only
    // exposes fields whose location matches the active probe mode, so
    // any subsequent ``set_field`` cannot land on a field the picker
    // would silently refuse. If the filter would empty the catalog
    // (unusual MAT without any node/gauss field), keep the unfiltered
    // list so the UI does not lock up.
    if (this._viz?.probeMode) {
      const kind = this._viz.probeKind || 'node';
      const allowed = kind === 'node'
        ? new Set(['node'])
        : new Set(['gauss', 'element']);
      const filtered = keys.filter(k =>
        allowed.has(String(fieldMeta[k]?.location || '').toLowerCase())
      );
      if (filtered.length) keys = filtered;
    }
    const splitMagnitudeKeys = new Set(keys.filter(key => this._vizIsShellSplitMagnitudeKey(key)));
    const peeqKeys = keys.filter(key => this._vizIsShellPeeqKey(key, fieldMeta[key]));
    const groupPeeq = peeqKeys.length > 1 && peeqKeys.some(key => {
      const raw = String(key || '');
      return raw.includes('shell_layer_peeq')
        || raw.includes('validation_jax_PEEQ')
        || raw.includes('validation_abaqus_PEEQ');
    });
    const peeqKeySet = new Set(groupPeeq ? peeqKeys : []);
    const peeqGroupId = '__shell_peeq__';
    const quantitiesByFamily = {};
    const variables = [];
    const codeCounts = {};
    keys.forEach(key => {
      if (splitMagnitudeKeys.has(key) || peeqKeySet.has(key)) return;
      const meta = fieldMeta[key];
      if (!meta) return;
      const code = this._vizVariableCode(key, meta);
      codeCounts[code] = (codeCounts[code] || 0) + 1;
    });
    if (groupPeeq) codeCounts.PEEQ = 1;
    const pushUnique = (variableKey, option) => {
      if (!quantitiesByFamily[variableKey]) quantitiesByFamily[variableKey] = [];
      if (!quantitiesByFamily[variableKey].some(item => item.value === option.value)) {
        quantitiesByFamily[variableKey].push(option);
      }
    };
    let peeqInserted = false;
    keys.forEach(key => {
      if (splitMagnitudeKeys.has(key)) return;
      if (peeqKeySet.has(key)) {
        if (peeqInserted) return;
        peeqInserted = true;
        const meta = this._vizShellPeeqGroupMeta(fieldMeta, peeqKeys);
        variables.push({ id: peeqGroupId, label: 'PEEQ', code: 'PEEQ', key: peeqKeys[0], meta });
        const seenPeeqLabels = new Set();
        peeqKeys
          .map(peeqKey => {
            const label = this._vizPeeqGroupOptionLabel(peeqKey, fieldMeta[peeqKey]);
            return { key: peeqKey, label, meta: fieldMeta[peeqKey] || {} };
          })
          .sort((a, b) => this._vizPeeqGroupSort(a.label) - this._vizPeeqGroupSort(b.label))
          .forEach(item => {
            if (seenPeeqLabels.has(item.label)) return;
            seenPeeqLabels.add(item.label);
            pushUnique(peeqGroupId, {
              family: peeqGroupId,
              fieldFamily: 'strain',
              key: item.key,
              component: null,
              label: item.label,
              value: this._vizEncodeFieldChoice(item.key, null),
            });
          });
        return;
      }
      const meta = fieldMeta[key];
      if (!meta) return;
      const family = this._vizClassifyFieldFamily(key, meta);
      const code = this._vizVariableCode(key, meta);
      const label = this._vizVariableDisplayLabel(code, key, meta, codeCounts[code] || 1);
      variables.push({ id: key, label, code, key, meta });
      const nComp = Number(meta.n_components || 1);
      if (nComp <= 1) {
        pushUnique(key, {
          family: key,
          fieldFamily: family,
          key,
          component: null,
          label: this._vizScalarOptionLabel(family, key, meta),
          value: this._vizEncodeFieldChoice(key, null),
        });
        return;
      }
      const splitOptions = this._vizShellSplitMagnitudeOptions(key, fieldMeta);
      const shellSplitVector = this._vizIsShellSplitVectorField(key, meta);
      if (splitOptions.length) {
        splitOptions.forEach(item => {
          pushUnique(key, {
            family: key,
            fieldFamily: family,
            key: item.key,
            component: null,
            label: item.label,
            value: this._vizEncodeFieldChoice(item.key, null),
          });
        });
      } else if (!shellSplitVector) {
        pushUnique(key, {
          family: key,
          fieldFamily: family,
          key,
          component: null,
          label: this._vizEquivalentLabel(family, key, meta),
          value: this._vizEncodeFieldChoice(key, null),
        });
      }
      this._vizComponentLabels(family, meta, key).forEach((label, index) => {
        pushUnique(key, {
          family: key,
          fieldFamily: family,
          key,
          component: index,
          label,
          value: this._vizEncodeFieldChoice(key, index),
        });
      });
    });
    return {
      families: variables,
      quantitiesByFamily,
      allOptions: variables.flatMap(variable => quantitiesByFamily[variable.id] || []),
    };
  },

  _vizFindActiveOption(catalog, info) {
    if (!catalog || !info) return null;
    const normalized = this._vizNormalizeChoice(info.active_field, info.active_component);
    const target = this._vizEncodeFieldChoice(normalized.key, normalized.component);
    return catalog.allOptions.find(option => option.value === target) || null;
  },

  _vizPopulateFieldSelectors(slot, { preferState = false } = {}) {
    const info = slot === 'a' ? this._viz.info_a : this._viz.info_b;
    const familySel = document.getElementById(`viz-family-${slot}`);
    const quantitySel = document.getElementById(`viz-quantity-${slot}`);
    const mapEl = document.getElementById(`viz-field-map-${slot}`);
    if (!familySel || !quantitySel) return;
    if (!info) {
      familySel.innerHTML = '<option value="">类别</option>';
      quantitySel.innerHTML = '<option value="">结果量</option>';
      familySel.disabled = true;
      quantitySel.disabled = true;
      if (mapEl) mapEl.textContent = '';
      return;
    }
    const state = this._vizGetSlotState(slot);
    const catalog = this._vizBuildFieldCatalog(info);
    state.catalog = catalog;
    const activeOption = this._vizFindActiveOption(catalog, info);
    let family = (preferState ? state.family : activeOption?.family || state.family) || catalog.families[0]?.id || '';
    if (!catalog.quantitiesByFamily[family]?.length) {
      family = activeOption?.family || catalog.families[0]?.id || '';
    }
    state.family = family;
    familySel.innerHTML = catalog.families.length
      ? catalog.families.map(item => `<option value="${UI.escapeAttr(item.id)}">${UI.escapeHtml(item.label)}</option>`).join('')
      : '<option value="">类别</option>';
    familySel.disabled = catalog.families.length === 0;
    if (family) familySel.value = family;
    const quantities = catalog.quantitiesByFamily[family] || [];
    let quantityValue = preferState ? state.quantityValue : activeOption?.value || state.quantityValue;
    if (!quantities.some(item => item.value === quantityValue)) {
      quantityValue = quantities[0]?.value || '';
    }
    state.quantityValue = quantityValue;
    quantitySel.innerHTML = quantities.length
      ? quantities.map(item => `<option value="${UI.escapeAttr(item.value)}">${UI.escapeHtml(item.label)}</option>`).join('')
      : '<option value="">结果量</option>';
    quantitySel.disabled = quantities.length === 0;
    if (quantityValue) quantitySel.value = quantityValue;
    const variable = catalog.families.find(item => item.id === family);
    // The dedicated "validation_jax_S → …" inline pill was removed to keep
    // the toolbar one row tall; we still hold the same information in the
    // quantity-select's tooltip so power users can hover for the raw key.
    if (quantitySel) {
      const variableKey = variable?.key || '';
      const mapText = this._vizVariableMapText(variable) || '';
      const fieldLabel = variable?.meta?.label || '';
      quantitySel.title = [fieldLabel, mapText, variableKey ? `key: ${variableKey}` : '']
        .filter(Boolean)
        .join('  ·  ');
    }
    if (mapEl) {
      mapEl.textContent = this._vizVariableMapText(variable);
      mapEl.title = variable?.meta?.label || variable?.key || '';
    }
  },

  _vizApplyFieldSelection(slot, value) {
    const choice = this._vizDecodeFieldChoice(value);
    if (!choice.key) return;
    this._vizStopPlayback(slot);
    this._vizSendWs(slot, {
      action: 'set_field',
      key: choice.key,
      component: choice.component,
    });
  },

  _vizSlotInfo(slot) {
    return slot === 'a' ? this._viz.info_a : this._viz.info_b;
  },

  _vizSetSlotInfo(slot, info) {
    if (slot === 'a') this._viz.info_a = info;
    else this._viz.info_b = info;
  },

  _vizLatexInline(value) {
    const text = String(value || '').trim();
    return text ? `\\(${UI.escapeHtml(text)}\\)` : '';
  },

  _vizUpdateMathOverlay(slot) {
    const overlay = document.getElementById(`viz-math-overlay-${slot}`);
    if (!overlay) return;
    const info = this._vizSlotInfo(slot);
    if (!info || !info.show_overlay) {
      overlay.style.display = 'none';
      overlay.innerHTML = '';
      return;
    }
    const key = String(info.active_field || '');
    const meta = info.field_meta?.[key] || {};
    const generated = info.generated_field_info?.[key] || {};
    const symbol = meta.symbol || generated.symbol || '';
    const formula = meta.formula || generated.display_formula || generated.formula || '';
    const description = meta.description || generated.display_text || '';
    const title = info.active_scalar_label || info.active_field_label || meta.label || key || '-';
    const rows = [
      `<div class="viz-math-overlay-title">${UI.escapeHtml(title)}</div>`,
    ];
    if (symbol) {
      rows.push(`<div class="viz-math-overlay-row"><span>符号</span><strong>${this._vizLatexInline(symbol)}</strong></div>`);
    }
    if (formula) {
      rows.push(`<div class="viz-math-overlay-row"><span>公式</span><strong>${this._vizLatexInline(formula)}</strong></div>`);
    }
    if (description) {
      rows.push(`<div class="viz-math-overlay-desc">${UI.escapeHtml(description)}</div>`);
    }
    const frameText = this._vizFrameFullLabel(info);
    if (frameText) {
      rows.push(`<div class="viz-math-overlay-frame">${UI.escapeHtml(frameText)}</div>`);
    }
    overlay.innerHTML = rows.join('');
    overlay.style.display = 'block';
    this._typesetMath(overlay);
    this._vizAttachMathCardDrag(slot);
  },

  // Make the floating math info card draggable inside its canvas panel.
  // - Bound once per slot (idempotent guard via ``dataset.dragAttached``).
  // - On mousedown anywhere on the card, switch to absolute left/top
  //   positioning and follow the cursor until mouseup.
  // - Clamps the final position so the card stays at least partially
  //   visible inside the panel rectangle.
  _vizAttachMathCardDrag(slot) {
    const card = document.getElementById(`viz-math-overlay-${slot}`);
    if (!card || card.dataset.dragAttached === '1') return;
    card.dataset.dragAttached = '1';
    const panel = document.getElementById(`viz-panel-${slot}`);

    // Pixel width of the "grab-to-resize" band around the card edge.
    const EDGE = 8;
    // Bounds on the uniform CSS ``--s`` scale factor. The card's
    // dimensions, padding and every inner font-size are all multiples
    // of ``--s`` (see ``.viz-math-overlay`` rules), so scaling stays
    // aspect-locked and only the font (and proportionally the box)
    // grows/shrinks. 0.6 ≈ "tight but readable", 3.5 ≈ "fills most of
    // a 1080-tall canvas".
    const MIN_SCALE = 0.6;
    const MAX_SCALE = 3.5;

    const state = {
      mode: null, dx: 0, dy: 0, x0: 0, y0: 0,
      w0: 0, h0: 0, l0: 0, t0: 0, s0: 1,
    };

    // Compute which resize "zone" the cursor is in based on its
    // offset within the card. Returns one of:
    //   'n','s','e','w','ne','nw','se','sw','move'
    const zoneFor = (ev) => {
      const cr = card.getBoundingClientRect();
      const x = ev.clientX - cr.left;
      const y = ev.clientY - cr.top;
      const onLeft = x <= EDGE;
      const onRight = x >= cr.width - EDGE;
      const onTop = y <= EDGE;
      const onBottom = y >= cr.height - EDGE;
      if (onTop && onLeft) return 'nw';
      if (onTop && onRight) return 'ne';
      if (onBottom && onLeft) return 'sw';
      if (onBottom && onRight) return 'se';
      if (onTop) return 'n';
      if (onBottom) return 's';
      if (onLeft) return 'w';
      if (onRight) return 'e';
      return 'move';
    };

    const cursorFor = (zone) => {
      switch (zone) {
        case 'nw': case 'se': return 'nwse-resize';
        case 'ne': case 'sw': return 'nesw-resize';
        case 'n': case 's': return 'ns-resize';
        case 'e': case 'w': return 'ew-resize';
        default: return 'grab';
      }
    };

    const onCardHover = (ev) => {
      if (state.mode) return;  // active drag/resize sets its own cursor
      card.style.cursor = cursorFor(zoneFor(ev));
    };

    const currentScale = () => {
      const v = parseFloat(card.style.getPropertyValue('--s'));
      return Number.isFinite(v) && v > 0 ? v : 1;
    };

    const onMouseDown = (ev) => {
      if (ev.button !== 0) return;
      ev.preventDefault();
      ev.stopPropagation();
      const cr = card.getBoundingClientRect();
      const pr = panel.getBoundingClientRect();
      const zone = zoneFor(ev);
      state.mode = zone;
      state.x0 = ev.clientX;
      state.y0 = ev.clientY;
      state.w0 = cr.width;
      state.h0 = cr.height;
      state.l0 = cr.left - pr.left;
      state.t0 = cr.top - pr.top;
      state.s0 = currentScale();
      state.dx = ev.clientX - cr.left;
      state.dy = ev.clientY - cr.top;
      card.classList.add('viz-math-dragging');
      // Pin via left/top so the default ``right: 14px`` CSS anchor
      // stops fighting our explicit coordinates.
      card.style.right = 'auto';
      card.style.bottom = 'auto';
      card.style.cursor = cursorFor(zone);
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    };

    const onMouseMove = (ev) => {
      if (!state.mode) return;
      ev.preventDefault();
      const pr = panel.getBoundingClientRect();
      const dx = ev.clientX - state.x0;
      const dy = ev.clientY - state.y0;

      if (state.mode === 'move') {
        let left = ev.clientX - pr.left - state.dx;
        let top = ev.clientY - pr.top - state.dy;
        const margin = 24;
        left = Math.max(margin - state.w0, Math.min(left, pr.width - margin));
        top = Math.max(0, Math.min(top, pr.height - margin));
        card.style.left = `${left}px`;
        card.style.top = `${top}px`;
        return;
      }

      // Aspect-locked uniform scaling. Project the drag onto the
      // dominant axis for the active handle and convert that pixel
      // delta into a ``--s`` multiplier.
      //
      //   * For pure E/W handles the user obviously controls width;
      //     ``widthDelta = ±dx``.
      //   * For pure N/S handles the user controls height; we map that
      //     onto width via the original aspect ratio so a single drag
      //     produces one consistent scale change.
      //   * For corners, take the larger of {width-equiv from dx,
      //     width-equiv from dy} so the user can lead with either axis
      //     and still feel "I'm growing the card uniformly".
      const aspect = state.w0 / Math.max(1, state.h0);
      let widthDelta = -Infinity;
      const mode = state.mode;
      if (mode === 'e' || mode === 'se' || mode === 'ne') widthDelta = Math.max(widthDelta, dx);
      if (mode === 'w' || mode === 'sw' || mode === 'nw') widthDelta = Math.max(widthDelta, -dx);
      if (mode === 's' || mode === 'se' || mode === 'sw') widthDelta = Math.max(widthDelta, dy * aspect);
      if (mode === 'n' || mode === 'ne' || mode === 'nw') widthDelta = Math.max(widthDelta, -dy * aspect);
      if (!Number.isFinite(widthDelta)) return;

      const rawFactor = (state.w0 + widthDelta) / Math.max(1, state.w0);
      const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, state.s0 * rawFactor));
      const factor = newScale / state.s0;
      const newW = state.w0 * factor;
      const newH = state.h0 * factor;

      // 1) Bump the CSS scale variable so every inner length (padding,
      //    gap, font-sizes, grid column width) jumps in lock-step.
      card.style.setProperty('--s', String(newScale));
      // 2) Also pin the OUTER box to the analytically-predicted size.
      //    Natural content sizing alone doesn't reliably grow the box
      //    proportionally — the math row's formula text can sit
      //    inside a KaTeX-style fixed-pixel span that doesn't track
      //    --s, so the box would end up shorter and wider than the
      //    target ratio. Pinning width/height keeps the aspect locked
      //    regardless of inner-content rendering quirks; ``overflow:
      //    hidden`` in CSS clips anything that doesn't fit.
      card.style.width = `${newW}px`;
      card.style.height = `${newH}px`;

      // Anchor the opposite edge so a top/left drag visually grows
      // away from the anchored corner.
      let newL = state.l0;
      let newT = state.t0;
      if (mode.includes('w')) newL = state.l0 + (state.w0 - newW);
      if (mode.includes('n')) newT = state.t0 + (state.h0 - newH);
      card.style.left = `${newL}px`;
      card.style.top = `${newT}px`;
    };

    const onMouseUp = () => {
      if (!state.mode) return;
      state.mode = null;
      card.classList.remove('viz-math-dragging');
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      card.style.cursor = 'grab';
    };

    card.addEventListener('mousemove', onCardHover);
    card.addEventListener('mousedown', onMouseDown);
    card.addEventListener('contextmenu', (ev) => ev.preventDefault());
  },

  _vizValidationSideLabel(info) {
    const key = String(info?.active_field || '').toLowerCase();
    if (key.startsWith('validation_jax_')) return 'JAX';
    if (key.startsWith('validation_abaqus_')) return 'ABAQUS';
    const label = String(info?.active_field_label || '').trim();
    if (/^jax\b/i.test(label)) return 'JAX';
    if (/^abaqus\b/i.test(label)) return 'ABAQUS';
    return '';
  },

  _vizUpdatePanelLabels() {
    const labelA = document.querySelector('#viz-panel-a .viz-panel-label');
    const labelB = document.querySelector('#viz-panel-b .viz-panel-label');
    const sideA = this._vizValidationSideLabel(this._viz.info_a);
    const sideB = this._vizValidationSideLabel(this._viz.info_b);
    const validationCompare = !!this._viz.compareLocked || !!(sideA || sideB);
    if (labelA) {
      labelA.textContent = validationCompare ? (sideA || 'JAX') : 'A';
      labelA.title = validationCompare ? `Validation ${labelA.textContent} view` : 'View A';
    }
    if (labelB) {
      labelB.textContent = validationCompare ? (sideB || 'ABAQUS') : 'B';
      labelB.title = validationCompare ? `Validation ${labelB.textContent} view` : 'View B';
    }
  },

  _vizClampFrame(info, frame) {
    const nFrames = Math.max(1, Number(info?.n_frames || 0));
    const value = Number.isFinite(Number(frame)) ? Number(frame) : 0;
    return Math.max(0, Math.min(value, nFrames - 1));
  },

  _vizSetFrame(slot, frame, options = {}) {
    const info = this._vizSlotInfo(slot);
    if (!info) return;
    const target = this._vizClampFrame(info, frame);
    info.active_frame = target;
    this._vizSetSlotInfo(slot, info);
    const slider = document.getElementById(`viz-frame-slider-${slot}`);
    if (slider) slider.value = target;
    const labelEl = document.getElementById(`viz-frame-label-${slot}`);
    if (labelEl) labelEl.textContent = this._vizFrameLabel(info, target);
    this._vizUpdateFrameDetailButton(slot, info);
    this._vizUpdateMathOverlay(slot);
    if (options.fromPlayback) this._viz.playInFlight[slot] = true;
    const sent = this._vizSendWs(slot, {action: 'set_frame', frame: target});
    if (options.fromPlayback && !sent) this._viz.playInFlight[slot] = false;
  },

  _vizStepFrame(slot, delta, options = {}) {
    const info = this._vizSlotInfo(slot);
    const nFrames = Math.max(1, Number(info?.n_frames || 0));
    if (!info || nFrames <= 1) return;
    const current = this._vizClampFrame(info, info.active_frame || 0);
    const next = (current + Number(delta || 0) + nFrames) % nFrames;
    this._vizSetFrame(slot, next, options);
  },

  _vizFrameIcon(name) {
    const safe = ['play', 'pause', 'prev', 'next'].includes(name) ? name : 'play';
    return `<span class="viz-icon viz-icon-${safe}" aria-hidden="true"></span>`;
  },

  _vizStopPlayback(slot) {
    const timer = this._viz.playTimers?.[slot];
    if (timer) {
      window.clearInterval(timer);
      delete this._viz.playTimers[slot];
    }
    this._viz.playInFlight[slot] = false;
    const btn = document.getElementById(`viz-frame-play-${slot}`);
    if (btn) {
      btn.classList.remove('active');
      btn.innerHTML = this._vizFrameIcon('play');
      btn.setAttribute('aria-label', '连续播放');
      btn.title = '连续播放';
    }
  },

  _vizTogglePlayback(slot) {
    if (this._viz.playTimers?.[slot]) {
      this._vizStopPlayback(slot);
      return;
    }
    const info = this._vizSlotInfo(slot);
    const nFrames = Math.max(1, Number(info?.n_frames || 0));
    if (!info || nFrames <= 1) return;
    const btn = document.getElementById(`viz-frame-play-${slot}`);
    if (btn) {
      btn.classList.add('active');
      btn.innerHTML = this._vizFrameIcon('pause');
      btn.setAttribute('aria-label', '暂停播放');
      btn.title = '暂停播放';
    }
    this._viz.playTimers[slot] = window.setInterval(() => {
      const latest = this._vizSlotInfo(slot);
      if (!latest || Math.max(1, Number(latest.n_frames || 0)) <= 1) {
        this._vizStopPlayback(slot);
        return;
      }
      if (this._viz.playInFlight?.[slot]) return;
      this._vizStepFrame(slot, 1, { fromPlayback: true });
    }, 450);
  },

  _vizShowFrameDetail(slot) {
    const info = this._vizSlotInfo(slot);
    const full = this._vizFrameFullLabel(info);
    if (!full) return;
    this._vizSetStatus(full);
    const overlayBtn = document.getElementById(`viz-overlay-${slot}`);
    if (overlayBtn && !overlayBtn._on) {
      overlayBtn._on = true;
      overlayBtn.classList.add('active');
      info.show_overlay = true;
      this._vizSetSlotInfo(slot, info);
      this._vizUpdateMathOverlay(slot);
      this._vizSendWs(slot, {action: 'show_overlay', show: true});
    }
  },

  _vizUpdateControls(slot) {
    const info = slot === 'a' ? this._viz.info_a : this._viz.info_b;
    if (!info) return;
    this._vizPopulateFieldSelectors(slot);
    this._vizUpdateFrameAxisLabel(slot, info);
    const slider = document.getElementById(`viz-frame-slider-${slot}`);
    const label = document.getElementById(`viz-frame-label-${slot}`);
    if (slider && info.n_frames > 1) {
      slider.max = info.n_frames - 1;
      slider.value = info.active_frame || 0;
      slider.disabled = false;
      if (label) label.textContent = this._vizFrameLabel(info);
    } else if (slider) {
      slider.max = 0;
      slider.value = 0;
      slider.disabled = true;
      if (label) label.textContent = this._vizFrameLabel(info);
    }
    // Surface the full frame description (LF / step / time / etc.) via the
    // tooltip — we dropped the dedicated "详见信息" button to compact the
    // toolbar, but the data is still available on hover.
    if (label) {
      const full = this._vizFrameFullLabel(info);
      label.title = full || '';
    }
    this._vizUpdateFrameButtons(slot, info);
    this._vizUpdateFrameDetailButton(slot, info);
    const minInput = document.getElementById(`viz-clim-min-${slot}`);
    const maxInput = document.getElementById(`viz-clim-max-${slot}`);
    if (Array.isArray(info.clim) && Number.isFinite(info.clim[0]) && Number.isFinite(info.clim[1])) {
      if (minInput) minInput.value = info.clim[0];
      if (maxInput) maxInput.value = info.clim[1];
    } else if (Array.isArray(info.auto_clim) && Number.isFinite(info.auto_clim[0]) && Number.isFinite(info.auto_clim[1])) {
      if (minInput) minInput.value = info.auto_clim[0];
      if (maxInput) maxInput.value = info.auto_clim[1];
    } else {
      if (minInput) minInput.value = '';
      if (maxInput) maxInput.value = '';
    }
    const autoBtn = document.getElementById(`viz-clim-auto-${slot}`);
    if (autoBtn) autoBtn.classList.toggle('active', !info.clim);
    const fontSelect = document.getElementById(`viz-legend-font-${slot}`);
    if (fontSelect && info.legend_font_pt != null) {
      const desired = String(info.legend_font_pt);
      // Only update if it really differs to avoid clobbering any in-flight
      // user interaction with the dropdown.
      if (fontSelect.value !== desired) {
        const hasOption = Array.from(fontSelect.options).some(o => o.value === desired);
        if (hasOption) fontSelect.value = desired;
      }
    }
    const edgeBtn = document.getElementById(`viz-edges-${slot}`);
    if (edgeBtn) {
      edgeBtn._on = !!info.show_edges;
      edgeBtn.classList.toggle('active', !!info.show_edges);
    }
    const overlayBtn = document.getElementById(`viz-overlay-${slot}`);
    if (overlayBtn) {
      overlayBtn._on = !!info.show_overlay;
      overlayBtn.classList.toggle('active', !!info.show_overlay);
    }
    this._vizUpdateMathOverlay(slot);
    this._vizUpdatePanelLabels();
    this._vizSyncClipControls(slot, info);
    const emptyEl = document.getElementById(`viz-empty-${slot}`);
    if (emptyEl) emptyEl.style.display = 'none';
    this._vizRenderSourceSummary();
    this._vizRefreshCompareButtons();
  },

  _vizActiveClipAxis(slot) {
    return document.querySelector(`.viz-clip-axis.active[data-slot="${slot}"]`)?.dataset.axis || 'z';
  },

  _vizCurrentClipPosition(slot) {
    const slider = document.getElementById(`viz-clip-slider-${slot}`);
    if (!slider) return 0.5;
    return parseInt(slider.value, 10) / 1000;
  },

  _vizCurrentClipInvert(slot) {
    return !!document.getElementById(`viz-clip-flip-${slot}`)?.classList.contains('active');
  },

  _vizClipPayload(slot, overrides = {}) {
    // The 剖切 toggle replaced the legacy ``viz-clip-enable-*`` checkbox.
    // Read the current on/off state from the button's ``.active`` class —
    // if we forget this and fall back to the (now removed) checkbox, every
    // slider drag / 翻转 click without an explicit ``enabled`` override
    // ends up sending ``enabled: false`` and silently turns the cut off.
    const toggleEl = document.getElementById(`viz-clip-toggle-${slot}`);
    return {
      action: 'set_clip',
      enabled: overrides.enabled ?? !!toggleEl?.classList.contains('active'),
      axis: overrides.axis ?? this._vizActiveClipAxis(slot),
      position: overrides.position ?? this._vizCurrentClipPosition(slot),
      invert: overrides.invert ?? this._vizCurrentClipInvert(slot),
    };
  },

  _vizSendClip(slot, overrides = {}) {
    this._vizSendWs(slot, this._vizClipPayload(slot, overrides));
  },

  // Coalesce rapid slider input: keep only the latest desired clip state
  // per slot and never have more than one set_clip request in flight on
  // the server. Once the server returns its info JSON we flush any newer
  // pending state. This mirrors how ABAQUS view-cut keeps the UI fluid.
  _vizClipBusy: {a: false, b: false},
  _vizClipLatest: {a: null, b: null},

  _vizQueueClip(slot, overrides) {
    if (!this._vizClipLatest) this._vizClipLatest = {a: null, b: null};
    if (!this._vizClipBusy) this._vizClipBusy = {a: false, b: false};
    const prev = this._vizClipLatest[slot] || {};
    this._vizClipLatest[slot] = {...prev, ...overrides};
    if (!this._vizClipBusy[slot]) this._vizFlushClip(slot);
  },

  _vizFlushClip(slot) {
    if (!this._vizClipLatest) return;
    const latest = this._vizClipLatest[slot];
    if (!latest) return;
    this._vizClipLatest[slot] = null;
    if (!this._vizClipBusy) this._vizClipBusy = {a: false, b: false};
    this._vizClipBusy[slot] = true;
    this._vizSendClip(slot, latest);
  },

  _vizOnClipReply(slot) {
    if (!this._vizClipBusy || !this._vizClipBusy[slot]) return;
    this._vizClipBusy[slot] = false;
    if (this._vizClipLatest && this._vizClipLatest[slot]) {
      this._vizFlushClip(slot);
    }
  },

  _vizUpdateClipLabel(slot, position, clipInfo) {
    const label = document.getElementById(`viz-clip-label-${slot}`);
    if (!label) return;
    const axis = (clipInfo?.axis || this._vizActiveClipAxis(slot) || 'z').toUpperCase();
    // Show position as a relative percentage of the model's bounding box
    // along the chosen axis (0% = min bound, 100% = max bound). Absolute
    // world coordinates are kept in the tooltip for users who need them.
    const pct = Math.round(Math.max(0, Math.min(1, position)) * 100);
    label.textContent = `${axis} ${pct}%`;
    const bounds = clipInfo?.bounds?.[axis.toLowerCase()];
    if (Array.isArray(bounds) && bounds.length === 2) {
      const coord = bounds[0] + position * (bounds[1] - bounds[0]);
      label.title = `${axis} = ${coord.toPrecision(4)}  (${pct}% of ${bounds[0].toPrecision(3)} → ${bounds[1].toPrecision(3)})`;
    } else if (typeof clipInfo?.coordinate === 'number' && Number.isFinite(clipInfo.coordinate)) {
      label.title = `${axis} = ${clipInfo.coordinate.toPrecision(4)}`;
    } else {
      label.title = `${axis} ${pct}%`;
    }
  },

  // Update the on/off visual state of the 剖切 toggle + every dependent
  // control (axis buttons, slider, 翻转). Used both when the user clicks
  // the toggle and when server state arrives via WebSocket.
  _vizSetClipEnabledUI(slot, enabled) {
    const toggle = document.getElementById(`viz-clip-toggle-${slot}`);
    if (toggle) toggle.classList.toggle('active', !!enabled);
    const slider = document.getElementById(`viz-clip-slider-${slot}`);
    if (slider) slider.disabled = !enabled;
    document.querySelectorAll(`.viz-clip-axis[data-slot="${slot}"]`).forEach(btn => {
      btn.disabled = !enabled;
    });
    const flip = document.getElementById(`viz-clip-flip-${slot}`);
    if (flip) flip.disabled = !enabled;
  },

  _vizSyncClipControls(slot, info) {
    const clip = info?.clip || {};
    const enabled = !!clip.enabled;
    this._vizSetClipEnabledUI(slot, enabled);
    const slider = document.getElementById(`viz-clip-slider-${slot}`);
    if (slider) {
      // Don't fight the user's active drag; only reflect server state when idle
      if (document.activeElement !== slider) {
        const pos = Number.isFinite(clip.position) ? clip.position : 0.5;
        slider.value = String(Math.round(pos * 1000));
      }
    }
    const flip = document.getElementById(`viz-clip-flip-${slot}`);
    if (flip) flip.classList.toggle('active', !!clip.invert);
    const axis = String(clip.axis || 'z').toLowerCase();
    document.querySelectorAll(`.viz-clip-axis[data-slot="${slot}"]`).forEach(btn => {
      btn.classList.toggle('active', String(btn.dataset.axis || '').toLowerCase() === axis);
    });
    this._vizUpdateClipLabel(slot, Number.isFinite(clip.position) ? clip.position : 0.5, clip);
  },

  _vizConnectWs(slot) {
    const key = `ws_${slot}`;
    if (this._viz[key]) {
      try { this._viz[key].close(); } catch (_) {}
      this._viz[key] = null;
    }
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/api/viz/ws/${slot}`;
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    ws.onmessage = (evt) => {
      if (evt.data instanceof ArrayBuffer) {
        const blob = new Blob([evt.data], {type: 'image/jpeg'});
        const imgUrl = URL.createObjectURL(blob);
        const canvas = document.getElementById(`viz-canvas-${slot}`);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = () => {
          const panel = canvas.parentElement;
          const targetWidth = Math.max(1, Math.floor(panel?.clientWidth || img.width));
          const targetHeight = Math.max(1, Math.floor(panel?.clientHeight || img.height));
          if (canvas.width !== targetWidth) canvas.width = targetWidth;
          if (canvas.height !== targetHeight) canvas.height = targetHeight;
          const scale = Math.min(targetWidth / img.width, targetHeight / img.height);
          const drawWidth = Math.round(img.width * scale);
          const drawHeight = Math.round(img.height * scale);
          const dx = Math.floor((targetWidth - drawWidth) / 2);
          const dy = Math.floor((targetHeight - drawHeight) / 2);
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(0, 0, targetWidth, targetHeight);
          ctx.drawImage(img, dx, dy, drawWidth, drawHeight);
          URL.revokeObjectURL(imgUrl);
          this._vizProbeRefreshMarkers(slot);
          this._vizElementSelectionRefresh(slot);
        };
        img.src = imgUrl;
      } else {
        try {
          const parsed = JSON.parse(evt.data);
          // Probe replies are framed as ``{type:'probe_result',...}``
          // and must NOT replace ``info_a/info_b``; route them to the
          // probe handler and stop.
          if (parsed && parsed.type === 'probe_result') {
            this._vizProbeApplyResult(slot, parsed);
            return;
          }
          if (parsed && parsed.type === 'probe_extrema_result') {
            this._vizProbeApplyExtremaResult(slot, parsed);
            return;
          }
          if (parsed && parsed.type === 'probe_marker_projection') {
            this._vizProbeApplyMarkerProjection(slot, parsed);
            return;
          }
          if (parsed && parsed.type === 'element_selection') {
            this._vizElementSelectionApplyState(slot, parsed.state || {});
            return;
          }
          if (slot === 'a') this._viz.info_a = parsed;
          else this._viz.info_b = parsed;
          this._viz.playInFlight[slot] = false;
          this._vizOnClipReply(slot);
          this._vizUpdateControls(slot);
          if (slot === 'b') this._vizApplySyncDisplayLock();
        } catch (_) {}
      }
    };
    ws.onerror = () => {
      this._vizSetStatus('WebSocket 连接出错');
    };
    ws.onopen = () => {
      ws.send(JSON.stringify({action: 'render'}));
    };
    ws.onclose = () => {
      if (this._viz[key] === ws) this._viz[key] = null;
    };
    this._viz[key] = ws;
  },

  _vizSendWs(slot, msg) {
    const ws = this._viz[`ws_${slot}`];
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
      return true;
    } else if (ws && ws.readyState === WebSocket.CONNECTING) {
      ws.addEventListener('open', () => ws.send(JSON.stringify(msg)), {once: true});
      return true;
    }
    return false;
  },

  // ─────────────────────────────────────────────────────────────────
  //   Probe (Abaqus-style picking) — Step 3 of the probe plan
  // ─────────────────────────────────────────────────────────────────

  _vizProbeToggle() {
    const on = !this._viz.probeMode;
    this._viz.probeMode = on;
    const btn = document.getElementById('viz-probe-toggle');
    const allFrames = document.getElementById('viz-probe-all-frames');
    const extremaBtn = document.getElementById('viz-probe-extrema');
    const extremaCount = document.getElementById('viz-probe-extrema-count');
    const extremaVisible = document.getElementById('viz-probe-visible-only');
    const wrap = document.querySelector('.viz-canvas-wrap');
    if (btn) btn.classList.toggle('active', on);
    document.querySelectorAll('.viz-probe-mode-btn').forEach(b => { b.disabled = !on; });
    if (allFrames) allFrames.disabled = !on;
    if (extremaBtn) extremaBtn.disabled = !on;
    if (extremaCount) extremaCount.disabled = !on;
    if (extremaVisible) extremaVisible.disabled = !on;
    if (wrap) wrap.dataset.probing = on ? '1' : '0';
    if (on) {
      if (this._viz.elementSelectMode) this._vizElementSelectionToggle();
      const activeInfo = this._vizSlotInfo('a');
      const activeMeta = activeInfo?.field_meta?.[activeInfo?.active_field || ''];
      const loc = String(activeMeta?.location || '').toLowerCase();
      if (loc === 'gauss' || loc === 'element') this._viz.probeKind = 'element';
      else if (loc === 'node') this._viz.probeKind = 'node';
      document.querySelectorAll('.viz-probe-mode-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.probeMode === this._viz.probeKind);
      });
      // Remember per-slot active field so we can restore on toggle off
      // (probe mode temporarily filters the field dropdown).
      ['a', 'b'].forEach(slot => {
        const info = this._vizSlotInfo(slot);
        if (info) {
          this._viz.probeSavedFieldKey[slot] = info.active_field || '';
          this._viz.probeSavedFieldComponent[slot] = info.active_component;
        }
      });
      ['a', 'b'].forEach(slot => {
        this._vizUpdateControls(slot);
        this._vizProbeApplyFieldConstraint(slot);
      });
      this._vizSetStatus('探针已开启：移动鼠标预览命中点/单元；左键固定；按住 Shift + 左键拖拽旋转模型。', 'alert');
    } else {
      this._vizProbeHideHover();
      ['a', 'b'].forEach(slot => {
        const saved = this._viz.probeSavedFieldKey[slot];
        const info = this._vizSlotInfo(slot);
        if (saved && info && info.active_field !== saved) {
          this._vizSendWs(slot, {
            action: 'set_field',
            key: saved,
            component: this._viz.probeSavedFieldComponent[slot] ?? null,
          });
        }
        this._vizUpdateControls(slot);
      });
      this._vizSetStatus('探针已关闭');
    }
  },

  _vizProbeSetKind(kind) {
    if (kind !== 'node' && kind !== 'element') return;
    if (this._viz.probeKind === kind && this._viz.probeMode) return;
    this._viz.probeKind = kind;
    document.querySelectorAll('.viz-probe-mode-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.probeMode === kind);
    });
    if (!this._viz.probeMode) return;
    ['a', 'b'].forEach(slot => {
      this._vizUpdateControls(slot);
      this._vizProbeApplyFieldConstraint(slot);
    });
    this._vizProbeHideHover();
    this._vizSetStatus('探针已开启：移动鼠标预览命中点/单元；左键固定；按住 Shift + 左键拖拽旋转模型。', 'alert');
  },

  _vizProbeApplyFieldConstraint(slot) {
    if (!this._viz.probeMode) return;
    const info = this._vizSlotInfo(slot);
    if (!info) return;
    const meta = info.field_meta || {};
    const kind = this._viz.probeKind;
    const allowed = kind === 'node'
      ? new Set(['node'])
      : new Set(['gauss', 'element']);
    const activeKey = String(info.active_field || '');
    const activeLoc = String(meta[activeKey]?.location || '').toLowerCase();
    if (allowed.has(activeLoc)) return;
    const ordered = Array.isArray(info.fields) && info.fields.length
      ? info.fields
      : Object.keys(meta);
    const fallback = ordered.find(k =>
      !String(k).startsWith('derived_')
      && allowed.has(String(meta[k]?.location || '').toLowerCase())
    );
    if (fallback) {
      this._vizSendWs(slot, {
        action: 'set_field',
        key: fallback,
        component: null,
      });
    }
  },

  // Convert screen-space (page) coordinates into VizScene render-window
  // pixels. The server renders at the fixed DEFAULT_WINDOW_SIZE
  // (800×600) and the frontend draws the JPEG fit-inside-rect centred
  // in the panel — we must invert that letterbox transform so the
  // pick request lands on the rendered pixel the user actually sees.
  _vizCanvasToRenderPixel(slot, clientX, clientY) {
    const canvas = document.getElementById(`viz-canvas-${slot}`);
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const renderW = 800;
    const renderH = 600;
    const scale = Math.min(rect.width / renderW, rect.height / renderH);
    const drawW = renderW * scale;
    const drawH = renderH * scale;
    const dx = (rect.width - drawW) / 2;
    const dy = (rect.height - drawH) / 2;
    const sx = (clientX - rect.left - dx) / scale;
    const sy = (clientY - rect.top - dy) / scale;
    if (sx < 0 || sx >= renderW || sy < 0 || sy >= renderH) return null;
    return { x: Math.round(sx), y: Math.round(sy) };
  },

  _vizRenderPixelToPanel(slot, renderX, renderY) {
    const canvas = document.getElementById(`viz-canvas-${slot}`);
    const panel = document.getElementById(`viz-panel-${slot}`);
    if (!canvas || !panel) return null;
    const rect = canvas.getBoundingClientRect();
    const pr = panel.getBoundingClientRect();
    const renderW = 800;
    const renderH = 600;
    const scale = Math.min(rect.width / renderW, rect.height / renderH);
    const drawW = renderW * scale;
    const drawH = renderH * scale;
    const dx = (rect.width - drawW) / 2;
    const dy = (rect.height - drawH) / 2;
    return {
      x: (rect.left - pr.left) + dx + Number(renderX) * scale,
      y: (rect.top - pr.top) + dy + Number(renderY) * scale,
    };
  },

  _vizPointSlot(clientX, clientY) {
    for (const slot of ['a', 'b']) {
      const panel = document.getElementById(`viz-panel-${slot}`);
      if (!panel || getComputedStyle(panel).display === 'none') continue;
      const r = panel.getBoundingClientRect();
      if (clientX >= r.left && clientX <= r.right && clientY >= r.top && clientY <= r.bottom) return slot;
    }
    return null;
  },

  _vizElementSelectionToggle() {
    const on = !this._viz.elementSelectMode;
    this._viz.elementSelectMode = on;
    const btn = document.getElementById('viz-element-select-toggle');
    const panel = document.getElementById('viz-element-select-panel');
    const wrap = document.querySelector('.viz-canvas-wrap');
    if (btn) btn.classList.toggle('active', on);
    if (panel) panel.style.display = on ? 'block' : 'none';
    if (wrap) wrap.dataset.elementSelecting = on ? '1' : '0';
    if (on && this._viz.probeMode) this._vizProbeToggle();
    if (on) this._vizElementSelectionSetTab(this._viz.elementSelectTab || 'box');
    if (!on) {
      this._viz.elementBoxDrag = null;
      this._viz.elementRightClickCandidate = null;
      this._vizProbeHideHover();
      ['a', 'b'].forEach(slot => {
        document.getElementById(`viz-element-box-${slot}`)?.style.setProperty('display', 'none');
        this._vizElementSelectionClearMarkers(slot);
      });
    }
    this._vizSetStatus(on ? '单元选取：左键单击或拖框追加，按住shift+左键转动视角；右键单击撤销上一步，右键拖动仍平移模型；空格确认' : '单元选取已关闭', on ? 'alert' : undefined);
    this._vizElementSelectionUpdateButtons();
    ['a', 'b'].forEach(slot => this._vizElementSelectionRefresh(slot));
  },

  _vizElementSelectionSetTab(tab) {
    const active = tab === 'ids' ? 'ids' : 'box';
    this._viz.elementSelectTab = active;
    document.querySelectorAll('.viz-element-tab').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.elementTab === active);
      btn.setAttribute('aria-selected', btn.dataset.elementTab === active ? 'true' : 'false');
    });
    document.querySelectorAll('[data-element-tab-body]').forEach(body => {
      body.style.display = body.dataset.elementTabBody === active ? '' : 'none';
    });
    if (this._viz.elementSelectMode) {
      const status = active === 'ids'
        ? '单元选取：编号选可输入起止单元号追加选择；鼠标移动预览当前结果量探针；空格确认隐藏选中'
        : '单元选取：左键单击或拖框追加，按住shift+左键转动视角；右键单击撤销上一步，右键拖动仍平移模型；空格确认';
      this._vizSetStatus(status, 'alert');
    }
  },

  _vizElementSelectAt(slot, clientX, clientY, operation = 'add') {
    const px = this._vizCanvasToRenderPixel(slot, clientX, clientY);
    if (!px) return;
    this._vizSendWs(slot, { action: 'element_select', x: px.x, y: px.y, operation });
  },

  _vizElementBoxStart(slot, clientX, clientY, operation = 'add') {
    const startPx = this._vizCanvasToRenderPixel(slot, clientX, clientY);
    if (!startPx) return;
    const box = document.getElementById(`viz-element-box-${slot}`);
    const panel = document.getElementById(`viz-panel-${slot}`);
    if (!box || !panel) return;
    const pr = panel.getBoundingClientRect();
    this._viz.elementBoxDrag = {
      slot,
      startClientX: clientX,
      startClientY: clientY,
      startRender: startPx,
      panelLeft: pr.left,
      panelTop: pr.top,
      operation,
    };
    box.style.display = 'block';
    box.style.left = `${clientX - pr.left}px`;
    box.style.top = `${clientY - pr.top}px`;
    box.style.width = '0px';
    box.style.height = '0px';
  },

  _vizElementBoxUpdate(clientX, clientY) {
    const drag = this._viz.elementBoxDrag;
    if (!drag) return;
    const box = document.getElementById(`viz-element-box-${drag.slot}`);
    if (!box) return;
    const x0 = drag.startClientX - drag.panelLeft;
    const y0 = drag.startClientY - drag.panelTop;
    const x1 = clientX - drag.panelLeft;
    const y1 = clientY - drag.panelTop;
    box.style.left = `${Math.min(x0, x1)}px`;
    box.style.top = `${Math.min(y0, y1)}px`;
    box.style.width = `${Math.abs(x1 - x0)}px`;
    box.style.height = `${Math.abs(y1 - y0)}px`;
  },

  _vizElementBoxFinish(clientX, clientY) {
    const drag = this._viz.elementBoxDrag;
    if (!drag) return;
    this._viz.elementBoxDrag = null;
    const box = document.getElementById(`viz-element-box-${drag.slot}`);
    if (box) box.style.display = 'none';
    const endRender = this._vizCanvasToRenderPixel(drag.slot, clientX, clientY);
    if (!endRender) return;
    const dx = Math.abs(clientX - drag.startClientX);
    const dy = Math.abs(clientY - drag.startClientY);
    if (dx < 4 && dy < 4) {
      this._vizElementSelectAt(drag.slot, clientX, clientY, drag.operation || 'add');
      return;
    }
    this._vizSendWs(drag.slot, {
      action: 'element_box_select',
      x0: drag.startRender.x,
      y0: drag.startRender.y,
      x1: endRender.x,
      y1: endRender.y,
      append: true,
      operation: drag.operation || 'add',
    });
  },

  _vizElementSelectionCommand(action) {
    const slots = this._viz.compareMode && this._viz.info_b ? ['a', 'b'] : ['a'];
    slots.forEach(slot => {
      if (slot === 'b' && !this._viz.info_b) return;
      this._vizSendWs(slot, { action });
    });
  },

  _vizElementSelectionSelectIds() {
    if (!this._viz.elementSelectMode) return;
    const start = document.getElementById('viz-element-id-start')?.value ?? '';
    const end = document.getElementById('viz-element-id-end')?.value ?? '';
    const slots = this._viz.compareMode && this._viz.info_b ? ['a', 'b'] : ['a'];
    slots.forEach(slot => {
      if (slot === 'b' && !this._viz.info_b) return;
      this._vizSendWs(slot, {
        action: 'element_select_ids',
        start_id: start,
        end_id: end,
        operation: 'add',
      });
    });
  },

  _vizElementSelectionConfirm(slot = null) {
    const now = performance.now();
    if (now - Number(this._viz.elementLastConfirmAt || 0) < 160) return;
    this._viz.elementLastConfirmAt = now;
    if (slot && this._viz.compareMode && this._viz.info_b) {
      this._vizSendWs(slot, { action: 'element_hide_selected' });
      return;
    }
    this._vizElementSelectionCommand('element_hide_selected');
  },

  _vizElementSelectionRefresh(slot) {
    if (!this._viz.elementSelectMode) return;
    this._vizSendWs(slot, { action: 'element_selection_state' });
  },

  _vizElementSelectionApplyState(slot, state) {
    const current = this._viz.elementSelection?.[slot] || {};
    this._viz.elementSelection[slot] = {
      ...current,
      selected: Array.isArray(state.selected) ? state.selected : [],
      hidden_count: Number(state.hidden_count || 0),
      can_undo: !!state.can_undo,
      total_count: Number(state.total_count || current.total_count || 0),
      markers: Array.isArray(state.markers) ? state.markers : [],
    };
    this._vizElementSelectionRenderMarkers(slot);
    this._vizElementSelectionUpdateButtons();
  },

  _vizElementSelectionRenderMarkers(slot) {
    this._vizElementSelectionClearMarkers(slot);
    const state = this._viz.elementSelection?.[slot] || {};
    const els = [];
    (state.markers || []).forEach(item => {
      const result = { marker: item.marker };
      const el = this._vizProbeCreateMarker(slot, result, `select-${slot}-${item.cell_id}`, `Ele ${item.cell_id}`);
      if (el) {
        el.classList.add('viz-element-selection-marker');
        els.push(el);
      }
    });
    this._viz.elementSelectionMarkerEls[slot] = els;
  },

  _vizElementSelectionClearMarkers(slot) {
    (this._viz.elementSelectionMarkerEls?.[slot] || []).forEach(el => {
      try { el.parentElement?.removeChild(el); } catch (_) {}
    });
    this._viz.elementSelectionMarkerEls[slot] = [];
  },

  _vizElementSelectionClearForSlot(slot) {
    this._vizElementSelectionClearMarkers(slot);
    this._viz.elementSelection[slot] = { selected: [], hidden_count: 0, can_undo: false, total_count: 0, markers: [] };
    this._vizElementSelectionUpdateButtons();
  },

  _vizElementSelectionUpdateButtons() {
    const active = !!this._viz.elementSelectMode;
    const activeSlots = this._viz.compareMode && this._viz.info_b ? ['a', 'b'] : ['a'];
    const states = activeSlots.map(slot => this._viz.elementSelection?.[slot] || {});
    const selected = states.reduce((sum, st) => sum + (Array.isArray(st.selected) ? st.selected.length : 0), 0);
    const hidden = states.reduce((sum, st) => sum + Number(st.hidden_count || 0), 0);
    const canUndo = states.some(st => !!st.can_undo);
    document.querySelectorAll('[data-element-action="hide"]').forEach(btn => {
      btn.disabled = !active || selected <= 0;
    });
    document.querySelectorAll('[data-element-action="show-only"]').forEach(btn => {
      btn.disabled = !active || selected <= 0;
    });
    document.querySelectorAll('[data-element-action="undo"]').forEach(btn => {
      btn.disabled = !active || !canUndo;
    });
    document.querySelectorAll('[data-element-action="restore"]').forEach(btn => {
      btn.disabled = !active || hidden <= 0;
    });
    const idAdd = document.getElementById('viz-element-id-add');
    if (idAdd) idAdd.disabled = !active;
    const status = document.getElementById('viz-element-select-status');
    if (status) status.textContent = `选中 ${selected} / 隐藏 ${hidden}`;
  },

  _vizProbeHoverThrottled(slot, ev, options = {}) {
    const now = performance.now();
    if (this._viz.probeHoverThrottleAt && (now - this._viz.probeHoverThrottleAt) < 30) return;
    this._viz.probeHoverThrottleAt = now;
    this._vizProbeSendHover(slot, ev, options);
  },

  _vizProbeSendHover(slot, ev, options = {}) {
    const px = this._vizCanvasToRenderPixel(slot, ev.clientX, ev.clientY);
    if (!px) {
      this._vizProbeHideHover();
      return;
    }
    const reqId = ++this._viz.probeNextReqId;
    this._viz.probePending[slot] = reqId;
    this._viz.probeLastCursor = { slot, clientX: ev.clientX, clientY: ev.clientY };
    this._vizSendWs(slot, {
      action: 'probe',
      req_id: reqId,
      mode: options.mode || this._viz.probeKind,
      x: px.x,
      y: px.y,
    });
  },

  _vizProbeModeForSlot(slot) {
    const info = this._vizSlotInfo(slot);
    const meta = info?.field_meta || {};
    const key = String(info?.active_field || '');
    const loc = String(meta[key]?.location || '').toLowerCase();
    if (loc === 'node') return 'node';
    if (loc === 'gauss' || loc === 'element') return 'element';
    return this._viz.probeKind || 'node';
  },

  _vizProbePinAt(slot, clientX, clientY) {
    // Click commits a HARD pin: send a fresh probe with a marker
    // ``probePinPending`` so the reply lands as a pin instead of a
    // hover update. This avoids the race where the most-recent hover
    // result was for a slightly different cursor position.
    const px = this._vizCanvasToRenderPixel(slot, clientX, clientY);
    if (!px) return;
    const reqId = ++this._viz.probeNextReqId;
    this._viz.probePending[slot] = reqId;
    this._viz.probePinPending = { reqId, slot, clientX, clientY };
    this._vizSendWs(slot, {
      action: 'probe',
      req_id: reqId,
      mode: this._viz.probeKind,
      include_frames: true,
      x: px.x,
      y: px.y,
    });
  },

  _vizProbeApplyResult(slot, msg) {
    const hoverOnly = !!this._viz.elementSelectMode && !this._viz.probeMode;
    if (!this._viz.probeMode && !hoverOnly) return;
    const pin = this._viz.probePinPending;
    if (this._viz.probeMode && pin && pin.reqId === msg.req_id) {
      this._viz.probePinPending = null;
      if (msg.result) {
        this._vizProbeCommitPin(pin.slot, msg.result, pin.clientX, pin.clientY);
      }
      return;
    }
    // Hover: drop late replies — only the latest req_id wins.
    if (msg.req_id && msg.req_id !== this._viz.probePending[slot]) return;
    if (!msg.result) {
      this._vizProbeHideHover();
      return;
    }
    this._vizProbeShowHover(slot, msg.result);
  },

  _vizProbeMaxCards() {
    return 20;
  },

  _vizProbeRunExtrema() {
    if (!this._viz.probeMode) {
      this._vizSetStatus('请先开启“探针”，再使用极值探针。', 'alert');
      return;
    }
    const countEl = document.getElementById('viz-probe-extrema-count');
    const count = Math.max(1, Math.min(10, Number(countEl?.value || 1)));
    const visibleOnly = !!document.getElementById('viz-probe-visible-only')?.checked;
    const slots = (this._viz.compareMode && this._viz.info_b) ? ['a', 'b'] : ['a'];
    if (this._viz.elementSelectMode) this._vizElementSelectionToggle();
    this._vizProbeHideHover();
    this._viz.extremaPending = {
      total: slots.length,
      done: 0,
      added: 0,
      requested: count,
    };
    slots.forEach(slot => {
      const excluded = this._vizProbeExistingTargets(slot);
      const reqId = ++this._viz.probeNextReqId;
      this._vizSendWs(slot, {
        action: 'probe_extrema',
        req_id: reqId,
        count,
        include_frames: true,
        exclude_nodes: excluded.nodes,
        exclude_cells: excluded.cells,
        visible_only: visibleOnly,
      });
    });
    this._vizSetStatus(`正在生成极值探针：每个模型 ${count} 个${visibleOnly ? '，仅当前可见单元' : '，全部单元'}...`);
  },

  _vizProbeExistingTargets(slot) {
    const nodes = [];
    const cells = [];
    (this._viz.probeCards || []).forEach(card => {
      if (card.slot !== slot) return;
      const r = card.result || {};
      if (card.mode === 'node' && r.point_id !== undefined && r.point_id !== null) {
        nodes.push(Number(r.point_id));
      } else if (card.mode === 'element' && r.cell_id !== undefined && r.cell_id !== null) {
        cells.push(Number(r.cell_id));
      }
    });
    return {
      nodes: [...new Set(nodes.filter(Number.isFinite))],
      cells: [...new Set(cells.filter(Number.isFinite))],
    };
  },

  _vizProbeApplyExtremaResult(slot, msg) {
    const payload = msg?.payload || {};
    const results = Array.isArray(payload.results) ? payload.results : [];
    const before = this._viz.probeCards.length;
    results.forEach((result, idx) => {
      const p = this._vizProbeResultClientPoint(slot, result, idx);
      this._vizProbeCommitPin(slot, result, p.clientX, p.clientY, { confirmOverflow: false });
    });
    const added = Math.max(0, this._viz.probeCards.length - before);
    const pending = this._viz.extremaPending;
    if (pending) {
      pending.done += 1;
      pending.added += results.length;
      if (pending.done >= pending.total) {
        this._vizSetStatus(`已生成 ${pending.added} 个极值探针`);
        this._viz.extremaPending = null;
      }
    } else {
      this._vizSetStatus(`已生成 ${results.length} 个极值探针`);
    }
    if (!results.length) {
      const reason = payload.reason ? `（${payload.reason}）` : '';
      this._vizSetStatus(`当前场变量没有可用极值${reason}`, 'alert');
    }
  },

  _vizProbeResultClientPoint(slot, result, idx = 0) {
    const panel = document.getElementById(`viz-panel-${slot}`);
    const pr = panel?.getBoundingClientRect?.();
    const marker = result?.marker || {};
    let renderXY = null;
    if (Array.isArray(marker.render_xy)) renderXY = marker.render_xy;
    else if (Array.isArray(marker.label_xy)) renderXY = marker.label_xy;
    if (Array.isArray(renderXY)) {
      const p = this._vizRenderPixelToPanel(slot, renderXY[0], renderXY[1]);
      if (p && pr) return { clientX: pr.left + p.x, clientY: pr.top + p.y };
    }
    if (pr) {
      const offset = 18 + (Number(idx) % 8) * 18;
      return { clientX: pr.left + offset, clientY: pr.top + offset };
    }
    return { clientX: 24 + idx * 18, clientY: 120 + idx * 18 };
  },

  _vizProbeShowHover(slot, result) {
    let card = this._viz.probeHoverEl;
    if (!card) card = document.getElementById('viz-probe-hover-card');
    if (!card) return;
    this._viz.probeHoverEl = card;
    card.innerHTML = this._vizProbeRenderCardBody(slot, result, false);
    const last = this._viz.probeLastCursor;
    if (last && last.slot === slot) {
      // Keep the card away from the right/bottom edges so it stays
      // fully visible without scrolling the viewport.
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const cw = card.offsetWidth || 280;
      const ch = card.offsetHeight || 120;
      let left = last.clientX + 14;
      let top = last.clientY + 14;
      if (left + cw > vw - 8) left = Math.max(8, last.clientX - cw - 14);
      if (top + ch > vh - 8) top = Math.max(8, last.clientY - ch - 14);
      card.style.left = `${left}px`;
      card.style.top = `${top}px`;
    }
    card.style.display = 'block';
    this._viz.probeLastResult = { slot, result };
    this._vizProbeUpdateHoverMarker(slot, result);
  },

  _vizProbeHideHover() {
    const card = this._viz.probeHoverEl || document.getElementById('viz-probe-hover-card');
    if (card) card.style.display = 'none';
    this._vizProbeRemoveHoverMarker();
    this._viz.probeLastResult = null;
  },

  _vizProbeCommitPin(slot, result, clientX, clientY, options = {}) {
    const cards = this._viz.probeCards;
    const maxCards = this._vizProbeMaxCards();
    if (cards.length >= maxCards) {
      if (options.confirmOverflow !== false) {
        const ok = window.confirm(`已达 ${maxCards} 张上限，再添加将覆盖最老的（卡 #1）。继续？`);
        if (!ok) return;
      }
      this._vizProbeRemoveCard(cards[0].id);
    }
    this._vizProbeCreatePinnedCard(slot, result, clientX, clientY);
  },

  _vizProbeCreatePinnedCard(slot, result, clientX, clientY) {
    const host = document.getElementById(`viz-probe-pin-host-${slot}`);
    const panel = document.getElementById(`viz-panel-${slot}`);
    if (!host || !panel) return;
    const pr = panel.getBoundingClientRect();
    const id = this._viz.probeNextCardId++;
    const info = this._vizSlotInfo(slot) || {};
    const frozen = {
      id, slot, mode: result.mode, result,
      matIdentity: info.mat_identity || {},
      field_label: this._vizProbeFieldShortLabel(result, slot),
      field_key: result.field_key || '',
      component_index: result.component_index ?? null,
      frame_idx: result.frame_idx ?? null,
      frame_time: result.frame_time ?? null,
    };
    const el = document.createElement('div');
    el.className = 'viz-probe-card';
    el.dataset.cardId = String(id);
    el.innerHTML = this._vizProbeRenderCardBody(slot, result, true);
    host.appendChild(el);
    const w = el.offsetWidth || 280;
    const h = el.offsetHeight || 140;
    let left = (clientX - pr.left) - 12;
    let top = (clientY - pr.top) + 12;
    left = Math.max(8, Math.min(left, pr.width - w - 8));
    top = Math.max(8, Math.min(top, pr.height - h - 8));
    el.style.left = `${left}px`;
    el.style.top = `${top}px`;
    frozen.el = el;
    this._viz.probeCards.push(frozen);
    frozen.markerEl = this._vizProbeCreateMarker(slot, result, id, this._vizProbeMarkerLabel(id));
    el.querySelector('.viz-probe-card-close')?.addEventListener('click', (ev) => {
      ev.stopPropagation();
      this._vizProbeRemoveCard(id);
    });
    this._vizProbeAttachPinDrag(el, panel);
    this._vizProbeUpdateCounter();
    const exportBtn = document.getElementById('viz-probe-export');
    if (exportBtn) exportBtn.disabled = false;
    const compareBtn = document.getElementById('viz-probe-compare');
    if (compareBtn) compareBtn.disabled = false;
  },

  _vizProbeRemoveCard(id) {
    const idx = this._viz.probeCards.findIndex(c => c.id === id);
    if (idx < 0) return;
    const card = this._viz.probeCards[idx];
    try { card.el?.parentElement?.removeChild(card.el); } catch (_) {}
    try { card.markerEl?.parentElement?.removeChild(card.markerEl); } catch (_) {}
    this._viz.probeCards.splice(idx, 1);
    this._vizProbeUpdateCounter();
    const exportBtn = document.getElementById('viz-probe-export');
    if (exportBtn) exportBtn.disabled = this._viz.probeCards.length === 0;
    const compareBtn = document.getElementById('viz-probe-compare');
    if (compareBtn) compareBtn.disabled = this._viz.probeCards.length === 0;
  },

  _vizProbeUpdateCounter() {
    const el = document.getElementById('viz-probe-counter');
    if (!el) return;
    const n = this._viz.probeCards.length;
    const maxCards = this._vizProbeMaxCards();
    el.textContent = `${n}/${maxCards}`;
    el.classList.toggle('viz-probe-counter-warn', n >= maxCards - 1);
    this._viz.probeCards.forEach((card, idx) => {
      const label = card.el?.querySelector?.('[data-probe-card-index]');
      if (label) label.textContent = `探针${idx + 1}`;
    });
    ['a', 'b'].forEach(slot => this._vizProbeRefreshMarkers(slot));
  },

  _vizProbeCreateMarker(slot, result, cardId, labelText = '') {
    const host = document.getElementById(`viz-probe-pin-host-${slot}`);
    if (!host || !result?.marker) return null;
    const marker = result.marker;
    if (marker.type === 'node' && Array.isArray(marker.render_xy)) {
      const p = this._vizRenderPixelToPanel(slot, marker.render_xy[0], marker.render_xy[1]);
      if (!p) return null;
      const el = document.createElement('div');
      el.className = 'viz-probe-marker viz-probe-marker-node';
      el.dataset.cardId = String(cardId);
      el.style.left = `${p.x}px`;
      el.style.top = `${p.y}px`;
      if (labelText) {
        const lab = document.createElement('span');
        lab.className = 'viz-probe-marker-label';
        lab.textContent = labelText;
        el.appendChild(lab);
      }
      host.appendChild(el);
      return el;
    }
    if (marker.type === 'element' && Array.isArray(marker.render_edges)) {
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.classList.add('viz-probe-marker', 'viz-probe-marker-element');
      svg.dataset.cardId = String(cardId);
      marker.render_edges.forEach((edge) => {
        if (!Array.isArray(edge) || edge.length < 2) return;
        const p0 = this._vizRenderPixelToPanel(slot, edge[0]?.[0], edge[0]?.[1]);
        const p1 = this._vizRenderPixelToPanel(slot, edge[1]?.[0], edge[1]?.[1]);
        if (!p0 || !p1) return;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', String(p0.x));
        line.setAttribute('y1', String(p0.y));
        line.setAttribute('x2', String(p1.x));
        line.setAttribute('y2', String(p1.y));
        svg.appendChild(line);
      });
      if (!svg.childNodes.length) return null;
      if (labelText && Array.isArray(marker.label_xy)) {
        const lp = this._vizRenderPixelToPanel(slot, marker.label_xy[0], marker.label_xy[1]);
        if (lp) {
          const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          text.setAttribute('x', String(lp.x + 8));
          text.setAttribute('y', String(lp.y - 8));
          text.classList.add('viz-probe-marker-svg-label');
          text.textContent = labelText;
          svg.appendChild(text);
        }
      }
      host.appendChild(svg);
      return svg;
    }
    return null;
  },

  _vizProbeUpdateHoverMarker(slot, result) {
    this._vizProbeRemoveHoverMarker();
    const el = this._vizProbeCreateMarker(slot, result, 'hover');
    if (el) {
      el.classList.add('viz-probe-marker-hover');
      this._viz.probeHoverMarkerEl = el;
    }
  },

  _vizProbeRemoveHoverMarker() {
    const el = this._viz.probeHoverMarkerEl;
    if (el) {
      try { el.parentElement?.removeChild(el); } catch (_) {}
    }
    this._viz.probeHoverMarkerEl = null;
  },

  _vizProbeMarkerLabel(cardId) {
    const idx = this._viz.probeCards.findIndex(c => String(c.id) === String(cardId));
    return idx >= 0 ? `探针${idx + 1}` : '';
  },

  _vizProbeRefreshMarkers(slot) {
    const cards = (this._viz.probeCards || []).filter(c => c.slot === slot);
    if (!cards.length) return;
    const markers = cards.map(card => {
      const r = card.result || {};
      return {
        card_id: card.id,
        mode: card.mode,
        point_id: r.point_id,
        world_xyz: r.world_xyz,
        cell_id: r.cell_id,
      };
    });
    this._vizSendWs(slot, { action: 'project_probe_markers', markers });
  },

  _vizProbeApplyMarkerProjection(slot, msg) {
    const updates = Array.isArray(msg?.markers) ? msg.markers : [];
    updates.forEach(update => {
      const card = this._viz.probeCards.find(c => String(c.id) === String(update.card_id) && c.slot === slot);
      if (!card) return;
      try { card.markerEl?.parentElement?.removeChild(card.markerEl); } catch (_) {}
      if (update.marker) {
        card.result.marker = update.marker;
        card.markerEl = this._vizProbeCreateMarker(slot, card.result, card.id, this._vizProbeMarkerLabel(card.id));
      } else {
        card.markerEl = null;
      }
    });
  },

  _vizProbeFieldShortLabel(cardOrResult, slotOverride = null, gpIndex = null) {
    const result = cardOrResult?.result || cardOrResult || {};
    const slot = slotOverride || cardOrResult?.slot || '';
    const info = this._vizSlotInfo(slot) || {};
    const key = result.field_key || cardOrResult?.field_key || '';
    const meta = (info.field_meta || {})[key] || {};
    const code = this._vizVariableCode(key, meta);
    const comp = result.component_index ?? cardOrResult?.component_index ?? null;
    let suffix = '';
    if (comp !== null && comp !== undefined && !Number.isNaN(Number(comp))) {
      const labels = this._vizComponentLabels(meta.family || '', meta, key);
      suffix = labels[Number(comp)] || String(Number(comp) + 1);
      if (code === 'PE' && /^E/.test(suffix)) suffix = `PE${suffix.slice(1)}`;
      if (code === 'S' && /^S/.test(suffix)) return suffix;
      if (code === 'E' && /^E/.test(suffix)) return suffix;
      if (code === 'PE' && /^PE/.test(suffix)) return suffix;
      return `${code}${suffix}`;
    }
    if (code === 'S') suffix = 'mises';
    else if (code === 'E') suffix = 'equiv';
    else if (code === 'PE') suffix = 'mag';
    else if (['U', 'NFORC', 'RF'].includes(code)) suffix = 'mag';
    const base = suffix ? `${code}_${suffix}` : code;
    if (gpIndex !== null && gpIndex !== undefined && result.mode === 'element') {
      return `${base} GP${Number(gpIndex) + 1}`;
    }
    return base;
  },

  _vizProbeSubjectLabel(card, gpIndex = null) {
    const r = card?.result || {};
    const field = this._vizProbeFieldShortLabel(card, card.slot, gpIndex);
    if (card.mode === 'node') return `${field} (Node=${r.point_id ?? '-'})`;
    const gp = gpIndex !== null && gpIndex !== undefined ? `, GP${Number(gpIndex) + 1}` : '';
    return `${field} (Elem=${r.cell_id ?? '-'}${gp})`;
  },

  _vizProbeRenderCardBody(slot, result, pinned) {
    const slotLabel = String(slot || '').toUpperCase();
    const info = this._vizSlotInfo(slot) || {};
    const source = info.source || '';
    const sourceShort = source ? source.split(/[\\/]/).pop() : '-';
    const modeLabel = result.mode === 'element' ? '单元' : '节点';
    const fieldLabel = this._vizProbeFieldShortLabel(result, slot);
    const frameIdx = Number(result.frame_idx ?? 0) + 1;
    let frameTxt = `Frame ${frameIdx}`;
    if (result.frame_time !== null && result.frame_time !== undefined) {
      frameTxt += ` (t=${this._vizFormatNumber(result.frame_time, 4)})`;
    }
    const headRight = pinned
      ? `<span class="viz-probe-card-close" title="关闭这张卡">×</span>`
      : '';
    const head = `
      <div class="viz-probe-card-head">
        <span class="viz-probe-card-mode" data-mode="${UI.escapeAttr(result.mode)}">${modeLabel}</span>
        <span class="viz-probe-card-source">${UI.escapeHtml(slotLabel + '-' + sourceShort)}</span>
        ${pinned ? '<span class="viz-probe-card-index" data-probe-card-index></span>' : ''}
        ${headRight}
      </div>`;
    const fieldRow = `
      <div class="viz-probe-card-row"><span>Field</span><strong>${UI.escapeHtml(fieldLabel)}</strong></div>
      <div class="viz-probe-card-row"><span>Frame</span><strong>${UI.escapeHtml(frameTxt)}</strong></div>`;
    if (result.mode === 'node') {
      const xyz = result.world_xyz || [];
      const coordTxt = xyz.length === 3
        ? `(${this._vizFormatNumber(xyz[0])}, ${this._vizFormatNumber(xyz[1])}, ${this._vizFormatNumber(xyz[2])})`
        : '-';
      const owning = Array.isArray(result.owning_cell_ids) ? result.owning_cell_ids : [];
      const owningTxt = owning.length
        ? `${owning.length} (${owning.slice(0, 6).join(', ')}${owning.length > 6 ? '…' : ''})`
        : '0';
      return head + fieldRow + `
      <div class="viz-probe-card-row"><span>Node</span><strong>${result.point_id}</strong></div>
      <div class="viz-probe-card-row"><span>Coord</span><strong>${UI.escapeHtml(coordTxt)}</strong></div>
      <div class="viz-probe-card-row"><span>Value</span><strong>${this._vizFormatNumber(result.value, 6)}</strong></div>
      <div class="viz-probe-card-row"><span>Belong (Ele)</span><strong>${UI.escapeHtml(owningTxt)}</strong></div>`;
    }
    // element mode
    const c = result.centroid_xyz || [];
    const cTxt = c.length === 3
      ? `(${this._vizFormatNumber(c[0])}, ${this._vizFormatNumber(c[1])}, ${this._vizFormatNumber(c[2])})`
      : '-';
    const gauss = Array.isArray(result.gauss_values) ? result.gauss_values : [];
    const gpItems = gauss.map((v, i) =>
      `<div><code>GP${i + 1}</code><strong>${this._vizFormatNumber(v, 6)}</strong></div>`
    ).join('');
    return head + fieldRow + `
      <div class="viz-probe-card-row"><span>Element</span><strong>${result.cell_id}</strong></div>
      <div class="viz-probe-card-row"><span>Centroid</span><strong>${UI.escapeHtml(cTxt)}</strong></div>
      <div class="viz-probe-card-gauss">Gauss values (${result.n_gauss} GP):</div>
      <div class="viz-probe-card-gauss-list">${gpItems}</div>`;
  },

  _vizFormatNumber(v, sig = 4) {
    if (v === null || v === undefined || Number.isNaN(v)) return '-';
    const n = Number(v);
    if (!Number.isFinite(n)) return '-';
    const a = Math.abs(n);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return n.toExponential(Math.max(1, sig - 1));
    return n.toPrecision(Math.max(2, sig));
  },

  _vizProbeAttachPinDrag(el, panel) {
    let dx = 0, dy = 0, dragging = false;
    el.addEventListener('mousedown', (ev) => {
      if (ev.button !== 0) return;
      const tgt = ev.target;
      if (tgt && tgt.classList && tgt.classList.contains('viz-probe-card-close')) return;
      const r = el.getBoundingClientRect();
      if (ev.clientX >= r.right - 18 && ev.clientY >= r.bottom - 18) return;
      ev.preventDefault();
      ev.stopPropagation();
      dx = ev.clientX - r.left;
      dy = ev.clientY - r.top;
      dragging = true;
      el.classList.add('viz-probe-dragging');
      const onMove = (m) => {
        if (!dragging) return;
        const pr = panel.getBoundingClientRect();
        let nl = m.clientX - pr.left - dx;
        let nt = m.clientY - pr.top - dy;
        const w = el.offsetWidth;
        const h = el.offsetHeight;
        nl = Math.max(0, Math.min(nl, pr.width - w));
        nt = Math.max(0, Math.min(nt, pr.height - h));
        el.style.left = `${nl}px`;
        el.style.top = `${nt}px`;
      };
      const onUp = () => {
        dragging = false;
        el.classList.remove('viz-probe-dragging');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      };
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  },

  _vizProbeShowCompareModal(show) {
    const modal = document.getElementById('viz-probe-compare-modal');
    if (modal) modal.style.display = show ? 'flex' : 'none';
  },

  _vizProbeDefaultCompareSettings() {
    const cards = this._viz.probeCards || [];
    const a = cards[0]?.id ?? null;
    const b = cards[1]?.id ?? a;
    return {
      groups: [{
        a, b,
        gpA: 0,
        gpB: 0,
        allGp: false,
        separateAxis: false,
        logMode: 'none',
      }],
    };
  },

  _vizProbeOpenCompare() {
    if (!this._viz.probeCards.length) {
      this._vizSetStatus('请先固定至少一个探针。');
      return;
    }
    this._vizProbeCloseCharts();
    if (!this._viz.probeCompareSettings) {
      this._viz.probeCompareSettings = this._vizProbeDefaultCompareSettings();
    }
    this._vizProbeNormalizeCompareSettings();
    this._vizProbeRenderCompareModal();
    this._vizProbeShowCompareModal(true);
  },

  _vizProbeNormalizeCompareSettings() {
    const ids = new Set((this._viz.probeCards || []).map(c => c.id));
    const fallback = this._viz.probeCards[0]?.id ?? null;
    const settings = this._viz.probeCompareSettings || this._vizProbeDefaultCompareSettings();
    let groups = Array.isArray(settings.groups) && settings.groups.length ? settings.groups : this._vizProbeDefaultCompareSettings().groups;
    groups = groups.slice(0, 5).map(g => ({
      a: ids.has(g.a) ? g.a : fallback,
      b: ids.has(g.b) ? g.b : fallback,
      gpA: Number.isFinite(Number(g.gpA)) ? Number(g.gpA) : 0,
      gpB: Number.isFinite(Number(g.gpB)) ? Number(g.gpB) : 0,
      allGp: !!g.allGp,
      separateAxis: !!g.separateAxis,
      logMode: ['none', 'y1', 'y2', 'both'].includes(g.logMode) ? g.logMode : 'none',
    }));
    this._viz.probeCompareSettings = { groups };
  },

  _vizProbeCardOptionLabel(card, idx) {
    const r = card.result || {};
    const idText = card.mode === 'node' ? `Node=${r.point_id ?? '-'}` : `Elem=${r.cell_id ?? '-'}`;
    return `探针${idx + 1} · ${card.mode === 'node' ? '节点' : '单元'} · ${idText} · ${this._vizProbeFieldShortLabel(card)}`;
  },

  _vizProbeGpOptions(card, selected, forceAll = false) {
    if (forceAll) return '<option value="0" selected>全部</option>';
    if (!card || card.mode !== 'element') return '<option value="0">-</option>';
    const n = Math.max(1, Number(card.result?.n_gauss || 1));
    return Array.from({ length: n }, (_, i) =>
      `<option value="${i}" ${Number(selected) === i ? 'selected' : ''}>GP${i + 1}</option>`
    ).join('');
  },

  _vizProbeRenderCompareModal() {
    this._vizProbeNormalizeCompareSettings();
    const list = document.getElementById('viz-probe-compare-list');
    if (!list) return;
    const cards = this._viz.probeCards || [];
    const opts = cards.map((card, idx) =>
      `<option value="${card.id}">${UI.escapeHtml(this._vizProbeCardOptionLabel(card, idx))}</option>`
    ).join('');
    list.innerHTML = this._viz.probeCompareSettings.groups.map((g, idx) => {
      const ca = cards.find(c => c.id === g.a) || cards[0];
      const cb = cards.find(c => c.id === g.b) || cards[0];
      const canRemove = this._viz.probeCompareSettings.groups.length > 1;
      const canAdd = this._viz.probeCompareSettings.groups.length < 5;
      return `
        <div class="viz-probe-compare-row" data-compare-index="${idx}">
          <div class="viz-probe-compare-side">
            <button class="viz-btn viz-probe-compare-side-btn viz-probe-compare-add-row" title="在后面增加一组比较" ${canAdd ? '' : 'disabled'}>+</button>
            <button class="viz-btn viz-probe-compare-side-btn viz-probe-compare-remove" title="删除这组比较" ${canRemove ? '' : 'disabled'}>-</button>
          </div>
          <div class="viz-probe-compare-main">
            <div class="viz-probe-compare-row-head">
              <strong>比较 ${idx + 1}</strong>
            </div>
            <div class="viz-probe-compare-grid">
              <div></div>
              <div class="viz-probe-compare-colhead">探针</div>
              <div class="viz-probe-compare-colhead">子项</div>
              <div class="viz-probe-compare-index">1</div>
              <label>
                <select data-role="a">${opts}</select>
              </label>
              <label>
                <select data-role="gpA" ${g.allGp || ca?.mode !== 'element' ? 'disabled' : ''}>${this._vizProbeGpOptions(ca, g.gpA, g.allGp)}</select>
              </label>
              <div class="viz-probe-compare-index">2</div>
              <label>
                <select data-role="b">${opts}</select>
              </label>
              <label>
                <select data-role="gpB" ${g.allGp || cb?.mode !== 'element' ? 'disabled' : ''}>${this._vizProbeGpOptions(cb, g.gpB, g.allGp)}</select>
              </label>
            </div>
            <div class="viz-probe-compare-shared">
              <label class="viz-inline-check"><input type="checkbox" data-role="allGp" ${g.allGp ? 'checked' : ''}><span>所有子项</span></label>
              <label class="viz-inline-check"><input type="checkbox" data-role="separateAxis" ${g.separateAxis ? 'checked' : ''}><span>各自y轴</span></label>
              <label class="viz-probe-compare-log">对数坐标
                <select data-role="logMode">
                  <option value="none" ${g.logMode === 'none' ? 'selected' : ''}>无</option>
                  <option value="y1" ${g.logMode === 'y1' ? 'selected' : ''}>y1</option>
                  <option value="y2" ${g.logMode === 'y2' ? 'selected' : ''} ${g.separateAxis ? '' : 'disabled'}>y2</option>
                  <option value="both" ${g.logMode === 'both' ? 'selected' : ''} ${g.separateAxis ? '' : 'disabled'}>两个</option>
                </select>
              </label>
            </div>
          </div>
        </div>`;
    }).join('');
    this._viz.probeCompareSettings.groups.forEach((g, idx) => {
      const row = list.querySelector(`[data-compare-index="${idx}"]`);
      if (!row) return;
      row.querySelector('[data-role="a"]').value = String(g.a);
      row.querySelector('[data-role="b"]').value = String(g.b);
    });
    list.querySelectorAll('.viz-probe-compare-row').forEach(row => {
      const idx = Number(row.dataset.compareIndex || 0);
      row.querySelectorAll('select,input').forEach(el => {
        el.addEventListener('change', () => this._vizProbeCompareReadRow(idx, true));
      });
      row.querySelector('.viz-probe-compare-remove')?.addEventListener('click', () => {
        this._viz.probeCompareSettings.groups.splice(idx, 1);
        this._vizProbeRenderCompareModal();
      });
      row.querySelector('.viz-probe-compare-add-row')?.addEventListener('click', () => {
        if (this._viz.probeCompareSettings.groups.length >= 5) return;
        this._vizProbeCompareReadRow(idx, false);
        const src = this._viz.probeCompareSettings.groups[idx] || this._vizProbeDefaultCompareSettings().groups[0];
        this._viz.probeCompareSettings.groups.splice(idx + 1, 0, { ...src });
        this._vizProbeRenderCompareModal();
      });
    });
  },

  _vizProbeCompareReadRow(idx, rerender = false) {
    const row = document.querySelector(`#viz-probe-compare-list [data-compare-index="${idx}"]`);
    if (!row) return;
    const g = this._viz.probeCompareSettings.groups[idx];
    g.a = Number(row.querySelector('[data-role="a"]')?.value || 0);
    g.b = Number(row.querySelector('[data-role="b"]')?.value || 0);
    g.gpA = Number(row.querySelector('[data-role="gpA"]')?.value || 0);
    g.gpB = Number(row.querySelector('[data-role="gpB"]')?.value || 0);
    g.allGp = !!row.querySelector('[data-role="allGp"]')?.checked;
    g.separateAxis = !!row.querySelector('[data-role="separateAxis"]')?.checked;
    g.logMode = row.querySelector('[data-role="logMode"]')?.value || 'none';
    if (!g.separateAxis && (g.logMode === 'y2' || g.logMode === 'both')) g.logMode = 'y1';
    if (rerender) this._vizProbeRenderCompareModal();
  },

  _vizProbeCompareAddGroup() {
    this._vizProbeNormalizeCompareSettings();
    if (this._viz.probeCompareSettings.groups.length >= 5) return;
    const prev = this._viz.probeCompareSettings.groups[this._viz.probeCompareSettings.groups.length - 1] || this._vizProbeDefaultCompareSettings().groups[0];
    this._viz.probeCompareSettings.groups.push({ ...prev });
    this._vizProbeRenderCompareModal();
  },

  _vizProbeCloseCharts() {
    document.querySelectorAll('.viz-probe-chart').forEach(el => el.remove());
  },

  _vizProbeSeriesForCard(card, gpIndex, axis = 1, style = 'solid', groupIndex = 0, allGp = false) {
    const r = card.result || {};
    const out = [];
    if (card.mode === 'node') {
      const frames = Array.isArray(r.frame_series) && r.frame_series.length
        ? r.frame_series
        : [{ frame_idx: card.frame_idx, frame_time: card.frame_time, value: r.value }];
      out.push({
        card, axis, style, groupIndex, gpIndex: null,
        label: this._vizProbeSubjectLabel(card),
        points: frames.map(fr => ({ frame_index: fr.frame_idx ?? '', frame_time: fr.frame_time ?? fr.frame_idx ?? 0, value: Number(fr.value) })),
      });
      return out;
    }
    const frames = Array.isArray(r.frame_series) && r.frame_series.length
      ? r.frame_series
      : [{ frame_idx: card.frame_idx, frame_time: card.frame_time, n_gauss: r.n_gauss, gauss_values: r.gauss_values || [] }];
    const n = Math.max(1, Number(r.n_gauss || frames[0]?.n_gauss || 1));
    const gpList = allGp ? Array.from({ length: n }, (_, i) => i) : [Math.max(0, Math.min(Number(gpIndex || 0), n - 1))];
    gpList.forEach(gp => {
      out.push({
        card, axis, style, groupIndex, gpIndex: gp,
        label: this._vizProbeSubjectLabel(card, gp),
        points: frames.map(fr => {
          const arr = Array.isArray(fr.gauss_values) ? fr.gauss_values : [];
          return { frame_index: fr.frame_idx ?? '', frame_time: fr.frame_time ?? fr.frame_idx ?? 0, value: Number(arr[gp]) };
        }),
      });
    });
    return out;
  },

  _vizProbeCollectCompareSeries(group, groupIndex) {
    const cards = this._viz.probeCards || [];
    const a = cards.find(c => c.id === group.a) || cards[0];
    const b = cards.find(c => c.id === group.b) || a;
    if (!a || !b) return [];
    const ax1 = 1;
    const ax2 = group.separateAxis ? 2 : 1;
    return [
      ...this._vizProbeSeriesForCard(a, group.gpA, ax1, 'solid', groupIndex, group.allGp),
      ...this._vizProbeSeriesForCard(b, group.gpB, ax2, 'dash', groupIndex, group.allGp),
    ];
  },

  _vizProbeParula(i, n) {
    const colors = ['#352a87', '#0f5cdd', '#1484d4', '#06a7c6', '#38b99e', '#7dbf7b', '#c7c75a', '#f6d746', '#f9b233', '#f17c22'];
    if (n <= 1) return colors[0];
    return colors[Math.round((colors.length - 1) * (i / (n - 1)))];
  },

  _vizProbeRunCompare() {
    this._vizProbeNormalizeCompareSettings();
    this._viz.probeCompareSettings.groups.forEach((_, idx) => this._vizProbeCompareReadRow(idx, false));
    this._vizProbeCloseCharts();
    this._viz.probeCompareSettings.groups.forEach((group, idx) => {
      const series = this._vizProbeCollectCompareSeries(group, idx);
      if (series.length) this._vizProbeCreateCompareChart(series, group, idx);
    });
    this._vizProbeShowCompareModal(false);
  },

  _vizProbeAxisScale(values, logScale) {
    const clean = values.filter(v => Number.isFinite(v) && (!logScale || v > 0));
    if (!clean.length) return { min: logScale ? 1e-6 : 0, max: logScale ? 1 : 1, log: logScale };
    if (logScale) {
      const minP = Math.floor(Math.log10(Math.max(Math.min(...clean), 1e-30)));
      const maxP = Math.ceil(Math.log10(Math.max(...clean)));
      return { min: 10 ** minP, max: 10 ** maxP, log: true };
    }
    let min = Math.min(...clean);
    let max = Math.max(...clean);
    if (min === max) {
      const d = Math.abs(min || 1) * 0.1;
      min -= d;
      max += d;
    }
    const pad = (max - min) * 0.08;
    return { min: min - pad, max: max + pad, log: false };
  },

  _vizProbeMapY(v, scale, top, h) {
    if (!Number.isFinite(v)) return null;
    if (scale.log) {
      if (v <= 0) return null;
      const a = Math.log10(scale.min);
      const b = Math.log10(scale.max);
      return top + h - ((Math.log10(v) - a) / (b - a || 1)) * h;
    }
    return top + h - ((v - scale.min) / (scale.max - scale.min || 1)) * h;
  },

  _vizProbeSameNumericAxis(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i += 1) {
      const av = Number(a[i]);
      const bv = Number(b[i]);
      if (!Number.isFinite(av) || !Number.isFinite(bv)) return false;
      if (Math.abs(av - bv) > 1e-9 * Math.max(1, Math.abs(av), Math.abs(bv))) return false;
    }
    return true;
  },

  _vizProbeCompareXAxis(series) {
    const timeAxes = series.map(s => (s.points || []).map(p => Number(p.frame_time)));
    const indexAxes = series.map(s => (s.points || []).map(p => Number(p.frame_index)));
    const solidAxes = series.filter(s => s.style !== 'dash').map(s => (s.points || []).map(p => Number(p.frame_time)));
    const dashAxes = series.filter(s => s.style === 'dash').map(s => (s.points || []).map(p => Number(p.frame_time)));
    const allTimesFinite = timeAxes.length > 0 && timeAxes.every(axis => axis.length > 0 && axis.every(Number.isFinite));
    const sameWithinEachSide = [solidAxes, dashAxes].every(groupAxes => (
      groupAxes.length <= 1 || groupAxes.every(axis => this._vizProbeSameNumericAxis(axis, groupAxes[0]))
    ));
    const sameBetweenSides = solidAxes.length === 0 || dashAxes.length === 0
      ? true
      : this._vizProbeSameNumericAxis(solidAxes[0], dashAxes[0]);
    const useTime = allTimesFinite && sameWithinEachSide && sameBetweenSides;
    const allIndicesFinite = indexAxes.length > 0 && indexAxes.some(axis => axis.some(Number.isFinite));
    return {
      key: useTime ? 'frame_time' : 'frame_index',
      label: useTime ? 'time' : 'frame',
      fallbackKey: useTime ? 'frame_index' : 'frame_time',
      hasFrameIndex: allIndicesFinite,
    };
  },

  _vizProbePointX(point, xAxis) {
    const primary = Number(point?.[xAxis.key]);
    if (Number.isFinite(primary)) return primary;
    const fallback = Number(point?.[xAxis.fallbackKey]);
    return Number.isFinite(fallback) ? fallback : null;
  },

  _vizProbeXTicks(xmin, xmax, xs, xAxis) {
    if (xAxis.key === 'frame_index') {
      const unique = [...new Set(xs.map(v => Math.round(v * 1e9) / 1e9))].sort((a, b) => a - b);
      if (unique.length <= 7) return unique;
      const ticks = Array.from({ length: 5 }, (_, i) => Math.round(xmin + (xmax - xmin) * i / 4));
      return [...new Set(ticks)].filter(v => v >= xmin && v <= xmax);
    }
    return Array.from({ length: 5 }, (_, i) => xmin + (xmax - xmin) * i / 4);
  },

  _vizProbeCreateCompareChart(series, group, groupIndex) {
    const chartId = this._viz.probeNextChartId++;
    const wrap = document.createElement('div');
    wrap.className = 'viz-probe-chart';
    wrap.style.left = `${80 + groupIndex * 34}px`;
    wrap.style.top = `${110 + groupIndex * 34}px`;
    const svg = this._vizProbeRenderCompareSvg(series, group, groupIndex);
    wrap.innerHTML = `
      <div class="viz-probe-chart-head">
        <strong>探针比较 ${groupIndex + 1}</strong>
        <div>
          <button class="viz-btn" data-action="save-svg">保存图片</button>
          <button class="viz-btn" data-action="save-csv">保存数据</button>
          <button class="viz-btn" data-action="close">关闭</button>
        </div>
      </div>
      <div class="viz-probe-chart-body">${svg}</div>`;
    wrap.dataset.chartId = String(chartId);
    document.body.appendChild(wrap);
    this._vizProbeAttachChartDrag(wrap);
    wrap.querySelector('[data-action="close"]')?.addEventListener('click', () => wrap.remove());
    wrap.querySelector('[data-action="save-svg"]')?.addEventListener('click', () => {
      const content = wrap.querySelector('svg')?.outerHTML || svg;
      this._vizDownloadText(`probe_compare_${groupIndex + 1}.svg`, content, 'image/svg+xml;charset=utf-8');
    });
    wrap.querySelector('[data-action="save-csv"]')?.addEventListener('click', () => {
      this._vizDownloadText(`probe_compare_${groupIndex + 1}.csv`, this._vizProbeCompareCsv(series), 'text/csv;charset=utf-8');
    });
  },

  _vizProbeRenderCompareSvg(series, group, groupIndex) {
    const W = 900, L = 92, R = group.separateAxis ? 92 : 36, T = 26, PH = 300;
    const PW = W - L - R;
    const xAxis = this._vizProbeCompareXAxis(series);
    const xs = series.flatMap(s => s.points.map(p => this._vizProbePointX(p, xAxis))).filter(Number.isFinite);
    let xmin = Math.min(...xs), xmax = Math.max(...xs);
    if (!Number.isFinite(xmin) || xmin === xmax) { xmin = 0; xmax = Math.max(1, xmax || 1); }
    if (xmin > 0 && xAxis.key === 'frame_index') xmin = 0;
    const isLog1 = group.logMode === 'y1' || group.logMode === 'both' || (!group.separateAxis && group.logMode !== 'none');
    const isLog2 = group.separateAxis && (group.logMode === 'y2' || group.logMode === 'both');
    const y1Vals = series.filter(s => s.axis === 1).flatMap(s => s.points.map(p => p.value));
    const y2Vals = series.filter(s => s.axis === 2).flatMap(s => s.points.map(p => p.value));
    const y1 = this._vizProbeAxisScale(y1Vals.concat(group.separateAxis ? [] : y2Vals), isLog1);
    const y2 = this._vizProbeAxisScale(y2Vals, isLog2);
    const xMap = x => L + ((x - xmin) / (xmax - xmin || 1)) * PW;
    const yMap = (v, axis) => this._vizProbeMapY(v, axis === 2 ? y2 : y1, T, PH);
    const xAxisY = y1.min < 0 && y1.max > 0 && !y1.log ? yMap(0, 1) : T + PH;
    const yAxisX = xmin < 0 && xmax > 0 ? xMap(0) : L;
    const ticks = (min, max, log) => {
      if (log) {
        const a = Math.round(Math.log10(min));
        const b = Math.round(Math.log10(max));
        return Array.from({ length: Math.max(1, b - a + 1) }, (_, i) => 10 ** (a + i));
      }
      return Array.from({ length: 5 }, (_, i) => min + (max - min) * i / 4);
    };
    const fmt = v => {
      if (!Number.isFinite(v)) return '';
      const av = Math.abs(v);
      if (av > 0 && (av < 1e-3 || av >= 1e5)) return v.toExponential(0).replace('e', 'e');
      return Number(v).toPrecision(3);
    };
    const fmtX = v => {
      if (xAxis.key === 'frame_index' && Math.abs(v - Math.round(v)) < 1e-8) return String(Math.round(v));
      return fmt(v);
    };
    const yTicks1 = ticks(y1.min, y1.max, y1.log);
    const yTicks2 = group.separateAxis ? ticks(y2.min, y2.max, y2.log) : [];
    const colors = series.map((_, i) => this._vizProbeParula(i, series.length));
    const paths = series.map((s, i) => {
      const d = s.points.map(p => {
        const xValue = this._vizProbePointX(p, xAxis);
        if (xValue === null) return null;
        const x = xMap(xValue);
        const y = yMap(Number(p.value), s.axis);
        return y === null ? null : `${x.toFixed(1)},${y.toFixed(1)}`;
      }).filter(Boolean);
      const dash = s.style === 'dash' ? 'stroke-dasharray="8 5"' : '';
      return d.length ? `<polyline points="${d.join(' ')}" fill="none" stroke="${colors[i]}" stroke-width="2.4" ${dash}/>` : '';
    }).join('');
    const axisLabelY = T + PH + 58;
    const legendCols = series.length > 8 ? 2 : 1;
    const legendRowH = 22;
    const legendRows = Math.ceil(series.length / legendCols);
    const legendX = L;
    const legendY = axisLabelY + 22;
    const legendW = PW;
    const legendColW = legendW / legendCols;
    const legendH = Math.max(34, 14 + legendRows * legendRowH);
    const H = legendY + legendH + 22;
    const legend = series.map((s, i) => {
      const col = i % legendCols;
      const row = Math.floor(i / legendCols);
      return `<g transform="translate(${legendX + 14 + col * legendColW},${legendY + 21 + row * legendRowH})">
        <line x1="0" y1="0" x2="24" y2="0" stroke="${colors[i]}" stroke-width="2.4" ${s.style === 'dash' ? 'stroke-dasharray="8 5"' : ''}/>
        <text x="32" y="5" fill="#111827" font-size="16">${UI.escapeHtml(s.label)}</text>
      </g>`;
    }).join('');
    const yAxis1 = yTicks1.map(v => {
      const y = yMap(v, 1);
      return `<line x1="${L}" x2="${L + PW}" y1="${y}" y2="${y}" stroke="#d7d7d7" stroke-width="1"/><text x="${yAxisX - 10}" y="${y + 5}" fill="#111827" font-size="18" text-anchor="end">${fmt(v)}</text>`;
    }).join('');
    const yAxis2 = yTicks2.map(v => {
      const y = yMap(v, 2);
      return `<text x="${L + PW + 10}" y="${y + 5}" fill="#111827" font-size="18">${fmt(v)}</text>`;
    }).join('');
    const xTicks = this._vizProbeXTicks(xmin, xmax, xs, xAxis).map(v => {
      const x = xMap(v);
      return `<line x1="${x}" x2="${x}" y1="${T}" y2="${T + PH}" stroke="#d7d7d7" stroke-width="1"/><line x1="${x}" x2="${x}" y1="${xAxisY - 10}" y2="${xAxisY}" stroke="#111827" stroke-width="2"/><text x="${x}" y="${T + PH + 30}" fill="#111827" font-size="18" text-anchor="middle">${fmtX(v)}</text>`;
    }).join('');
    return `<svg class="viz-probe-chart-svg" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
      <rect width="${W}" height="${H}" fill="#ffffff"/>
      <style>
        .probe-chart-text { font-family: "Latin Modern Roman", "CMU Serif", "Cambria Math", "Times New Roman", serif; }
      </style>
      <g class="probe-chart-text">
      ${yAxis1}${yAxis2}${xTicks}
      <rect x="${L}" y="${T}" width="${PW}" height="${PH}" fill="none" stroke="#111827" stroke-width="2"/>
      <line x1="${yAxisX}" y1="${T}" x2="${yAxisX}" y2="${T + PH}" stroke="#111827" stroke-width="2"/>
      <line x1="${L}" y1="${xAxisY}" x2="${L + PW}" y2="${xAxisY}" stroke="#111827" stroke-width="2"/>
      ${group.separateAxis ? `<line x1="${L + PW}" y1="${T}" x2="${L + PW}" y2="${T + PH}" stroke="#111827" stroke-width="2"/>` : ''}
      ${paths}
      <text x="${L + PW / 2}" y="${axisLabelY}" fill="#111827" font-size="22" text-anchor="middle" font-style="italic">${xAxis.label} <tspan font-size="18" font-style="normal">[-]</tspan></text>
      <rect x="${legendX}" y="${legendY}" width="${legendW}" height="${legendH}" fill="#ffffff" stroke="#111827" stroke-width="2"/>
      ${legend}
      </g>
    </svg>`;
  },

  _vizProbeCompareCsv(series) {
    const cols = [
      'probe_id', 'mode',
      'node_id', 'coord_x', 'coord_y', 'coord_z',
      'element_id', 'gauss_index', 'n_gauss',
      'field_label', 'field_key', 'component_index', 'value',
      'frame_index', 'frame_time', 'owning_elements',
      'slot', 'mat_path', 'mat_mtime_iso', 'mat_size_bytes',
      'producer_module', 'analysis_type',
    ];
    const esc = v => {
      if (v === null || v === undefined) return '';
      const s = String(v);
      return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const rows = [];
    series.forEach(s => {
      const card = s.card;
      const r = card.result || {};
      const idx = this._viz.probeCards.findIndex(c => c.id === card.id) + 1;
      const m = card.matIdentity || {};
      const xyz = card.mode === 'node' ? (r.world_xyz || []) : (r.centroid_xyz || []);
      s.points.forEach(p => rows.push({
        probe_id: idx,
        mode: card.mode,
        node_id: card.mode === 'node' ? r.point_id ?? '' : '',
        coord_x: xyz[0] ?? '',
        coord_y: xyz[1] ?? '',
        coord_z: xyz[2] ?? '',
        element_id: card.mode === 'element' ? r.cell_id ?? '' : '',
        gauss_index: card.mode === 'element' ? s.gpIndex ?? '' : '',
        n_gauss: card.mode === 'element' ? r.n_gauss ?? '' : '',
        field_label: this._vizProbeFieldShortLabel(card, card.slot, s.gpIndex),
        field_key: card.field_key || '',
        component_index: card.component_index ?? '',
        value: p.value,
        frame_index: p.frame_index,
        frame_time: p.frame_time,
        owning_elements: card.mode === 'node' ? (r.owning_cell_ids || []).join('|') : '',
        slot: card.slot,
        mat_path: m.source_path || '',
        mat_mtime_iso: m.mtime_iso || '',
        mat_size_bytes: m.size_bytes ?? '',
        producer_module: m.producer_module || '',
        analysis_type: m.analysis_type || '',
      }));
    });
    return [cols.join(','), ...rows.map(row => cols.map(c => esc(row[c])).join(','))].join('\n');
  },

  _vizDownloadText(filename, text, type) {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 0);
  },

  _vizProbeAttachChartDrag(el) {
    let dragging = false, ox = 0, oy = 0;
    const head = el.querySelector('.viz-probe-chart-head');
    head?.addEventListener('mousedown', ev => {
      if (ev.button !== 0 || ev.target?.tagName === 'BUTTON') return;
      const r = el.getBoundingClientRect();
      ox = ev.clientX - r.left;
      oy = ev.clientY - r.top;
      dragging = true;
      ev.preventDefault();
    });
    document.addEventListener('mousemove', ev => {
      if (!dragging) return;
      el.style.left = `${Math.max(0, ev.clientX - ox)}px`;
      el.style.top = `${Math.max(0, ev.clientY - oy)}px`;
    });
    document.addEventListener('mouseup', () => { dragging = false; });
  },

  _vizProbeExportCsv() {
    const cards = this._viz.probeCards;
    if (!cards.length) return;
    const allFrames = !!document.getElementById('viz-probe-all-frames')?.checked;
    const cols = [
      'node_id', 'coord_x', 'coord_y', 'coord_z',
      'element_id', 'gauss_index', 'n_gauss',
      'field_label', 'field_key', 'component_index', 'value',
      'frame_index', 'frame_time',
      'owning_elements',
      'slot', 'mat_path', 'mat_mtime_iso', 'mat_size_bytes',
      'producer_module', 'analysis_type',
    ];
    cols.unshift('probe_id', 'mode');
    const csvCell = (v) => {
      if (v === null || v === undefined) return '';
      const s = String(v);
      if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
      return s;
    };
    const dataRows = [];
    cards.forEach((card, idx) => {
      const cardIdx = idx + 1;
      const m = card.matIdentity || {};
      const baseRow = {
        probe_id: cardIdx,
        mode: card.mode,
        slot: card.slot,
        mat_path: m.source_path || '',
        mat_mtime_iso: m.mtime_iso || '',
        mat_size_bytes: m.size_bytes ?? '',
        producer_module: m.producer_module || '',
        analysis_type: m.analysis_type || '',
        field_key: card.field_key || '',
        field_label: card.field_label || '',
        component_index: card.component_index ?? '',
        frame_index: card.frame_idx ?? '',
        frame_time: card.frame_time ?? '',
      };
      const r = card.result;
      if (card.mode === 'node') {
        const xyz = r.world_xyz || [];
        const series = allFrames && Array.isArray(r.frame_series) && r.frame_series.length
          ? r.frame_series
          : [{ frame_idx: card.frame_idx, frame_time: card.frame_time, value: r.value }];
        series.forEach((fr) => {
          const row = {
            ...baseRow,
            frame_index: fr.frame_idx ?? '',
            frame_time: fr.frame_time ?? '',
            node_id: r.point_id ?? '',
            coord_x: xyz[0] ?? '',
            coord_y: xyz[1] ?? '',
            coord_z: xyz[2] ?? '',
            element_id: '',
            gauss_index: '',
            n_gauss: '',
            value: fr.value ?? '',
            owning_elements: (r.owning_cell_ids || []).join('|'),
          };
          dataRows.push(row);
        });
      } else {
        const c0 = r.centroid_xyz || [];
        const series = allFrames && Array.isArray(r.frame_series) && r.frame_series.length
          ? r.frame_series
          : [{
              frame_idx: card.frame_idx,
              frame_time: card.frame_time,
              n_gauss: r.n_gauss,
              gauss_values: r.gauss_values || [],
            }];
        series.forEach((fr) => {
          const gauss = Array.isArray(fr.gauss_values) ? fr.gauss_values : [];
          if (gauss.length === 0) {
            // Defensive: element-located field with n_gauss=0 — emit an
            // empty-value row so the card still appears in the export.
            const row = {
              ...baseRow,
              frame_index: fr.frame_idx ?? '',
              frame_time: fr.frame_time ?? '',
              node_id: '',
              coord_x: c0[0] ?? '',
              coord_y: c0[1] ?? '',
              coord_z: c0[2] ?? '',
              element_id: r.cell_id ?? '',
              gauss_index: '',
              n_gauss: 0,
              value: '',
              owning_elements: '',
            };
            dataRows.push(row);
          } else {
            gauss.forEach((v, gi) => {
              const row = {
                ...baseRow,
                frame_index: fr.frame_idx ?? '',
                frame_time: fr.frame_time ?? '',
                node_id: '',
                coord_x: c0[0] ?? '',
                coord_y: c0[1] ?? '',
                coord_z: c0[2] ?? '',
                element_id: r.cell_id ?? '',
                gauss_index: gi,
                n_gauss: fr.n_gauss ?? r.n_gauss ?? gauss.length,
                value: v,
                owning_elements: '',
              };
              dataRows.push(row);
            });
          }
        });
      }
    });
    if (allFrames) {
      dataRows.sort((a, b) => {
        const frameA = Number.isFinite(Number(a.frame_index)) ? Number(a.frame_index) : Number.MAX_SAFE_INTEGER;
        const frameB = Number.isFinite(Number(b.frame_index)) ? Number(b.frame_index) : Number.MAX_SAFE_INTEGER;
        if (frameA !== frameB) return frameA - frameB;
        const gpA = Number.isFinite(Number(a.gauss_index)) ? Number(a.gauss_index) : -1;
        const gpB = Number.isFinite(Number(b.gauss_index)) ? Number(b.gauss_index) : -1;
        if (gpA !== gpB) return gpA - gpB;
        return Number(a.probe_id || 0) - Number(b.probe_id || 0);
      });
    }
    const rows = [cols.join(','), ...dataRows.map(row => cols.map(c => csvCell(row[c])).join(','))];
    const csv = rows.join('\n');
    const now = new Date();
    const pad = (x) => String(x).padStart(2, '0');
    const ts = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    const filename = `viz_probes_${ts}.csv`;
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 0);
    this._vizSetStatus(`已导出 ${cards.length} 张探针卡${allFrames ? '（全部 frames）' : ''}到 ${filename}`);
  },

  _vizProbeClearForSlot(slot) {
    // Called when a slot's MAT changes — drop pinned cards bound to
    // this slot to avoid stale references to a closed scene.
    const remaining = [];
    this._viz.probeCards.forEach(card => {
      if (card.slot === slot) {
        try { card.el?.parentElement?.removeChild(card.el); } catch (_) {}
        try { card.markerEl?.parentElement?.removeChild(card.markerEl); } catch (_) {}
      } else {
        remaining.push(card);
      }
    });
    this._viz.probeCards = remaining;
    this._vizProbeUpdateCounter();
    const exportBtn = document.getElementById('viz-probe-export');
    if (exportBtn) exportBtn.disabled = this._viz.probeCards.length === 0;
    const compareBtn = document.getElementById('viz-probe-compare');
    if (compareBtn) compareBtn.disabled = this._viz.probeCards.length === 0;
    this._viz.probeSavedFieldKey[slot] = '';
    this._viz.probeSavedFieldComponent[slot] = null;
  },

  async _vizRunDiff() {
    if (!(this._viz.info_a && this._viz.info_b && this._viz.compareMode)) {
      this._vizSetStatus('请先在对比模式下同时加载 A 和 B 两个场景。');
      return;
    }
    if (this._vizActiveUsesDiff()) {
      const msg = 'Diff 不能再次参与对比。请先在 A 和 B 中选择原始结果量，再点击“对比”；本次操作未执行。';
      this._vizSetStatus(msg);
      window.alert(msg);
      return;
    }
    this._vizSetStatus('正在计算 Diff...');
    const res = await this.apiPost('/api/viz/diff', {
      absolute: this._viz.absoluteDiff,
    }).catch(() => null);
    if (!res || !res.ok) {
      this._vizSetStatus(`Diff 计算失败：${res?.detail || '未知错误'}`);
      return;
    }
    this._viz.diffPayload = res;
    if (res.scene_a) {
      this._viz.info_a = res.scene_a;
      this._vizUpdateControls('a');
    }
    if (res.scene_b) {
      this._viz.info_b = res.scene_b;
      this._vizUpdateControls('b');
    }
    this._vizSendWs('a', {action: 'render'});
    this._vizSendWs('b', {action: 'render'});
    this._vizHideBoxplot();
    this._vizRefreshCompareButtons();
    this._vizSetStatus(`Diff 已生成：${res.diff_info_text || res.label || 'B-A'}`);
  },

  async _vizOpenDiffBoxplot() {
    if (!this._vizHasDiffField()) {
      this._vizSetStatus('请先点击“对比”生成 Diff。');
      return;
    }
    let payload = this._viz.diffPayload;
    if (!payload || !payload.stats) {
      payload = await this.apiPost('/api/viz/diff-boxplot', {}).catch(() => null);
      if (payload?.ok) this._viz.diffPayload = payload;
    }
    if (!payload || !payload.ok) {
      this._vizSetStatus(`箱线图生成失败：${payload?.detail || 'Diff 不可用'}`);
      return;
    }
    this._vizRenderBoxplot(payload);
    this._vizShowBoxplot(true);
    this._vizSetStatus('Diff 箱线图已生成');
  },

  async _vizSaveDiffMat() {
    if (!this._vizHasDiffField()) {
      this._vizSetStatus('请先点击“对比”生成 Diff。');
      return;
    }
    this._vizSetStatus('正在保存 Diff MAT...');
    try {
      const res = await fetch('/api/viz/diff-mat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({}),
      });
      if (!res.ok) {
        let detail = '';
        try { detail = (await res.json())?.detail || ''; } catch (_) {}
        throw new Error(detail || '保存 Diff 失败');
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = this._vizFilenameFromDisposition(res.headers.get('Content-Disposition')) || 'Diff.mat';
      a.click();
      URL.revokeObjectURL(url);
      this._vizSetStatus(`Diff MAT 已下载：${a.download}`);
    } catch (err) {
      this._vizSetStatus(`保存 Diff 失败：${err.message}`);
    }
  },

  _vizFilenameFromDisposition(disposition) {
    const text = String(disposition || '');
    const utf = text.match(/filename\\*=UTF-8''([^;]+)/i);
    if (utf) {
      try { return decodeURIComponent(utf[1]); } catch (_) {}
    }
    const plain = text.match(/filename=\"?([^\";]+)\"?/i);
    return plain ? plain[1] : '';
  },

  _vizShowBoxplot(show) {
    const modal = document.getElementById('viz-boxplot-modal');
    if (!modal) return;
    modal.style.display = show ? 'flex' : 'none';
  },

  _vizHideBoxplot() {
    this._vizShowBoxplot(false);
  },

  async _vizDownloadBoxplotScreenshot() {
    const svg = document.getElementById('viz-boxplot-svg');
    const titleEl = document.querySelector('#viz-boxplot-modal .viz-modal-title');
    const subtitleEl = document.getElementById('viz-boxplot-subtitle');
    const statCards = [...document.querySelectorAll('#viz-boxplot-stats .viz-stat-card')];
    if (!svg) {
      this._vizSetStatus('箱线图尚未生成');
      return;
    }

    const width = 1200;
    const svgHeight = 455;
    const cardW = 210;
    const cardH = 78;
    const gap = 14;
    const cardsPerRow = 5;
    const rows = Math.max(1, Math.ceil(statCards.length / cardsPerRow));
    const height = 112 + svgHeight + 24 + rows * cardH + Math.max(0, rows - 1) * gap + 44;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#0b1121';
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '700 28px Inter, Segoe UI, sans-serif';
    ctx.fillText((titleEl?.textContent || '差值箱线图').trim(), 42, 50);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '16px Inter, Segoe UI, sans-serif';
    this._vizDrawWrappedText(ctx, (subtitleEl?.textContent || '').trim(), 42, 80, width - 84, 22, 2);

    const svgText = new XMLSerializer().serializeToString(svg);
    const svgBlob = new Blob([svgText], {type: 'image/svg+xml;charset=utf-8'});
    const svgUrl = URL.createObjectURL(svgBlob);
    try {
      const img = await new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = svgUrl;
      });
      ctx.drawImage(img, 42, 112, width - 84, svgHeight);
    } finally {
      URL.revokeObjectURL(svgUrl);
    }

    const startY = 112 + svgHeight + 24;
    statCards.forEach((card, index) => {
      const col = index % cardsPerRow;
      const row = Math.floor(index / cardsPerRow);
      const x = 42 + col * (cardW + gap);
      const y = startY + row * (cardH + gap);
      const label = card.querySelector('.viz-stat-label')?.textContent?.trim() || '';
      const value = card.querySelector('.viz-stat-value')?.textContent?.trim() || '';
      ctx.fillStyle = '#111827';
      this._vizRoundRect(ctx, x, y, cardW, cardH, 10);
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.10)';
      ctx.stroke();
      ctx.fillStyle = '#64748b';
      ctx.font = '12px Inter, Segoe UI, sans-serif';
      ctx.fillText(label, x + 14, y + 26);
      ctx.fillStyle = '#e2e8f0';
      ctx.font = '700 18px Inter, Segoe UI, sans-serif';
      ctx.fillText(value, x + 14, y + 54);
    });

    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = `viz_boxplot_${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
    link.click();
    this._vizSetStatus('箱线图截图已生成');
  },

  _vizDrawWrappedText(ctx, text, x, y, maxWidth, lineHeight, maxLines = 2) {
    if (!text) return;
    const words = text.split(/\s+/);
    let line = '';
    let lines = 0;
    for (const word of words) {
      const test = line ? `${line} ${word}` : word;
      if (ctx.measureText(test).width > maxWidth && line) {
        ctx.fillText(line, x, y + lines * lineHeight);
        line = word;
        lines += 1;
        if (lines >= maxLines) return;
      } else {
        line = test;
      }
    }
    if (line && lines < maxLines) ctx.fillText(line, x, y + lines * lineHeight);
  },

  _vizRoundRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  },

  _vizRenderBoxplot(payload) {
    const svg = document.getElementById('viz-boxplot-svg');
    const subtitle = document.getElementById('viz-boxplot-subtitle');
    const statsEl = document.getElementById('viz-boxplot-stats');
    if (!svg || !subtitle || !statsEl) return;
    const stats = payload.stats || {};
    const width = 820;
    const height = 310;
    const margin = { left: 72, right: 44, top: 34, bottom: 86 };
    const axisY = 224;
    const boxTop = 82;
    const boxBottom = 150;
    const yMid = (boxTop + boxBottom) / 2;
    let domainMin = Math.min(stats.min, stats.whisker_low, stats.q1, stats.median, stats.q3, stats.whisker_high, stats.max);
    let domainMax = Math.max(stats.min, stats.whisker_low, stats.q1, stats.median, stats.q3, stats.whisker_high, stats.max);
    if (!(domainMax > domainMin)) {
      const pad = Math.max(Math.abs(domainMax) * 0.05, 1.0);
      domainMin -= pad;
      domainMax += pad;
    }
    const innerWidth = width - margin.left - margin.right;
    const scale = value => margin.left + ((value - domainMin) / (domainMax - domainMin)) * innerWidth;
    const ticks = [stats.min, stats.q1, stats.median, stats.q3, stats.max]
      .map((value, index) => ({
        value,
        index,
        label: this._vizFormatNumber(value),
        x: scale(value),
        lane: 0,
      }))
      .sort((a, b) => a.x - b.x || a.index - b.index);
    const laneRight = [-Infinity, -Infinity, -Infinity, -Infinity];
    ticks.forEach(tick => {
      const labelWidth = Math.max(42, tick.label.length * 7);
      let lane = laneRight.findIndex(right => tick.x - labelWidth / 2 > right + 8);
      if (lane < 0) lane = laneRight.indexOf(Math.min(...laneRight));
      tick.lane = lane;
      laneRight[lane] = tick.x + labelWidth / 2;
    });
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    subtitle.textContent = `${payload.label} · A 场景：${payload.selection_a?.scalar_label || '-'} · B 场景：${payload.selection_b?.scalar_label || '-'}`;
    svg.innerHTML = `
      <rect x="0" y="0" width="${width}" height="${height}" rx="14" fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.08)" />
      <line x1="${margin.left}" y1="${axisY}" x2="${width - margin.right}" y2="${axisY}" stroke="rgba(255,255,255,0.18)" stroke-width="2" />
      <line x1="${scale(stats.whisker_low)}" y1="${yMid}" x2="${scale(stats.whisker_high)}" y2="${yMid}" stroke="rgba(191,219,254,0.95)" stroke-width="4" stroke-linecap="round" />
      <line x1="${scale(stats.whisker_low)}" y1="${boxTop + 10}" x2="${scale(stats.whisker_low)}" y2="${boxBottom - 10}" stroke="rgba(191,219,254,0.95)" stroke-width="3" />
      <line x1="${scale(stats.whisker_high)}" y1="${boxTop + 10}" x2="${scale(stats.whisker_high)}" y2="${boxBottom - 10}" stroke="rgba(191,219,254,0.95)" stroke-width="3" />
      <rect x="${scale(stats.q1)}" y="${boxTop}" width="${Math.max(2, scale(stats.q3) - scale(stats.q1))}" height="${boxBottom - boxTop}" rx="10" fill="rgba(59,130,246,0.22)" stroke="rgba(96,165,250,0.95)" stroke-width="2.5" />
      <line x1="${scale(stats.median)}" y1="${boxTop}" x2="${scale(stats.median)}" y2="${boxBottom}" stroke="rgba(248,250,252,0.95)" stroke-width="3" />
      ${ticks.map(tick => `
        <line x1="${tick.x}" y1="${axisY}" x2="${tick.x}" y2="${axisY + 8 + tick.lane * 12}" stroke="rgba(255,255,255,0.2)" stroke-width="2" />
        <text x="${tick.x}" y="${axisY + 28 + tick.lane * 17}" fill="rgba(226,232,240,0.92)" font-size="12" text-anchor="middle">${tick.label}</text>
      `).join('')}
      <text x="${margin.left}" y="${margin.top}" fill="rgba(255,255,255,0.65)" font-size="12">离群点: ${stats.outlier_count}</text>
      <text x="${width - margin.right}" y="${margin.top}" fill="rgba(255,255,255,0.65)" font-size="12" text-anchor="end">n = ${stats.count}</text>
    `;
    const items = [
      ['样本数', stats.count],
      ['均值', this._vizFormatNumber(stats.mean)],
      ['标准差', this._vizFormatNumber(stats.std)],
      ['Q1', this._vizFormatNumber(stats.q1)],
      ['中位数', this._vizFormatNumber(stats.median)],
      ['Q3', this._vizFormatNumber(stats.q3)],
      ['下须', this._vizFormatNumber(stats.whisker_low)],
      ['上须', this._vizFormatNumber(stats.whisker_high)],
      ['最小值', this._vizFormatNumber(stats.min)],
      ['最大值', this._vizFormatNumber(stats.max)],
    ];
    statsEl.innerHTML = items.map(([label, value]) => `
      <div class="viz-stat-card">
        <div class="viz-stat-label">${UI.escapeHtml(String(label))}</div>
        <div class="viz-stat-value">${UI.escapeHtml(String(value))}</div>
      </div>
    `).join('');
  },

  _vizFormatNumber(value) {
    if (!Number.isFinite(value)) return '-';
    const abs = Math.abs(value);
    if ((abs >= 1e4) || (abs > 0 && abs < 1e-3)) return value.toExponential(3);
    if (abs >= 100) return value.toFixed(2);
    if (abs >= 1) return value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
    return value.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
  },

  _vizFrameLabel(info, frameOverride = null) {
    const nFrames = Math.max(1, Number(info?.n_frames || 0));
    const activeFrame = Math.max(0, Math.min(
      Number(frameOverride ?? (info?.active_frame || 0)),
      nFrames - 1,
    ));
    const labels = Array.isArray(info?.frame_labels) ? info.frame_labels : [];
    if (labels.length > activeFrame && labels[activeFrame]) {
      return labels[activeFrame];
    }
    const detail = this._vizFrameDetail(info, activeFrame);
    if (detail?.full) return String(detail.full);
    if (this._vizIsShakedownInfo(info)) {
      return `Load frame ${activeFrame + 1}/${nFrames}`;
    }
    const frameText = `${activeFrame + 1}/${nFrames}`;
    if (nFrames <= 1) return frameText;
    const times = Array.isArray(info?.frame_times) ? info.frame_times : [];
    if (!times.length) return frameText;
    const currentTime = Number(times[Math.min(activeFrame, times.length - 1)]);
    const totalTime = Number(times[times.length - 1]);
    if (!Number.isFinite(currentTime) || !Number.isFinite(totalTime)) return frameText;
    return `${frameText} time=${this._vizFormatTimeValue(currentTime)}/${this._vizFormatTimeValue(totalTime)}`;
  },

  _vizIsShakedownInfo(info) {
    const active = String(info?.active_field || '');
    if (active.startsWith('shakedown_') || active.startsWith('rsdms_')) return true;
    const fields = Array.isArray(info?.fields) ? info.fields : [];
    return fields.some(key => {
      const text = String(key);
      return text.startsWith('shakedown_') || text.startsWith('rsdms_');
    });
  },

  _vizFrameDetail(info, frameOverride = null) {
    const details = Array.isArray(info?.frame_details) ? info.frame_details : [];
    if (!details.length) return null;
    const nFrames = Math.max(1, Number(info?.n_frames || details.length));
    const activeFrame = Math.max(0, Math.min(
      Number(frameOverride ?? (info?.active_frame || 0)),
      nFrames - 1,
    ));
    const detail = details[activeFrame];
    return detail && typeof detail === 'object' ? detail : null;
  },

  _vizFrameFullLabel(info, frameOverride = null) {
    const detail = this._vizFrameDetail(info, frameOverride);
    if (detail?.full) return String(detail.full);
    const labels = Array.isArray(info?.frame_labels) ? info.frame_labels : [];
    const nFrames = Math.max(1, Number(info?.n_frames || labels.length));
    const activeFrame = Math.max(0, Math.min(
      Number(frameOverride ?? (info?.active_frame || 0)),
      nFrames - 1,
    ));
    return labels[activeFrame] || this._vizFrameLabel(info, activeFrame);
  },

  _vizUpdateFrameButtons(slot, info) {
    const nFrames = Math.max(1, Number(info?.n_frames || 0));
    const enabled = nFrames > 1;
    ['prev', 'next', 'play'].forEach(kind => {
      const btn = document.getElementById(`viz-frame-${kind}-${slot}`);
      if (btn) btn.disabled = !enabled;
    });
    if (!enabled) this._vizStopPlayback(slot);
  },

  _vizUpdateFrameDetailButton(slot, info) {
    const btn = document.getElementById(`viz-frame-detail-${slot}`);
    if (!btn || btn.tagName !== 'BUTTON') return;  // span placeholder
    const full = this._vizFrameFullLabel(info);
    const enabled = this._vizIsShakedownInfo(info) && !!full;
    btn.style.display = enabled ? '' : 'none';
    btn.disabled = !enabled;
    btn.title = enabled ? full : '';
  },

  _vizUpdateFrameAxisLabel(slot, info) {
    const slider = document.getElementById(`viz-frame-slider-${slot}`);
    const group = slider?.closest('.viz-group');
    const label = group?.querySelector('.viz-group-label');
    if (label) label.textContent = this._vizIsShakedownInfo(info) ? '载荷帧' : '帧';
  },

  _vizFormatTimeValue(value) {
    if (!Number.isFinite(value)) return '-';
    const abs = Math.abs(value);
    if ((abs >= 1e4) || (abs > 0 && abs < 1e-4)) return value.toExponential(3);
    return value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
  },

  async _vizDownloadScreenshot(slot) {
    this._vizSetStatus('正在生成截图...');
    try {
      const res = await fetch('/api/viz/screenshot', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({slot, scale: 4, dpi: 300}),
      });
      if (!res.ok) throw new Error('截图导出失败');
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `viz_${slot}_screenshot.png`;
      a.click();
      URL.revokeObjectURL(url);
      this._vizSetStatus('截图已下载');
    } catch (err) {
      this._vizSetStatus(`截图导出失败：${err.message}`);
    }
  },

};

window.App = App;

// Boot
document.addEventListener('DOMContentLoaded', () => App.init());
