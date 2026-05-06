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
    'direct-methods': ['direct_methods'],
    'validation':     ['validation'],
  },

  // ──────────── Initialization ────────────
  async init() {
    this.bindNav();
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
        '根据 INP 进行线弹性增量分析，并生成主 .mat 文件', 'blue', '开始分析') +
      UI.moduleCard('validation', '🔍', 'ODB 验证',
        '🔒 完整版功能：elastic / nonlinear ODB 对标验证', 'amber', '完整版') +
      UI.moduleCard('shakedown', '🛡️', '安定分析',
        '基于弹性 .mat 结果进行结构安定性 SOCP 优化分析（C formulation / CVXPY）', 'purple', '开始分析') +
      UI.moduleCard('direct-methods', '∿', 'Direct Methods',
        '🔒 完整版功能：DCA 周期求解及验证', 'green', '完整版');

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
        '<div class="task-list">' + tasks.map(t => UI.taskItem(t)).join('') + '</div></div>';
    }

    // Load slow status data in background (WSL + JAX detection)
    this.api('/api/status').then(status => {
      this.envStatus = status;
      if (this.currentPage !== 'dashboard') return;  // User navigated away
      const grid = document.getElementById('status-grid');
      if (!grid || !status) return;
      const wslOk = status.wsl?.available;
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
      this.updateSidebarStatus(wslOk);
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
    // Show confirmation modal before navigating
    UI.confirm(
      `已选取模型：<strong>${UI.escapeHtml(name)}</strong><br>点击确定后选择分析模块。`,
      () => this.navigate('elastic')
    );

  },

  _syncElasticDraftValues() {
    const draft = this._elastic?.cfgValues || {};
    const fieldMap = {
      material_model: 'elastic-material-model',
      E_override: 'elastic-E',
      nu_override: 'elastic-nu',
      yield_stress_override: 'elastic-yield',
      gauss_order: 'elastic-gauss-order',
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
    if (!this._isElasticBlankValue(analysisInput.gauss_order)) {
      controls.push({ label: 'Gauss order', value: this._formatElasticValue(analysisInput.gauss_order) });
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
      gauss_order: draft.gauss_order ?? '',
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
      gauss_order: '',
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
    const supportsPlastic = Boolean(primary.supports_elastoplastic);
    let materialModel = String(draft.material_model || primary.suggested_material_model || 'linear_elastic').trim();
    if (!supportsPlastic) materialModel = 'linear_elastic';

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
    const gaussOrderValue = resolveDraft('gauss_order', '');
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
        <div class="form-hint" style="margin-top:4px">适用于实体单元 C3D8 的体积锁定缓解。</div>
      </div>`;
    }

    html += `<div style="font-size:12px;color:var(--text-secondary);line-height:1.7">
      INP Step 信息：procedure = ${UI.escapeHtml(String(step.procedure || '-'))}，
      initial_increment = ${UI.escapeHtml(String(step.initial_increment ?? '-'))}，
      total_time = ${UI.escapeHtml(String(step.total_time ?? '-'))}。
    </div>
        <div class="form-group" style="margin:0">
          <label class="form-label">Increment count</label>
          <input class="form-input" type="number" id="elastic-n-increments" min="1" step="1" value="${UI.escapeAttr(String(nIncrementsValue))}" readonly style="opacity:0.72;cursor:not-allowed" />
          <div class="form-hint" style="margin-top:4px">Demo 版仅支持线弹性单步分析，增量数固定为 INP 默认值。</div>
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
      container.innerHTML = UI.hint('warning', '请先在模型浏览中选择一个工作模型。', '前往模型浏览', "App.navigate('models')");
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
    const gaussOrder = document.getElementById('elastic-gauss-order')?.value?.trim() || '';
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
      gaussOrder,
      nIncrements,
      maxIterations,
      convergenceTol,
    };
  },

  async saveAndRunElastic() {
    const payload = this._collectElasticSubmission();
    if (!payload) return;

    const { model, selectedInps, materialModel, useBExt, eValue, nuValue, yieldValue, gaussOrder, nIncrements, maxIterations, convergenceTol } = payload;

    const modelRoot = this.winToWsl(model.path);

    await this.apiPost('/api/models/prepare-inc-analysis', {
      model_path: model.path,
      inp_paths: selectedInps.map(f => `${model.path}\\abaqus\\${f}`),
      analysis_input: {
        material_model: materialModel,
        E_override: eValue,
        nu_override: nuValue,
        yield_stress_override: materialModel === 'j2_perfect_plastic' ? yieldValue : '',
        use_b_ext: useBExt,
        gauss_order: gaussOrder,
        n_increments: nIncrements,
        max_iterations: maxIterations,
        convergence_tol: convergenceTol,
      },
      source: 'web_inc_analysis',
    });

    let cfgLines = [
      '# inc_analysis configuration (auto-generated by jaxmech web incremental form)',
      '',
    ];
    if (selectedInps.length < model.abaqus.inp_files.length) {
      // Only write inp_files if it's a subset
      const fullPaths = selectedInps.map(f => `    "${modelRoot}/abaqus/${f}"`);
      cfgLines.push('inp_files = [');
      cfgLines.push(fullPaths.join(',\n'));
      cfgLines.push(']');
      cfgLines.push('');
    }
    if (useBExt) cfgLines.push('use_b_ext = 1');
    if (gaussOrder) cfgLines.push(`gauss_order = ${gaussOrder}`);
    cfgLines.push(`n_increments = ${nIncrements}`);
    if (maxIterations) cfgLines.push(`max_iterations = ${maxIterations}`);
    if (convergenceTol) cfgLines.push(`convergence_tol = ${convergenceTol}`);
    cfgLines.push(`material_model = ${materialModel}`);
    if (eValue) cfgLines.push(`E_override = ${eValue}`);
    if (nuValue) cfgLines.push(`nu_override = ${nuValue}`);
    if (materialModel === 'j2_perfect_plastic' && yieldValue) cfgLines.push(`yield_stress_override = ${yieldValue}`);
    cfgLines.push('');
    const content = cfgLines.join('\n');

    await this.apiPut(`/api/config/inc_analysis?model_path=${encodeURIComponent(model.path)}`, { content });

    // Also pass the config path as WSL path so run_in_wsl.ps1 receives a proper /mnt/... path
    const configPath = this.winToWsl(model.path) + '/inc_analysis';
    const result = await this.apiPost('/api/run', {
      module: 'jaxmech.tools.build_model_mat',
      args: ['--config', configPath],
    });
    this.viewTask(result.id);
  },


  // ──────────── Direct Methods ────────────
  _selectedDcaInpName() {
    return document.querySelector('input[name="dca-inp-radio"]:checked')?.value?.trim() || '';
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

  _buildDcaDraftFromParsed(primary) {
    if (!primary) return {};
    const material = primary.materials?.[0] || {};
    const elastic = material.elastic || {};
    const plastic = material.plastic || {};
    return {
      n_harmonics: primary.n_harmonics ?? '',
      convergence_tol: '1.0e-7',
      cycle_tolerance: '1.0e-7',
      use_b_ext: '1',
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
      n_harmonics: document.getElementById('dca-n-harmonics')?.value ?? draft.n_harmonics ?? '',
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

    return {
      n_harmonics: this._isElasticBlankValue(draft.n_harmonics) ? (inpMeta.n_harmonics ?? '') : draft.n_harmonics,
      convergence_tol: this._isElasticBlankValue(draft.convergence_tol) ? '1.0e-7' : draft.convergence_tol,
      cycle_tolerance: this._isElasticBlankValue(draft.cycle_tolerance) ? '1.0e-7' : draft.cycle_tolerance,
      use_b_ext: draft.use_b_ext ?? '1',
      E: this._isElasticBlankValue(draft.E) ? (elastic.E ?? '') : draft.E,
      nu: this._isElasticBlankValue(draft.nu) ? (elastic.nu ?? '') : draft.nu,
      yield_stress: this._isElasticBlankValue(draft.yield_stress) ? (plastic.yield_stress ?? '') : draft.yield_stress,
      parsed_defaults: {
        n_harmonics: inpMeta.n_harmonics ?? null,
        material_E: elastic.E ?? null,
        material_nu: elastic.nu ?? null,
        yield_stress: plastic.yield_stress ?? null,
      },
    };
  },

  _buildDcaDiffItems(analysisInput, inpMeta) {
    if (!analysisInput || !inpMeta) {
      return { diffs: [], controls: [] };
    }

    const parsedDefaults = analysisInput.parsed_defaults || {};
    const diffs = [];
    const pushDiff = (label, current, original) => {
      if (this._elasticValuesEqual(current, original)) return;
      diffs.push({
        label,
        current: this._formatElasticValue(current),
        original: this._formatElasticValue(original),
      });
    };

    pushDiff('Harmonics', analysisInput.n_harmonics, parsedDefaults.n_harmonics ?? inpMeta.n_harmonics);
    pushDiff("Young's modulus (E)", analysisInput.E, parsedDefaults.material_E);
    pushDiff('Poisson ratio (nu)', analysisInput.nu, parsedDefaults.material_nu);
    pushDiff('Yield stress', analysisInput.yield_stress, parsedDefaults.yield_stress);

    return {
      diffs,
      controls: [
        { label: 'Convergence tolerance', value: this._formatElasticValue(analysisInput.convergence_tol) },
        { label: 'Cycle tolerance', value: this._formatElasticValue(analysisInput.cycle_tolerance) },
        { label: 'B-bar 扩展', value: String(analysisInput.use_b_ext ?? '1') !== '0' ? '启用' : '关闭' },
      ],
    };
  },

  _onDcaDraftChange() {
    this._syncDcaDraftValues();
    this._renderDcaCurrentDiffPreview();
  },

  _onDcaInpSelectionChange() {
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
    target.innerHTML = `<div style="font-size:13px;color:var(--text-secondary);margin-top:10px;line-height:1.7">当前提醒只基于这次 fresh parse 得到的临时输入快照；不会拿已有 DCA MAT 或旧 cfg 反推差异。</div>
      <div style="margin-top:12px">${this._renderElasticDiffSummary(summary, {
        currentLabel: '当前表单',
        originalLabel: '解析快照',
        emptyMessage: '当前表单与 fresh parse 快照一致；如果现在提交，这次 DCA 会直接使用解析默认值。',
        controlsCaption: '附加求解控制（非 INP 原始字段）',
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
    const nHarmonics = this._isElasticBlankValue(vals.n_harmonics) ? (meta.n_harmonics ?? '') : vals.n_harmonics;
    const convergenceTol = this._isElasticBlankValue(vals.convergence_tol) ? '1.0e-7' : vals.convergence_tol;
    const cycleTolerance = this._isElasticBlankValue(vals.cycle_tolerance) ? '1.0e-7' : vals.cycle_tolerance;
    const useBExt = String(vals.use_b_ext ?? '1') !== '0';
    const eValue = this._isElasticBlankValue(vals.E) ? (elastic.E ?? '') : vals.E;
    const nuValue = this._isElasticBlankValue(vals.nu) ? (elastic.nu ?? '') : vals.nu;
    const yieldValue = this._isElasticBlankValue(vals.yield_stress) ? (plastic.yield_stress ?? '') : vals.yield_stress;
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
        <div class="form-group" style="margin:0">
          <label class="form-label">Harmonics</label>
          <input class="form-input" type="number" min="0" step="1" id="dca-n-harmonics" value="${UI.escapeAttr(String(nHarmonics))}" oninput="App._onDcaDraftChange()" onchange="App._onDcaDraftChange()" />
          <div class="form-hint" style="margin-top:4px">控制周期位移与残差在频域中保留的谐波阶数；默认直接沿用 INP 中 DCA step 解析出的设置。</div>
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

        <div class="form-group" style="margin:0">
          <label class="form-label">B-bar 扩展</label>
          <div style="display:flex;align-items:center;gap:10px;margin-top:6px">
            <input type="checkbox" id="dca-use-b-ext" ${useBExt ? 'checked' : ''} onchange="App._onDcaDraftChange()" style="width:16px;height:16px;accent-color:var(--accent);cursor:pointer">
            <label for="dca-use-b-ext" style="font-size:14px;cursor:pointer">对 C3D8 启用 B-bar 扩展</label>
          </div>
          <div class="form-hint" style="margin-top:4px">用于缓解实体单元的体积锁定；当前 DCA Web 求解只支持 solid，因此这个开关保留在表单里。</div>
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
      <button class="btn btn-primary" onclick="App.saveAndRunDirectCyclic()">▶ 基于当前解析结果运行 DCA</button>
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

    const modelRoot = this.winToWsl(model.path);
    const inpPath = `${modelRoot}/abaqus/${inpName}`;
    const abaqusReferenceMat = meta.abaqus_reference_mat ? this.winToWsl(meta.abaqus_reference_mat) : '';
    const stepName = meta.step_name;
    const amplitudeName = meta.amplitude_name;
    const nHarmonics = document.getElementById('dca-n-harmonics')?.value?.trim() || String(meta.n_harmonics || '15');
    const nTimePoints = String(meta.n_time_points || '33');
    const periodicStart = '2';
    const maxIterations = meta.max_iterations > 0 ? String(meta.max_iterations) : '100';
    const convergenceTol = document.getElementById('dca-convergence-tol')?.value?.trim() || '1.0e-7';
    const cycleTolerance = document.getElementById('dca-cycle-tolerance')?.value?.trim() || '1.0e-7';
    const constitutiveBackend = 'j2_perfect_plastic';
    const gaussOrder = '2';
    const useBExt = document.getElementById('dca-use-b-ext')?.checked ? '1' : '0';
    const outMatName = `${inpName.replace(/\.inp$/i, '')}_jax_dca.mat`;

    const constitutivePairs = [];
    const eOverride = document.getElementById('dca-E')?.value?.trim();
    const nuOverride = document.getElementById('dca-nu')?.value?.trim();
    const yieldOverride = document.getElementById('dca-yield-stress')?.value?.trim();
    if (eOverride) constitutivePairs.push(`'E_override': ${eOverride}`);
    if (nuOverride) constitutivePairs.push(`'nu_override': ${nuOverride}`);
    if (yieldOverride) constitutivePairs.push(`'yield_stress_override': ${yieldOverride}`);
    const constitutiveOptions = constitutivePairs.length ? `{${constitutivePairs.join(', ')}}` : '{}';

    const lines = [
      '# dca configuration (auto-generated by jaxmech web direct-methods form)',
      '',
      `inp_file = "${inpPath}"`,
    ];
    if (abaqusReferenceMat) lines.push(`abaqus_reference_mat = "${abaqusReferenceMat}"`);
    lines.push(`step_name = "${stepName}"`);
    lines.push(`amplitude_name = "${amplitudeName}"`);
    lines.push(`n_harmonics = ${nHarmonics}`);
    lines.push(`n_time_points = ${nTimePoints}`);
    lines.push(`periodic_start_iteration = ${periodicStart}`);
    lines.push(`max_iterations = ${maxIterations}`);
    lines.push(`convergence_tol = ${convergenceTol}`);
    lines.push(`cycle_tolerance = ${cycleTolerance}`);
    lines.push('relaxation_factor = 1.0');
    lines.push('');
    lines.push(`constitutive_backend = "${constitutiveBackend}"`);
    lines.push(`constitutive_options = ${constitutiveOptions}`);
    lines.push(`use_b_ext = ${useBExt}`);
    lines.push(`gauss_order = ${gaussOrder}`);
    lines.push('solver_options = {"scipy": {}}');
    lines.push(`out_mat_name = "${outMatName}"`);
    lines.push('');

    await this.apiPut(`/api/config/dca?model_path=${encodeURIComponent(model.path)}`, {
      content: lines.join('\n'),
    });

    const result = await this.apiPost('/api/run', {
      module: 'jaxmech.modules.direct_methods.dca.run',
      args: ['--config', `${modelRoot}/dca`],
    });
    this.viewTask(result.id);
  },

  async loadDirectMethods() {
    const container = document.getElementById('direct-methods-content');
    if (!this.selectedModel) {
      container.innerHTML = UI.hint('warning', '请先在模型浏览中选择一个工作模型。', '前往模型浏览', "App.navigate('models')");
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

    const cfgRes = await this.api(`/api/config/dca?model_path=${encodeURIComponent(model.path)}`).catch(() => ({ values: {} }));
    const vals = cfgRes?.values || {};

    this._directMethods = {
      model,
      parsedInpMetaByName: {},
      parsedSelectionKey: '',
      lastParsedAt: '',
      cfgValues: {},
    };

    const cfgInpName = String(vals.inp_file || '').trim().split(/[\\/]/).pop() || model.abaqus.inp_files[0];

    let html = `<div style="font-size:13px;color:var(--text-secondary);margin-bottom:12px;padding:10px 14px;background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid rgba(255,255,255,0.06)">
      模型: <b style="color:var(--text-primary)">${UI.escapeHtml(model.name)}</b>
      <span style="color:rgba(255,255,255,0.25);margin:0 8px">|</span>
      <code style="font-size:11px;color:var(--accent)">${UI.escapeHtml(model.path)}</code>
    </div>`;

    html += `<div class="card" style="margin-bottom:16px">
      <div class="card-title">DCA workflow — INP 选择与解析</div>
      <div style="font-size:13px;color:var(--text-secondary);margin:10px 0 14px">当前 GUI 只面向“一个 INP 对应一个 DCA workflow”。和 inc_analysis 一样，先选 INP、再解析、再只改少量 override。</div>
      <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px">
        <button class="btn btn-secondary" onclick="App.parseSelectedDcaInp()">解析所选 INP</button>
        <span id="dca-parse-status" style="font-size:12px;color:var(--text-secondary)">当前所选 INP 尚未解析。</span>
      </div>
      <div id="dca-inp-list">`;
    model.abaqus.inp_files.forEach((f, i) => {
      const checked = cfgInpName === f ? 'checked' : '';
      html += `<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06)">
        <input type="radio" name="dca-inp-radio" id="dca-inp-${i}" value="${UI.escapeAttr(f)}" ${checked} onchange="App._onDcaInpSelectionChange()" style="width:16px;height:16px;flex-shrink:0;accent-color:var(--accent);cursor:pointer">
        <label for="dca-inp-${i}" style="font-size:14px;cursor:pointer;user-select:none">📄 ${UI.escapeHtml(f)}</label>
      </div>`;
    });
    html += `</div></div>`;

    html += `<div id="direct-methods-config-area"></div>`;
    container.innerHTML = html;

    this._renderDcaParseStatus();
    this._renderDirectMethodsConfigForm();
  },

  async _browseDcaValidationReferenceMat() {
    const model = this._validation?.model || this._directMethods?.model;
    const initialDir = model ? `${model.path}\\abaqus` : 'C:\\';
    const res = await this.apiPost('/api/browse', {
      field_type: 'mat_file',
      title: '选择 Abaqus DCA 参考 MAT',
      initial_dir: initialDir,
    }).catch(() => null);
    if (res?.path) {
      const input = document.getElementById('dca-val-abaqus-mat');
      if (input) input.value = res.path;
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
    let abaqusMat = String(directInfo.abaqus_reference_mat || '').trim();
    if (abaqusMat.startsWith('/mnt/')) abaqusMat = this.wslToWin(abaqusMat);
    if (!abaqusMat && directInfo.source_inp) {
      const srcInp = String(directInfo.source_inp).trim();
      abaqusMat = srcInp.replace(/\.inp$/i, '.mat');
      if (abaqusMat.startsWith('/mnt/')) abaqusMat = this.wslToWin(abaqusMat);
    }
    if (!abaqusMat && model) {
      const stem = String(matMeta?.name || '').replace(/_jax_dca\.mat$/i, '').replace(/\.mat$/i, '');
      abaqusMat = `${model.path}\\abaqus\\${stem}.mat`;
    }
    const amplitudeName = String(directInfo.amplitude_name || '').trim();
    const validationDir = model ? `${model.path}\\validation\\direct_methods\\${String(matMeta?.name || 'dca_result').replace(/\.mat$/i, '')}` : '';

    formArea.innerHTML = `<div class="card">
      <div class="card-title">DCA MAT validation 配置</div>
      <div style="margin-top:10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap"><span style="display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;background:rgba(52,211,153,0.14);color:#34d399;font-size:12px;font-weight:600">DCA / MAT compare</span></div>
      <div style="margin-top:12px;display:flex;flex-direction:column;gap:14px">
        <div class="form-group">
          <label class="form-label">JAX DCA MAT</label>
          <input class="form-input" type="text" id="dca-val-mat" value="${UI.escapeAttr(matAbsPath)}" readonly style="opacity:0.7;cursor:not-allowed;font-family:monospace;font-size:12px" />
        </div>
        <div class="form-group">
          <label class="form-label">ABAQUS 参考 MAT</label>
          <div style="display:flex;gap:8px;align-items:center">
            <input class="form-input" type="text" id="dca-val-abaqus-mat" value="${UI.escapeAttr(abaqusMat)}" style="flex:1;font-family:monospace;font-size:12px" />
            <button class="btn btn-secondary" style="flex-shrink:0;padding:8px 14px" onclick="App._browseDcaValidationReferenceMat()">浏览...</button>
          </div>
          <div class="form-hint" style="margin-top:4px">优先使用 DCA MAT 自身记录的 <code>abaqus_reference_mat</code>；若为空，再回退到与 source INP 同名的 <code>abaqus/*.mat</code>。</div>
        </div>
        <div style="font-size:12px;color:var(--text-secondary);line-height:1.7">这条链直接比较 JAX DCA MAT 与 Abaqus DCA MAT，不读取 ODB；summary、steady boxplot 和逐变量 frame boxplot 的组织方式继续沿用 nonlinear validation 的输出风格。</div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px">
          <div class="form-group" style="margin:0">
            <label class="form-label">amplitude_name</label>
            <input class="form-input" type="text" id="dca-val-amplitude-name" value="${UI.escapeAttr(amplitudeName)}" />
          </div>
          <div class="form-group" style="margin:0">
            <label class="form-label">validation 输出目录</label>
            <input class="form-input" type="text" id="dca-val-output-dir" value="${UI.escapeAttr(validationDir)}" readonly style="opacity:0.7;cursor:not-allowed;font-family:monospace;font-size:12px" />
          </div>
        </div>
        <div style="display:flex;gap:12px">
          <button class="btn btn-primary" onclick="App.runDirectMethodValidation()">▶ 运行 DCA MAT validation</button>
        </div>
      </div>
    </div>`;
    formArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
  },

  async runDirectMethodValidation() {
    const model = this._validation?.model || this._directMethods?.model;
    const matPath = document.getElementById('dca-val-mat')?.value?.trim();
    const abaqusMat = document.getElementById('dca-val-abaqus-mat')?.value?.trim();
    const amplitudeName = document.getElementById('dca-val-amplitude-name')?.value?.trim();
    const outputDir = document.getElementById('dca-val-output-dir')?.value?.trim();
    if (!model || !matPath) {
      alert('请先选择一个 DCA MAT。');
      return;
    }
    if (!abaqusMat) {
      alert('请先选择一个 Abaqus 参考 MAT。');
      return;
    }

    const args = [
      '--mat', this.winToWsl(matPath),
      '--abaqus-mat', this.winToWsl(abaqusMat),
      '--validation-dir', this.winToWsl(outputDir),
      '--copy-mat',
    ];
    if (amplitudeName) {
      args.push('--amplitude-name', amplitudeName);
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
      container.innerHTML = UI.hint('warning', '请先在模型浏览中选择一个工作模型。', '前往模型浏览', "App.navigate('models')");
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
    const incMats = allScanMats.filter(m => m.subdir.toLowerCase().replace(/\\/g, '/') === 'inc_analysis');
    const dcaMats = allScanMats.filter(m => m.subdir.toLowerCase().replace(/\\/g, '/') === 'dca');
    const validationMats = allScanMats.filter(m => m.subdir.toLowerCase().replace(/\\/g, '/') === 'validation');
    const validationByName = new Map(validationMats.map(m => [m.name, m]));

    if (!incMats.length && !dcaMats.length) {
      container.innerHTML = UI.hint('error', '当前模型下既没有 inc_analysis MAT，也没有 dca MAT。验证模块需要已有结果文件。', '前往增量分析', "App.navigate('elastic')");
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
    }
    const validationMetaByName = new Map(validationDetails.map(d => [d.name, d]));
    let dcaMetaByName = new Map();
    if (dcaMats.length > 0) {
      const dcaMetaRes = await this.apiPost('/api/models/mat-meta', {
        paths: dcaMats.map(m => m.abs_path),
      }).catch(() => null);
      dcaMetaByName = new Map((dcaMetaRes?.details || []).map(d => [d.name, d]));
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
          状态依据：这里只检查 validation 文件夹中是否存在同名 MAT；这不是在判断 inc_analysis 结果 MAT 是否存在。若 validation 目录中的同名 MAT 可解析出 validated_with_ODB 等字段，则显示“已验证”，否则仅显示“validation 目录已有同名 MAT”。选中 MAT 后，页面会按单步 / 多步自动切换 elastic 或 nonlinear 验证表单。
        </div>
      </div>`;
    }

    if (dcaMats.length) {
      const dcaRows = dcaMats.map((m) => {
        const meta = dcaMetaByName.get(m.name) || {};
        const directInfo = meta.direct_method_info || {};
        const ref = directInfo.abaqus_reference_mat
          ? `<span style="color:#60a5fa;font-size:11px;font-weight:600;white-space:nowrap">参考 MAT 已记录</span>`
          : `<span style="color:var(--text-secondary);font-size:11px;font-weight:600;white-space:nowrap">未记录参考 MAT</span>`;
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
        <div class="card-title">DCA 结果 MAT 文件 — 点击选择进行结果 validation</div>
        <div style="margin-top:12px;display:flex;flex-direction:column;gap:8px">${dcaRows}</div>
        <div style="font-size:12px;color:var(--text-secondary);margin-top:12px">这条链不读取 ODB，而是直接把 dca 目录下的 JAX DCA MAT 与 Abaqus DCA MAT 做 compare。summary 与 boxplot 组织方式继续沿用 nonlinear validation 的风格，便于和弹塑性分析结果一起查看。</div>
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

    const isNonlinear = matMeta?.validation_branch === 'nonlinear';
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

    let module = 'jaxmech.modules.validation.elastic.run';
    let args = [
      '--mat', matWsl,
      '--odb', odbWsl,
      '--model-root', modelWsl,
      '--copy-mat',
    ];

    if (matMeta.validation_branch === 'nonlinear') {
      const alignMode = document.querySelector('input[name="val-align-mode"]:checked')?.value || 'interpolate';
      const emitPeeqPlot = document.getElementById('val-emit-peeq-plot')?.checked;
      module = 'jaxmech.modules.validation.nonlinear.run';
      args = [
        '--mat', matWsl,
        '--odb', odbWsl,
        '--model-root', modelWsl,
        '--force',
      ];
      if (alignMode === 'align_odb') {
        args.push('--align-time-steps');
      } else {
        args.push('--copy-mat');
      }
      if (emitPeeqPlot) {
        args.push('--emit-peeq-plot');
      }
    }

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
      container.innerHTML = UI.hint('warning', '请先在模型浏览中选择一个工作模型。', '前往模型浏览', "App.navigate('models')");
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
    // 检测 validation 同名 MAT 状态（非阻塞，仅警告）
    const allScanMats = Array.isArray(this._sd.allScanMats) ? this._sd.allScanMats : [];
    const validationMats = allScanMats.filter(m => m.subdir.toLowerCase().replace(/\\/g, '/') === 'validation');
    const validationByName = new Map(validationMats.map(m => [m.name, m]));
    let validationMetaByName = new Map();
    if (validationMats.length > 0) {
      const validationMetaRes = await this.apiPost('/api/models/mat-meta', {
        paths: validationMats.map(m => m.abs_path)
      }).catch(() => null);
      validationMetaByName = new Map((validationMetaRes?.details || []).map(d => [d.name, d]));
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

    // ── (1) Formulation — Demo locked to C ──
    html += `<div class="form-group">
      <label class="form-label">单元公式 (formulation)</label>
      <input class="form-input" type="text" id="sd-formulation" value="C" readonly
        style="opacity:0.55;cursor:not-allowed" />
      ${hint('Demo 版仅支持 VersionC（残余应力 ρ 变量）。完整版支持 MN 等其他公式。')}
    </div>`;


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
        <input class="form-input" type="number" id="sd-ilyushin-c" value="0.9999" min="0" max="1" step="0.0001" />
        ${hint('\\(\\varphi = \\varphi_N + \\varphi_M + 2c|\\varphi_{NM}|\\)，c=0.9999 近乎纯 Ilyushin。')}
      </div>`;
    }

    // ── (3) Solver — Demo locked to CVXPY ──
    html += `<div class="form-group">
      <label class="form-label">求解器 (solver)</label>
      <input class="form-input" type="text" id="sd-solver" value="cvxpy" readonly
        style="opacity:0.55;cursor:not-allowed" />
      ${hint('Demo 版仅支持 CVXPY（WSL 侧 Clarabel）。完整版支持 Gurobi 等其他后端。')}
    </div>`;

    // (Gurobi config section removed — demo only supports CVXPY)

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
      html += `<div id="sd-angles-section" style="display:none;flex-direction:column;gap:12px;
          padding:14px 16px;background:rgba(255,255,255,0.04);border-radius:10px;border:1px solid rgba(255,255,255,0.08)">
        <div style="font-size:12px;font-weight:600;color:var(--text-secondary)">角度列表（单位：degree °）</div>
        <div style="display:flex;gap:32px;flex-wrap:wrap;align-items:flex-start">
          ${hasPhi ? `
          <div id="sd-phi-group" class="form-group" style="margin:0;min-width:150px">
            <label class="form-label">\\(\\varphi\\)（仰角）</label>
            <div id="sd-phi-list" style="display:flex;flex-direction:column;gap:4px;margin-top:6px">
              <div class="sd-angle-item" style="display:flex;align-items:center;gap:6px">
                <span style="font-size:11px;color:var(--text-secondary);min-width:22px">1.</span>
                <input class="form-input sd-phi-item" type="number" step="any" value="0" style="width:90px" placeholder="° (deg)"/>
              </div>
            </div>
            <button class="btn btn-secondary" style="margin-top:8px;padding:4px 12px;font-size:12px"
              onclick="App._addAngle('sd-phi-list','sd-phi-item')">＋ 添加 \\(\\varphi\\)</button>
            ${hint('\\(a_1=\\cos\\varphi\\cos\\theta,\\; a_2=\\cos\\varphi\\sin\\theta,\\; a_3=\\sin\\varphi\\)。单位 degree。')}
          </div>` : ''}
          <div id="sd-theta-group" class="form-group" style="margin:0;min-width:150px">
            <label class="form-label">\\(\\theta\\)（方位角）</label>
            <div id="sd-theta-list" style="display:flex;flex-direction:column;gap:4px;margin-top:6px">
              <div class="sd-angle-item" style="display:flex;align-items:center;gap:6px">
                <span style="font-size:11px;color:var(--text-secondary);min-width:22px">1.</span>
                <input class="form-input sd-theta-item" type="number" step="any" value="0" style="width:90px" placeholder="° (deg)"/>
              </div>
            </div>
            <button class="btn btn-secondary" style="margin-top:8px;padding:4px 12px;font-size:12px"
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

    // ── (8) 执行步骤 — Demo locked to all ──
    html += `<div class="form-group">
      <label class="form-label">执行步骤</label>
      <input class="form-input" type="text" id="shakedown-step" value="all" readonly
        style="opacity:0.55;cursor:not-allowed" />
      <div class="form-hint" style="margin-top:4px">Demo 版固定为完整执行。prepare / collect 仅在 Gurobi 模式下使用（完整版）。</div>
    </div>`;

    html += `</div></div>`; // end config card

    html += `<div style="display:flex;gap:12px;margin-top:4px">
      <button class="btn btn-primary" onclick="App.saveAndRunShakedown()">&#9654; 保存配置并运行安定分析</button>
    </div>`;

    container.innerHTML = html;

    // 初始化联动
    this._onNVertChange();

    // 触发 MathJax 渲染（处理 async 加载时机）
    const _doTypeset = (el) => {
      if (window.MathJax?.typesetPromise) {
        MathJax.typesetPromise([el]).catch(e => console.warn('MathJax typeset:', e));
      } else if (window.MathJax?.startup?.promise) {
        MathJax.startup.promise.then(() => MathJax.typesetPromise([el])).catch(e => console.warn('MathJax:', e));
      }
    };
    _doTypeset(container);
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
    wrapper.style.cssText = 'display:flex;align-items:center;gap:4px';
    const lbl = document.createElement('span');
    lbl.style.cssText = 'font-size:11px;color:var(--text-secondary);min-width:18px';
    lbl.textContent = count + '.';
    const inp = document.createElement('input');
    inp.type = 'number'; inp.step = 'any'; inp.value = '0'; inp.style.width = '84px';
    inp.placeholder = '° (deg)';
    inp.className = 'form-input ' + itemClass;
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
    const ilyushinC = document.getElementById('sd-ilyushin-c')?.value || '0.9999';
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
      const gurobiUser = document.getElementById('sd-gurobi-user')?.value || '';
      const gurobiPass = document.getElementById('sd-gurobi-pass')?.value || '';
      const timeLimit = document.getElementById('sd-gurobi-timelimit')?.value || '3600';
      const feasTol = document.getElementById('sd-gurobi-feastol')?.value || '1e-6';
      const gurobiBackend = document.getElementById('sd-gurobi-backend')?.value || 'python';
      lines.push(`gurobi_backend = ${gurobiBackend}`);
      if (gurobiRoot) lines.push(`windows_gurobi_root = ${gurobiRoot}`);
      if (gurobiUser) lines.push(`windows_gurobi_user = ${gurobiUser}`);
      if (gurobiPass) lines.push(`windows_gurobi_password = ${gurobiPass}`);
      lines.push(`gurobi_params = {'TimeLimit': ${timeLimit}, 'FeasibilityTol': ${feasTol}}`);
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

  _renderTaskArtifacts(artifacts) {
    if (!Array.isArray(artifacts) || artifacts.length === 0) return '';
    const cards = artifacts.map((artifact) => {
      const label = UI.escapeHtml(artifact.label || artifact.path || 'artifact');
      const path = UI.escapeHtml(artifact.path || '');
      const url = this._taskArtifactUrl(artifact.path || '');
      if ((artifact.kind || '').toLowerCase() === 'image') {
        return `<div style="display:flex;flex-direction:column;gap:10px;padding:14px;border-radius:10px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08)">
          <div style="font-size:13px;font-weight:600">${label}</div>
          <img src="${url}" alt="${label}" style="width:100%;max-height:480px;object-fit:contain;border-radius:8px;background:rgba(15,23,42,0.35);border:1px solid rgba(255,255,255,0.06)" />
          <div style="font-size:11px;color:var(--text-secondary);font-family:monospace;word-break:break-all">${path}</div>
        </div>`;
      }
      const lowerKind = (artifact.kind || '').toLowerCase();
      const lowerPath = String(artifact.path || '').toLowerCase();
      const canVisualize = lowerKind === 'mat' || lowerPath.endsWith('.mat');
      const vizButton = canVisualize
        ? `<button class="btn btn-secondary" style="padding:6px 10px;font-size:12px" onclick="App.openArtifactVisualization('${UI.escapeAttr(artifact.path || '')}')">打开 visualization</button>`
        : '';
      return `<div style="padding:14px;border-radius:10px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08)">
        <div style="font-size:13px;font-weight:600;margin-bottom:8px">${label}</div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <a href="${url}" target="_blank" style="color:var(--accent);text-decoration:none;font-size:12px">打开文件</a>
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

  async loadTasks() {
    const container = document.getElementById('tasks-content');
    const tasks = await this.api('/api/tasks?limit=50').catch(() => []);
    this.tasks = tasks;

    if (tasks.length === 0) {
      container.innerHTML = UI.empty('📋', '暂无任务记录');
      return;
    }
    container.innerHTML = '<div class="task-list">' +
      tasks.map(t => UI.taskItem(t)).join('') + '</div>';
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
    if (!task) {
      container.innerHTML = UI.hint('error', '任务未找到');
      return;
    }

    const statusMap = {
      running: '运行中', success: '完结', failed: '失败', pending: '等待中'
    };
    const taskArtifacts = Array.isArray(task.artifacts) ? [...task.artifacts] : [];

    let html = `
      <button class="btn btn-secondary" onclick="App.loadTasks()" style="margin-bottom:16px">← 返回任务列表</button>
      <div class="card" style="margin-bottom:16px">
        <div class="card-header">
          <span class="card-title">${task.module}</span>
          <span class="task-status-badge ${task.status}">${statusMap[task.status]}</span>
        </div>
        <div style="font-size:13px;color:var(--text-secondary)">
          <div>任务 ID: ${task.id}</div>
          <div>参数: ${task.args.join(' ')}</div>
          <div>创建: ${task.created_at}</div>
          ${task.started_at ? `<div>开始: ${task.started_at}</div>` : ''}
          ${task.finished_at ? `<div>完成: ${task.finished_at}</div>` : ''}
        </div>
      </div>
    `;

    if (task.error_summary) {
      html += UI.hint('error', '错误摘要: ' + task.error_summary);
    }

    html += `<div id="task-artifacts-section">${this._renderTaskArtifacts(taskArtifacts)}</div>`;

    html += '<div class="card"><div class="card-title" style="margin-bottom:12px">运行日志</div>' +
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
      viewer.innerHTML = `<div class="doc-content">${marked.parse(doc.content)}</div>`;
    } else {
      viewer.innerHTML = `<div class="doc-content"><pre>${UI.escapeHtml(doc.content)}</pre></div>`;
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
        readOnly: true,
        hint: '🔒 Demo 版锁定。该字段仅供完整版 ODB validation 模块使用。'
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
          <input class="form-input" id="cfg-${f.key}" value="${val}" ${f.readOnly ? 'readonly style="opacity:0.55;cursor:not-allowed"' : ''} />
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
    const [status, winPyRes, wslPyRes] = await Promise.all([
      this.api('/api/status').catch(() => null),
      this.api('/api/status/detect-windows-python').catch(() => null),
      this.api('/api/status/detect-wsl-python').catch(() => null),
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

    html += UI.hint('info', 'ABAQUS / ODB validation：🔒 Demo 版未开放，设置项已锁定。');

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

    const [winPyRes, wslPyRes] = await Promise.all([
      this.api('/api/status/detect-windows-python').catch(() => null),
      this.api('/api/status/detect-wsl-python').catch(() => null),
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

    html += UI.hint('info', 'ABAQUS / ODB validation：🔒 Demo 版未开放，未参与自动检测。');

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

  _viz: {
    ws_a: null,
    ws_b: null,
    info_a: null,
    info_b: null,
    compareMode: false,
    compareLocked: false,
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
    slotState: {
      a: { modelKey: '', matPath: '', mats: [], family: '', quantityValue: '', inspect: null, selectedFields: [], sourceLocked: false },
      b: { modelKey: '', matPath: '', mats: [], family: '', quantityValue: '', inspect: null, selectedFields: [], sourceLocked: false },
    },
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
            <span class="viz-group-label">模型 / 结果 A</span>
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
          <div class="viz-group">
            <span class="viz-group-label">结果量</span>
            <select id="viz-family-a" class="viz-field-family" disabled>
              <option value="">类别</option>
            </select>
            <select id="viz-quantity-a" class="viz-field-select" disabled>
              <option value="">结果量</option>
            </select>
            <span class="viz-field-map" id="viz-field-map-a"></span>
          </div>
          <div class="viz-sep"></div>
          <div class="viz-group viz-action-group">
            <button class="viz-btn" id="viz-edges-a" title="显示或隐藏网格边线">网格</button>
            <button class="viz-btn" id="viz-overlay-a" title="显示或隐藏文件信息与坐标轴">信息</button>
            <button class="viz-btn viz-frame-step" id="viz-frame-prev-a" title="上一帧" aria-label="上一帧"><span class="viz-icon viz-icon-prev" aria-hidden="true"></span></button>
            <button class="viz-btn viz-frame-play" id="viz-frame-play-a" title="连续播放" aria-label="连续播放"><span class="viz-icon viz-icon-play" aria-hidden="true"></span></button>
            <button class="viz-btn viz-frame-step" id="viz-frame-next-a" title="下一帧" aria-label="下一帧"><span class="viz-icon viz-icon-next" aria-hidden="true"></span></button>
            <button class="viz-btn" id="viz-screenshot-a" title="导出 300 DPI PNG">截图</button>
          </div>
          <div class="viz-sep viz-row-break"></div>
          <div class="viz-group viz-frame-group">
            <span class="viz-group-label">帧</span>
            <input type="range" id="viz-frame-slider-a" min="0" max="0" value="0" disabled>
            <span class="viz-frame-display" id="viz-frame-label-a">1/1</span>
            <button class="viz-btn viz-frame-detail" id="viz-frame-detail-a" title="" disabled>详见信息</button>
          </div>
          <div class="viz-sep"></div>
          <div class="viz-group">
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
          <div class="viz-group">
            <span class="viz-group-label">图例</span>
            <input type="number" id="viz-clim-min-a" placeholder="min" step="any" style="width:70px">
            <span style="color:var(--text-muted)">~</span>
            <input type="number" id="viz-clim-max-a" placeholder="max" step="any" style="width:70px">
            <button class="viz-btn" id="viz-clim-apply-a">应用</button>
            <button class="viz-btn active" id="viz-clim-auto-a" title="锁定当前自动范围或恢复自动范围">自动</button>
          </div>
          <div class="viz-sep"></div>
          <div class="viz-group viz-compare-toolbar" id="viz-compare-toolbar">
            <span class="viz-group-label">对比</span>
            <button class="viz-btn viz-compare-btn" id="viz-compare-toggle" title="切换 A/B 双栏对比">双栏对比</button>
            <button class="viz-btn" id="viz-sync-camera" title="同步 A/B 视角和缩放大小" style="display:none">同步视角</button>
            <button class="viz-btn" id="viz-diff-run" title="计算当前 A/B 显示标量的 Diff" style="display:none">对比</button>
            <label class="viz-inline-check" id="viz-diff-absolute-wrap" style="display:none" title="勾选后计算 |B-A|，未勾选时计算 B-A">
              <input type="checkbox" id="viz-diff-absolute">
              <span>|B-A|</span>
            </label>
            <button class="viz-btn" id="viz-diff-boxplot" title="显示当前 Diff 的箱线图" style="display:none" disabled>箱线图</button>
            <button class="viz-btn" id="viz-diff-save" title="将当前 Diff 下载为 MAT 文件" style="display:none" disabled>保存Diff</button>
          </div>
        </div>

        <div class="viz-toolbar" id="viz-toolbar-b" style="display:none">
          <div class="viz-group viz-source-group">
            <span class="viz-group-label">模型 / 结果 B</span>
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
          <div class="viz-group">
            <span class="viz-group-label">结果量</span>
            <select id="viz-family-b" class="viz-field-family" disabled>
              <option value="">类别</option>
            </select>
            <select id="viz-quantity-b" class="viz-field-select" disabled>
              <option value="">结果量</option>
            </select>
            <span class="viz-field-map" id="viz-field-map-b"></span>
          </div>
          <div class="viz-sep"></div>
          <div class="viz-group viz-action-group">
            <button class="viz-btn" id="viz-edges-b" title="显示或隐藏网格边线">网格</button>
            <button class="viz-btn" id="viz-overlay-b" title="显示或隐藏文件信息与坐标轴">信息</button>
            <button class="viz-btn viz-frame-step" id="viz-frame-prev-b" title="上一帧" aria-label="上一帧"><span class="viz-icon viz-icon-prev" aria-hidden="true"></span></button>
            <button class="viz-btn viz-frame-play" id="viz-frame-play-b" title="连续播放" aria-label="连续播放"><span class="viz-icon viz-icon-play" aria-hidden="true"></span></button>
            <button class="viz-btn viz-frame-step" id="viz-frame-next-b" title="下一帧" aria-label="下一帧"><span class="viz-icon viz-icon-next" aria-hidden="true"></span></button>
            <button class="viz-btn" id="viz-screenshot-b" title="导出 300 DPI PNG">截图</button>
          </div>
          <div class="viz-sep viz-row-break"></div>
          <div class="viz-group viz-frame-group">
            <span class="viz-group-label">帧</span>
            <input type="range" id="viz-frame-slider-b" min="0" max="0" value="0" disabled>
            <span class="viz-frame-display" id="viz-frame-label-b">1/1</span>
            <button class="viz-btn viz-frame-detail" id="viz-frame-detail-b" title="" disabled>详见信息</button>
          </div>
          <div class="viz-sep"></div>
          <div class="viz-group">
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
          <div class="viz-group">
            <span class="viz-group-label">图例</span>
            <input type="number" id="viz-clim-min-b" placeholder="min" step="any" style="width:70px">
            <span style="color:var(--text-muted)">~</span>
            <input type="number" id="viz-clim-max-b" placeholder="max" step="any" style="width:70px">
            <button class="viz-btn" id="viz-clim-apply-b">应用</button>
            <button class="viz-btn active" id="viz-clim-auto-b" title="锁定当前自动范围或恢复自动范围">自动</button>
          </div>
        </div>

        <div class="viz-canvas-wrap">
          <div class="viz-canvas-panel" id="viz-panel-a">
            <span class="viz-panel-label">A</span>
            <canvas id="viz-canvas-a" width="800" height="600"></canvas>
            <div class="viz-math-overlay" id="viz-math-overlay-a" style="display:none"></div>
            <div class="viz-empty" id="viz-empty-a"></div>
          </div>
          <div class="viz-canvas-panel" id="viz-panel-b" style="display:none">
            <span class="viz-panel-label">B</span>
            <canvas id="viz-canvas-b" width="800" height="600"></canvas>
            <div class="viz-math-overlay" id="viz-math-overlay-b" style="display:none"></div>
            <div class="viz-empty" id="viz-empty-b"></div>
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
        this._vizStopPlayback('b');
        if (this._viz.ws_b) {
          try { this._viz.ws_b.close(); } catch (_) {}
          this._viz.ws_b = null;
        }
        this._viz.info_b = null;
        const emptyEl = document.getElementById('viz-empty-b');
        if (emptyEl) {
          emptyEl.style.display = '';
          emptyEl.textContent = '';
        }
        const canvas = document.getElementById('viz-canvas-b');
        if (canvas) {
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        this._vizHideBoxplot();
      } else if (res && res.scene_b) {
        this._viz.info_b = res.scene_b;
        this._vizUpdateControls('b');
        this._vizConnectWs('b');
      }
      this._vizRenderSourceSummary();
    });

    const syncBtn = document.getElementById('viz-sync-camera');
    if (syncBtn) {
      syncBtn._on = false;
      syncBtn.addEventListener('click', () => {
        syncBtn._on = !syncBtn._on;
        syncBtn.classList.toggle('active', syncBtn._on);
        this.apiPost('/api/viz/sync-camera', {enabled: syncBtn._on}).catch(() => {});
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
        this._viz.dragging = true;
        this._viz.dragSlot = slot;
        this._viz.lastX = e.clientX;
        this._viz.lastY = e.clientY;
        this._viz.dragButton = e.button;
      });
      canvas.addEventListener('wheel', e => {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.1 : 0.9;
        this._vizSendWs(slot, {action: 'zoom', factor});
      }, {passive: false});
    });

    document.addEventListener('mousemove', e => {
      if (!this._viz.dragging) return;
      const dx = e.clientX - this._viz.lastX;
      const dy = e.clientY - this._viz.lastY;
      this._viz.lastX = e.clientX;
      this._viz.lastY = e.clientY;
      const slot = this._viz.dragSlot || 'a';
      if (this._viz.dragButton === 2) {
        this._vizSendWs(slot, {action: 'pan', dx, dy});
      } else {
        this._vizSendWs(slot, {action: 'orbit', dx, dy});
      }
    });

    document.addEventListener('mouseup', () => {
      this._viz.dragging = false;
    });
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
    const state = this._vizGetSlotState(slot);
    state.sourceLocked = !!loaded;
    if (picker) picker.style.display = loaded ? 'none' : 'inline-flex';
    if (current) current.style.display = loaded ? 'inline-flex' : 'none';
    if (currentText) {
      const model = this._vizGetModelByKey(state.modelKey);
      const modelText = model ? model.name : 'No model';
      currentText.textContent = `${slot.toUpperCase()} · ${modelText} · ${this._vizBasename(state.matPath)}`;
      currentText.title = state.matPath || '';
    }
    if (toolbar) {
      Array.from(toolbar.children).forEach((child, index) => {
        if (index === 0) return;
        child.style.display = loaded ? '' : 'none';
      });
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
      this._viz.info_a = res.validation_compare && res.scene_a ? res.scene_a : res;
      this._viz.compareLocked = !!res.compare_locked;
      this._vizUpdateControls('a');
      this._vizConnectWs('a');
      if (res.validation_compare && res.scene_b) {
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
    const text = item => String(item.key || '').toLowerCase();
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
      const symbolHtml = symbol ? `\\(${UI.escapeHtml(symbol)}\\)` : '-';
      return `
        <label class="viz-field-table-row">
          <input type="checkbox" class="viz-field-check" value="${UI.escapeAttr(item.key)}" ${checked}>
          <span class="viz-field-main">
            <span class="viz-field-name">${UI.escapeHtml(item.key)}</span>
            <span class="viz-field-label">${UI.escapeHtml(item.label || item.key)}</span>
          </span>
          <span class="viz-field-cell">${UI.escapeHtml(item.location || 'unknown')}</span>
          <span class="viz-field-cell">${UI.escapeHtml(item.frames || 'single')}</span>
          <span class="viz-field-cell">${UI.escapeHtml(item.tensor_kind || 'components')}</span>
          <span class="viz-field-cell">${UI.escapeHtml(family)}</span>
          <span class="viz-field-symbol" title="${UI.escapeAttr(formula || symbol)}">${symbolHtml}</span>
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

  _vizSetStatus(text) {
    const statusEl = document.getElementById('viz-status-text');
    if (statusEl) statusEl.textContent = text;
  },

  _vizShowCompare(show) {
    const panelB = document.getElementById('viz-panel-b');
    const toolbarB = document.getElementById('viz-toolbar-b');
    const btn = document.getElementById('viz-compare-toggle');
    const syncBtn = document.getElementById('viz-sync-camera');
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
    if (syncBtn) syncBtn.style.display = show ? '' : 'none';
    if (diffBtn) diffBtn.style.display = show ? '' : 'none';
    if (boxplotBtn) boxplotBtn.style.display = show ? '' : 'none';
    if (saveBtn) saveBtn.style.display = show ? '' : 'none';
    optionWraps.forEach(wrap => {
      if (wrap) wrap.style.display = show ? 'inline-flex' : 'none';
    });
    if (!show) {
      this._viz.diffPayload = null;
      this._vizHideBoxplot();
    }
    this._vizUpdatePanelLabels();
    this._vizRefreshCompareButtons();
    this._vizRenderSourceSummary();
  },

  _vizRefreshCompareButtons() {
    const syncBtn = document.getElementById('viz-sync-camera');
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

  _vizClassifyFieldFamily(key, meta = {}) {
    const text = `${key} ${meta.label || ''}`.toLowerCase();
    if (text.includes('stress')) return 'stress';
    if (text.includes('strain') || text.includes('peeq')) return 'strain';
    if (/(^u$|^frame_u$|^elastic_u$|^solid_u_nodal$|_u_|displacement)/.test(text)) return 'displacement';
    if (text.includes('reaction') || text.includes('internal force') || text.includes('nforc') || text.includes('force')) return 'reaction';
    return 'other';
  },

  _vizComponentLabels(family, meta = {}, key = '') {
    const nComp = Number(meta.n_components || 1);
    const text = `${key} ${meta.label || ''}`.toLowerCase();
    if (nComp <= 1) return [];
    if (text.includes('shell generalized stress') && nComp === 6) {
      return ['SF11', 'SF22', 'SF12', 'SM11', 'SM22', 'SM12'];
    }
    if (text.includes('shell generalized strain') && nComp === 6) {
      return ['GE11', 'GE22', 'GE12', 'GK11', 'GK22', 'GK12'];
    }
    if (text.includes('shell displacement') && nComp === 6) {
      return ['U1', 'U2', 'U3', 'UR1', 'UR2', 'UR3'];
    }
    if (text.includes('shell generalized nforc') && nComp === 6) {
      return ['F1', 'F2', 'F3', 'M1', 'M2', 'M3'];
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
      const labels = text.includes('validation_') ? ['NFORC1', 'NFORC2', 'NFORC3'] : ['1', '2', '3'];
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
    const text = `${key} ${meta.label || ''}`.toLowerCase();
    if (text.includes('viz_diff') || /\bdiff\b/.test(text)) return 'Diff';
    if (text.includes('cineq') || text.includes('inequality_violation')) return 'CInEQ';
    if (text.includes('equality_violation') || text.includes('self-equilibrium')) return 'CEQ';
    if (text.includes('generalized_residual_sf') || text.includes('generalized_total_sf') || text.includes('shakedown_sf_') || text.includes('rsdms_sf_') || /\bsf\b/.test(text)) return 'SF';
    if (text.includes('generalized_residual_sm') || text.includes('generalized_total_sm') || text.includes('shakedown_sm_') || text.includes('rsdms_sm_') || /\bsm\b/.test(text)) return 'SM';
    if (text.includes('shakedown_phi') || text.includes('ilyushin_phi') || text.includes('yield function')) return 'Phi';
    if (text.includes('peeq')) return 'PEEQ';
    if (text.includes('shell generalized stress')) return 'SGEN';
    if (text.includes('shell generalized strain')) return 'EGEN';
    if (text.includes('residual_stress') || text.includes('residual stress')) return 'RS';
    if (text.includes('force_error')) return 'FERR';
    if (text.includes('stress')) return 'S';
    if (text.includes('strain')) return 'E';
    if (/(^u$|^frame_u$|^elastic_u$|^solid_u_nodal$|_u_|displacement)/.test(text)) return 'U';
    if (text.includes('reaction')) return 'RF';
    if (text.includes('internal force') || text.includes('internal_force') || text.includes('nforc') || text.includes('force')) return 'NFORC';
    return String(key || 'VAR').replace(/^shakedown_/, '').replace(/^frame_/, '').replace(/^gauss_/, '').toUpperCase();
  },

  _vizShortFieldSource(key) {
    return String(key || '')
      .replace(/^shakedown_/, '')
      .replace(/^frame_/, '')
      .replace(/^gauss_/, '')
      .replace(/_stress$/, '')
      .replace(/_strain$/, '')
      .replace(/_/g, ' ');
  },

  _vizVariableMapText(variable) {
    if (!variable) return '';
    const key = String(variable.key || '');
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
      return `${variable.code} (${key}): shell generalized stress [SF, SM]`;
    }
    if (variable.code === 'EGEN') {
      return `${variable.code} (${key}): shell generalized strain [GE, GK]`;
    }
    return `${variable.code} (${key})`;
  },

  _vizScalarOptionLabel(family, key, meta = {}) {
    const text = `${key} ${meta.label || ''}`.toLowerCase();
    if (text.includes('viz_diff') || /\bdiff\b/.test(text)) return 'Value';
    if (text.includes('peeq')) return 'PEEQ';
    if (text.includes('violation')) return 'Magnitude';
    if (text.includes('shell generalized stress') || text.includes('shell generalized strain')) return 'Magnitude';
    if (family === 'stress') return 'Mises';
    if (family === 'strain') return 'Equivalent';
    if (family === 'displacement') return 'Magnitude';
    if (family === 'reaction') return 'Magnitude';
    return 'Value';
  },

  _vizEquivalentLabel(family, key, meta = {}) {
    const text = `${key} ${meta.label || ''}`.toLowerCase();
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
    const keys = visibleKeys.length ? visibleKeys : orderedKeys;
    const quantitiesByFamily = {};
    const variables = [];
    const codeCounts = {};
    keys.forEach(key => {
      const meta = fieldMeta[key];
      if (!meta) return;
      const code = this._vizVariableCode(key, meta);
      codeCounts[code] = (codeCounts[code] || 0) + 1;
    });
    const pushUnique = (variableKey, option) => {
      if (!quantitiesByFamily[variableKey]) quantitiesByFamily[variableKey] = [];
      if (!quantitiesByFamily[variableKey].some(item => item.value === option.value)) {
        quantitiesByFamily[variableKey].push(option);
      }
    };
    keys.forEach(key => {
      const meta = fieldMeta[key];
      if (!meta) return;
      const family = this._vizClassifyFieldFamily(key, meta);
      const code = this._vizVariableCode(key, meta);
      const label = codeCounts[code] > 1 ? `${code} (${this._vizShortFieldSource(key)})` : code;
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
      pushUnique(key, {
        family: key,
        fieldFamily: family,
        key,
        component: null,
        label: this._vizEquivalentLabel(family, key, meta),
        value: this._vizEncodeFieldChoice(key, null),
      });
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

  _vizSetFrame(slot, frame) {
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
    this._vizSendWs(slot, {action: 'set_frame', frame: target});
  },

  _vizStepFrame(slot, delta) {
    const info = this._vizSlotInfo(slot);
    const nFrames = Math.max(1, Number(info?.n_frames || 0));
    if (!info || nFrames <= 1) return;
    const current = this._vizClampFrame(info, info.active_frame || 0);
    const next = (current + Number(delta || 0) + nFrames) % nFrames;
    this._vizSetFrame(slot, next);
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
      this._vizStepFrame(slot, 1);
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
    const emptyEl = document.getElementById(`viz-empty-${slot}`);
    if (emptyEl) emptyEl.style.display = 'none';
    this._vizRenderSourceSummary();
    this._vizRefreshCompareButtons();
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
        };
        img.src = imgUrl;
      } else {
        try {
          const info = JSON.parse(evt.data);
          if (slot === 'a') this._viz.info_a = info;
          else this._viz.info_b = info;
          this._vizUpdateControls(slot);
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
    } else if (ws && ws.readyState === WebSocket.CONNECTING) {
      ws.addEventListener('open', () => ws.send(JSON.stringify(msg)), {once: true});
    }
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
    if (!btn) return;
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
