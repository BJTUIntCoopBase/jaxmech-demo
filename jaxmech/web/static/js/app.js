const App = {
  currentPage: 'dashboard',
  models: [],
  activeModel: null,
  selectedMat: '',
  tasksTimer: null,
  _viz: { info_a: null, info_b: null, ws_a: null, ws_b: null },

  async init() {
    document.querySelectorAll('.nav-item').forEach((item) => {
      item.addEventListener('click', () => this.navigate(item.dataset.page));
    });
    await Promise.all([this.loadStatus(), this.loadModels()]);
    this.navigate('dashboard');
  },

  async api(path, options = {}) {
    const response = await fetch(path, {
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    });
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  },

  navigate(page) {
    this.currentPage = page;
    document.querySelectorAll('.page').forEach((el) => el.classList.remove('active'));
    const pageEl = document.getElementById(`page-${page}`);
    if (pageEl) pageEl.classList.add('active');
    document.querySelectorAll('.nav-item').forEach((el) => el.classList.toggle('active', el.dataset.page === page));
    this.renderPage(page);
  },

  renderPage(page) {
    if (['plastic', 'direct-methods', 'validation', 'shell', 'topopt'].includes(page)) return this.renderLocked(page);
    if (page === 'dashboard') return this.renderDashboard();
    if (page === 'models') return this.renderModels();
    if (page === 'elastic') return this.renderElastic();
    if (page === 'shakedown') return this.renderShakedown();
    if (page === 'visualization') return this.renderVisualization();
    if (page === 'tasks') return this.renderTasks();
    if (page === 'docs') return this.renderDocs();
    if (page === 'settings') return this.renderSettings();
  },

  async loadStatus() {
    try {
      const status = await this.api('/api/status/quick');
      const ok = !!status.wsl?.available;
      document.getElementById('wsl-dot').className = `status-dot ${ok ? 'online' : 'offline'}`;
      document.getElementById('wsl-text').textContent = ok ? 'WSL 可用' : 'WSL 未就绪';
    } catch {
      document.getElementById('wsl-dot').className = 'status-dot offline';
      document.getElementById('wsl-text').textContent = '状态未知';
    }
  },

  async loadModels() {
    try {
      this.models = await this.api('/api/models');
      if (!this.activeModel && this.models.length) this.activeModel = this.models[0];
      this.updateActiveModelLabel();
    } catch {
      this.models = [];
    }
  },

  updateActiveModelLabel() {
    document.getElementById('sidebar-active-model').textContent = this.activeModel?.name || '未选择';
  },

  async selectModel(source, name) {
    this.activeModel = await this.api(`/api/models/${source}/${name}`);
    this.updateActiveModelLabel();
    this.navigate('models');
  },

  renderDashboard() {
    const el = document.getElementById('dashboard-content');
    const active = this.activeModel?.name || '未选择';
    el.innerHTML = `
      ${UI.header('jaxmech demo', '实体单元弹性 inc_analysis、CVXPY shakedown、Web 端和 MAT 可视化')}
      <div class="status-grid">
        ${UI.statusCard('EL', '弹性分析', 'Solid linear elastic')}
        ${UI.statusCard('SD', 'Shakedown', 'C formulation / CVXPY')}
        ${UI.statusCard('VZ', '可视化', 'Elastic + shakedown MAT')}
        ${UI.statusCard('MD', '当前模型', active)}
      </div>
      <div class="module-grid" style="margin-top:18px">
        ${this.moduleCard('elastic', 'EL', '弹性 inc_analysis', '从 solid INP 生成主 MAT')}
        ${this.moduleCard('shakedown', 'SD', 'Shakedown', '读取弹性 MAT 并用 CVXPY 求解')}
        ${this.moduleCard('visualization', 'VZ', 'MAT 可视化', '查看应力、位移和 shakedown 场变量')}
        ${this.moduleCard('validation', 'VA', 'ODB 验证', '完整版功能', true)}
      </div>`;
  },

  moduleCard(id, icon, title, desc, locked = false) {
    return `
      <div class="module-card ${locked ? 'locked' : ''}" onclick="App.navigate('${id}')">
        <div class="module-icon">${icon}</div>
        <div class="module-title">${UI.escapeHtml(title)}</div>
        <div class="module-desc">${UI.escapeHtml(desc)}</div>
        <button class="module-action">${locked ? '查看说明' : '打开'}</button>
      </div>`;
  },

  renderModels() {
    const el = document.getElementById('models-content');
    el.innerHTML = `
      ${UI.header('模型浏览', 'Examples/ 与 StoredModels/ 中的 demo 模型')}
      <div class="model-list">
        ${this.models.length ? this.models.map((model) => UI.modelItem(model)).join('') : UI.empty('暂无模型')}
      </div>
      ${this.activeModel ? `<div class="panel" style="margin-top:16px">
        <h3>${UI.escapeHtml(this.activeModel.name)}</h3>
        <p>${UI.escapeHtml(this.activeModel.path)}</p>
        <p>INP: ${(this.activeModel.abaqus?.inp_files || []).map(UI.escapeHtml).join(', ') || '无'}</p>
        <p>Elastic MAT: ${(this.activeModel.inc_analysis?.mat_files || []).map(UI.escapeHtml).join(', ') || '无'}</p>
        <p>Shakedown MAT: ${(this.activeModel.shakedown?.mat_files || []).map(UI.escapeHtml).join(', ') || '无'}</p>
      </div>` : ''}`;
  },

  cfgPath(module) {
    if (!this.activeModel) return '';
    const file = module === 'shakedown' ? 'shakedown_analysis.template.cfg' : 'inc_analysis.template.cfg';
    return `${this.activeModel.path}\\${module}\\${file}`;
  },

  renderElastic() {
    const el = document.getElementById('elastic-content');
    if (!this.activeModel) {
      el.innerHTML = `${UI.header('弹性 inc_analysis', '请选择一个模型')}${UI.empty('尚未选择模型')}`;
      return;
    }
    const cfg = this.cfgPath('inc_analysis');
    el.innerHTML = `
      ${UI.header('弹性 inc_analysis', '仅支持实体单元 linear elastic，结果写入 inc_analysis/*.mat')}
      <div class="panel">
        <p><strong>模型</strong> ${UI.escapeHtml(this.activeModel.name)}</p>
        <p><strong>配置</strong> ${UI.escapeHtml(cfg)}</p>
        <p><strong>输入</strong> ${(this.activeModel.abaqus?.inp_files || []).map(UI.escapeHtml).join(', ') || '无 INP'}</p>
        ${UI.button('运行弹性分析', `App.runTask('jaxmech.tools.build.build_model_mat', ['--config', '${UI.escapeAttr(cfg)}'])`)}
      </div>`;
  },

  renderShakedown() {
    const el = document.getElementById('shakedown-content');
    if (!this.activeModel) {
      el.innerHTML = `${UI.header('Shakedown', '请选择一个模型')}${UI.empty('尚未选择模型')}`;
      return;
    }
    const cfg = this.cfgPath('shakedown');
    el.innerHTML = `
      ${UI.header('Shakedown', '仅支持实体单元、C formulation、CVXPY/Clarabel backend')}
      <div class="panel">
        <p><strong>模型</strong> ${UI.escapeHtml(this.activeModel.name)}</p>
        <p><strong>配置</strong> ${UI.escapeHtml(cfg)}</p>
        <p><strong>弹性 MAT</strong> ${(this.activeModel.inc_analysis?.mat_files || []).map(UI.escapeHtml).join(', ') || '请先运行弹性分析'}</p>
        ${UI.button('运行 Shakedown', `App.runTask('jaxmech.modules.shakedown.run', ['--config', '${UI.escapeAttr(cfg)}'])`)}
      </div>`;
  },

  async runTask(module, args) {
    const task = await this.api('/api/run', { method: 'POST', body: JSON.stringify({ module, args }) });
    this.navigate('tasks');
    this.viewTask(task.id);
  },

  async renderTasks() {
    const el = document.getElementById('tasks-content');
    const tasks = await this.api('/api/tasks').catch(() => []);
    el.innerHTML = `
      ${UI.header('任务管理', '查看分析任务、日志和产物')}
      <div class="task-list">${tasks.length ? tasks.map((task) => UI.taskItem(task)).join('') : UI.empty('暂无任务')}</div>
      <div id="task-detail" style="margin-top:16px"></div>`;
  },

  async viewTask(id) {
    const detail = await this.api(`/api/tasks/${id}`);
    const el = document.getElementById('task-detail');
    if (!el) return;
    const logs = (detail.logs || []).slice(-120).map((entry) => `<div class="log-line ${entry.type}">${UI.escapeHtml(entry.text || '')}</div>`).join('');
    const artifacts = (detail.artifacts || []).map((item) => `<li>${UI.escapeHtml(item.label)}: ${UI.escapeHtml(item.path)}</li>`).join('');
    el.innerHTML = `
      <div class="panel">
        <h3>Task ${UI.escapeHtml(detail.id)}</h3>
        <p>${UI.escapeHtml(detail.module)} | ${UI.escapeHtml(detail.status)}</p>
        ${artifacts ? `<ul>${artifacts}</ul>` : ''}
        <div class="log-box">${logs}</div>
      </div>`;
    if (['pending', 'running'].includes(detail.status)) {
      clearTimeout(this.tasksTimer);
      this.tasksTimer = setTimeout(() => this.viewTask(id), 1500);
    } else {
      await this.loadModels();
    }
  },

  async renderVisualization() {
    const el = document.getElementById('viz-content');
    const mats = this.activeModel ? await this.scanMats() : [];
    const options = mats.map((item) => `<option value="${UI.escapeAttr(item.abs_path)}">${UI.escapeHtml(item.rel_path)}</option>`).join('');
    el.innerHTML = `
      ${UI.header('MAT 场变量可视化', '加载 elastic 或 shakedown MAT，查看场变量和 A/B 对比')}
      <div class="viz-layout">
        <div class="panel viz-controls">
          <label>MAT 文件</label>
          <select id="viz-mat-select">${options}</select>
          <button class="btn-primary" onclick="App.vizLoadSelected()">加载场景 A</button>
          <button class="btn-secondary" onclick="App.vizLoadSelected('b')">加载场景 B</button>
          <select id="viz-field-select" onchange="App.vizSetField(this.value)"></select>
          <div id="viz-status-text">请选择 MAT 文件</div>
        </div>
        <div class="viz-canvas-wrap">
          <canvas id="viz-canvas-a" width="800" height="600"></canvas>
        </div>
      </div>`;
    if (mats.length) {
      this.selectedMat = mats[0].abs_path;
    }
  },

  async scanMats() {
    if (!this.activeModel) return [];
    const data = await this.api('/api/models/scan-mats', { method: 'POST', body: JSON.stringify({ model_path: this.activeModel.path }) });
    return data.mat_files || [];
  },

  async vizLoadSelected(slot = 'a') {
    const select = document.getElementById('viz-mat-select');
    const path = select?.value || this.selectedMat;
    await this._vizLoadSelectedMat(slot, path);
  },

  async _vizLoadSelectedMat(slot, path, selectedFields = null) {
    this.selectedMat = path;
    const data = await this.api('/api/viz/load', { method: 'POST', body: JSON.stringify({ mat_path: path, slot, selected_fields: selectedFields }) });
    this._viz[`info_${slot}`] = data;
    this.populateFields(data);
    document.getElementById('viz-status-text').textContent = `场景 ${slot.toUpperCase()} 已加载`;
    this.openVizSocket(slot);
    return data;
  },

  populateFields(info) {
    const select = document.getElementById('viz-field-select');
    if (!select || !info.fields) return;
    select.innerHTML = info.fields.map((key) => `<option value="${UI.escapeAttr(key)}">${UI.escapeHtml(info.field_labels?.[key] || key)}</option>`).join('');
    if (info.active_field) select.value = info.active_field;
  },

  async vizSetField(key, slot = 'a') {
    const data = await this.api('/api/viz/set-field', { method: 'POST', body: JSON.stringify({ key, slot }) });
    this._viz[`info_${slot}`] = data;
  },

  openVizSocket(slot = 'a') {
    const old = this._viz[`ws_${slot}`];
    if (old) old.close();
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/api/viz/ws/${slot}`);
    ws.binaryType = 'blob';
    ws.onmessage = async (event) => {
      if (typeof event.data === 'string') {
        try {
          const info = JSON.parse(event.data);
          this._viz[`info_${slot}`] = info;
          this.populateFields(info);
        } catch {}
        return;
      }
      if (slot !== 'a') return;
      const canvas = document.getElementById('viz-canvas-a');
      if (!canvas) return;
      const bitmap = await createImageBitmap(event.data);
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      bitmap.close();
    };
    this._viz[`ws_${slot}`] = ws;
  },

  renderDocs() {
    document.getElementById('docs-content').innerHTML = `
      ${UI.header('说明文件', 'demo 只包含公开演示所需说明')}
      <div class="panel"><p>请从仓库根目录 README.md 开始。完整版本与合作请联系 gengchen@bjtu.edu.cn。</p></div>`;
  },

  async renderSettings() {
    const status = await this.api('/api/status').catch(() => null);
    document.getElementById('settings-content').innerHTML = `
      ${UI.header('设置', '环境检测')}
      <div class="panel"><pre>${UI.escapeHtml(JSON.stringify(status, null, 2))}</pre></div>`;
  },

  renderLocked(page) {
    const names = {
      plastic: '弹塑性增量分析',
      'direct-methods': 'Direct Methods / DCA',
      validation: 'ODB 验证',
      shell: 'Shell 分析',
      topopt: 'Topology Optimization',
    };
    const el = document.getElementById(`${page}-content`);
    el.innerHTML = `
      ${UI.header(names[page] || '完整版功能', '此功能在 demo 中锁定')}
      <div class="panel locked-panel">
        <h3>完整版功能</h3>
        <p>当前公开 demo 只开放实体单元弹性 inc_analysis、CVXPY-backend shakedown、Web 端和 MAT 可视化。</p>
        <p>该菜单保留用于展示完整 jaxmech 工作流结构，但不包含对应代码与后端入口。</p>
        <p>完整版本或合作请联系 gengchen@bjtu.edu.cn。</p>
      </div>`;
  },
};

window.App = App;
document.addEventListener('DOMContentLoaded', () => App.init());
