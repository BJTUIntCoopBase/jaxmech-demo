/**
 * jaxmech Web Frontend — Reusable UI Components
 */
const UI = {
  /** Create a status card */
  statusCard(icon, label, value, colorClass) {
    return `
      <div class="status-card">
        <div class="status-icon ${colorClass}">${icon}</div>
        <div class="status-info">
          <div class="status-label">${label}</div>
          <div class="status-value">${value}</div>
        </div>
      </div>`;
  },

  /** Create a module card */
  moduleCard(id, icon, title, desc, colorClass, action) {
    return `
      <div class="module-card ${colorClass}" data-module="${id}" onclick="App.navigate('${id}')">
        <div class="module-icon">${icon}</div>
        <div class="module-title">${title}</div>
        <div class="module-desc">${desc}</div>
        <button class="module-action" onclick="event.stopPropagation(); App.navigate('${id}')">
          ${action} →
        </button>
      </div>`;
  },

  /** Context hint bar */
  hint(type, text, actionText, actionFn) {
    const icons = { warning: '⚠️', error: '❌', info: 'ℹ️', success: '✅' };
    const actionHtml = actionText
      ? `<button class="hint-action" onclick="${actionFn}">${actionText}</button>`
      : '';
    return `
      <div class="hint-bar ${type}">
        <span>${icons[type] || ''}</span>
        <span class="hint-text">${text}</span>
        ${actionHtml}
      </div>`;
  },

  /** Model list item */
  modelItem(model) {
    const badges = [];
    if (model.abaqus.inp_count > 0)
      badges.push(`<span class="badge has-inp">${model.abaqus.inp_count} INP</span>`);
    if (model.inc_analysis.has_mat)
      badges.push(`<span class="badge has-mat">${model.inc_analysis.mat_files.length} MAT</span>`);
    if (model.dca?.has_mat)
      badges.push(`<span class="badge has-mat">${model.dca.mat_files.length} DCA</span>`);
    if (model.validation.validated)
      badges.push(`<span class="badge validated">已验证</span>`);
    if (model.shakedown.has_results)
      badges.push(`<span class="badge has-sd">安定</span>`);

    return `
      <div class="model-item" onclick="App.selectModel('${model.source}', '${model.name}')">
        <span style="font-size:20px">📁</span>
        <span class="model-name">${model.name}</span>
        <div class="model-badges">${badges.join('')}</div>
        <span class="model-source">${model.source === 'examples' ? 'Examples' : 'Stored'}</span>
      </div>`;
  },

  /** Task list item */
  taskModuleLabel(task) {
    const module = String(task?.module || '');
    const args = Array.isArray(task?.args) ? task.args.join(' ') : String(task?.args || '');
    const text = `${module} ${args}`.toLowerCase();
    if (text.includes('steady_state_dca') || text.includes('direct_methods.dca') || (module.includes('run_steady_state') && text.includes('dca'))) {
      return 'DCA';
    }
    if (text.includes('steady_state_rsdm') || text.includes('direct_methods.rsdm') || (module.includes('run_steady_state') && text.includes('rsdm'))) {
      return 'RSDM';
    }
    if (text.includes('rsdm_shakedown') || text.includes('rsdms')) {
      return 'RSDM-S';
    }
    if (module.includes('shakedown')) {
      return 'Shakedown';
    }
    if (module.includes('inc_analysis') || module.includes('build_model_mat')) {
      return 'Incremental Analysis';
    }
    return module || 'Task';
  },

  taskArtifactPath(task) {
    const artifacts = Array.isArray(task?.artifacts) ? task.artifacts : [];
    const mat = artifacts.find((artifact) => {
      const kind = String(artifact?.kind || '').toLowerCase();
      const path = String(artifact?.path || '').toLowerCase();
      return kind.includes('mat') || path.endsWith('.mat');
    });
    return String(mat?.path || '').trim();
  },

  prepareTaskPathScroll(row) {
    const box = row?.querySelector?.('.task-artifact-path');
    const text = row?.querySelector?.('.task-artifact-path-text');
    if (!box || !text) return;
    const overflow = Math.max(0, text.scrollWidth - box.clientWidth);
    box.style.setProperty('--task-path-scroll', overflow > 4 ? `-${overflow}px` : '0px');
    box.style.setProperty('--task-path-duration', `${Math.min(18, Math.max(4, overflow / 42))}s`);
    box.classList.toggle('is-overflowing', overflow > 4);
  },

  /** Task list item */
  taskItem(task, options = {}) {
    const statusMap = {
      running: '运行中', success: '完结', failed: '失败', pending: '等待中', stopped: '已停止'
    };
    const selectable = Boolean(options.selectable);
    const checkbox = selectable
      ? `<input type="checkbox" class="task-select-checkbox" value="${this.escapeAttr(task.id)}"
          onclick="event.stopPropagation(); App.updateTaskSelectionToolbar && App.updateTaskSelectionToolbar()" />`
      : '';
    const openButton = task.task_dir
      ? `<button class="btn btn-secondary task-open-btn" onclick="event.stopPropagation(); App.openTaskFolder('${this.escapeAttr(task.id)}')">打开文件夹</button>`
      : '';
    const moduleLabel = this.taskModuleLabel(task);
    const showArtifactPath = Object.prototype.hasOwnProperty.call(options, 'showArtifactPath')
      ? Boolean(options.showArtifactPath)
      : selectable;
    const artifactPath = showArtifactPath ? this.taskArtifactPath(task) : '';
    const artifactPathHtml = artifactPath
      ? `<span class="task-artifact-path" title="${this.escapeAttr(artifactPath)}"><span class="task-artifact-path-text">${this.escapeHtml(artifactPath)}</span></span>`
      : '';
    return `
      <div class="task-item" onclick="App.viewTask('${this.escapeAttr(task.id)}')" onmouseenter="UI.prepareTaskPathScroll(this)">
        ${checkbox}
        <span class="task-status-badge ${task.status}">${statusMap[task.status] || task.status}</span>
        <span class="task-main">
          <span class="task-module">${this.escapeHtml(moduleLabel)}</span>
          ${artifactPathHtml}
        </span>
        <span class="task-time">${task.created_at}</span>
        ${openButton}
      </div>`;
  },

  /** Log line */
  logLine(entry) {
    const cls = entry.type === 'stderr' ? 'stderr' : (entry.type === 'info' ? 'info' : 'stdout');
    return `<div class="log-line ${cls}">${this.escapeHtml(entry.text)}</div>`;
  },

  /** Doc list item */
  docItem(doc) {
    const badge = `<span style="font-size:11px;padding:2px 8px;border-radius:99px;
      background:rgba(255,255,255,0.07);color:var(--text-secondary);margin-left:8px;
      vertical-align:middle">${doc.category_icon || '📄'} ${doc.category_label || ''}</span>`;
    return `
      <div class="doc-item" onclick="App.viewDoc('${doc.rel_path}')">
        <span style="font-size:18px">📄</span>
        <span style="font-weight:600">${doc.name}</span>${badge}
      </div>`;
  },

  /** Empty state */
  empty(icon, text) {
    return `
      <div class="empty-state">
        <div class="empty-icon">${icon}</div>
        <div class="empty-text">${text}</div>
      </div>`;
  },

  /** Escape HTML */
  escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
  },

  /** Escape for use inside an HTML attribute value (double-quoted) */
  escapeAttr(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  },


  /**
   * 双源筛选 Toggle 按钮组
   * @param {{ id: string, label: string }[]} options
   * @param {Set<string>} activeSet   当前激活的 id 集合
   * @param {string} onToggleFn       全局可调用的函数名，接收 (id) 参数
   */
  toggleBar(options, activeSet, onToggleFn) {
    const btns = options.map(o => {
      const isActive = activeSet.has(o.id);
      return `<button class="toggle-btn${isActive ? ' active' : ''}"
        onclick="${onToggleFn}('${o.id}')">${o.label}</button>`;
    }).join('');
    return `<div class="toggle-bar">${btns}</div>`;
  },

  /**
   * MAT 多选列表（用于安定分析 Step 1）
   * @param {string[]} matFiles  文件名列表
   * @param {Set<string>} selectedSet
   */
  matSelector(matFiles, selectedSet) {
    if (!matFiles.length) return `<div style="color:var(--text-secondary);font-size:13px">暂无可用 .mat 文件</div>`;
    return matFiles.map((f, i) => {
      const checked = selectedSet.has(f) ? 'checked' : '';
      return `<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06)">
        <input type="checkbox" id="mat-check-${i}" data-mat="${this.escapeHtml(f)}" ${checked}
          style="width:16px;height:16px;flex-shrink:0;accent-color:var(--accent);cursor:pointer"
          onchange="App.onMatSelectionChange()">
        <label for="mat-check-${i}" style="font-size:14px;cursor:pointer;user-select:none">📦 ${this.escapeHtml(f)}</label>
      </div>`;
    }).join('');
  },

  /**
   * Show a confirmation modal dialog.
   * @param {string} messageHtml - HTML content to show in the dialog body
   * @param {Function} onConfirm  - Callback invoked when user clicks "确定"
   */
  confirm(messageHtml, onConfirm) {
    // Remove any existing modal
    const existing = document.getElementById('ui-confirm-modal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'ui-confirm-modal';
    modal.style.cssText = [
      'position:fixed;inset:0;z-index:9999',
      'display:flex;align-items:center;justify-content:center',
      'background:rgba(0,0,0,0.55);backdrop-filter:blur(4px)',
      'animation:fadeIn .15s ease',
    ].join(';');

    modal.innerHTML = `
      <div style="
        background:var(--surface,#1e1e2e);border:1px solid rgba(255,255,255,0.12);
        border-radius:16px;padding:32px 28px;max-width:420px;width:90%;
        box-shadow:0 20px 60px rgba(0,0,0,0.5);text-align:center;
        animation:slideUp .18s ease;
      ">
        <div style="font-size:36px;margin-bottom:16px">📂</div>
        <div style="font-size:15px;line-height:1.6;color:var(--text-primary,#fff);margin-bottom:24px">
          ${messageHtml}
        </div>
        <div style="display:flex;gap:12px;justify-content:center">
          <button id="ui-confirm-ok" style="
            padding:10px 36px;border-radius:8px;border:none;
            background:var(--accent,#7c3aed);color:#fff;font-size:14px;
            font-weight:600;cursor:pointer;
          ">确定 →</button>
        </div>
      </div>`;

    const close = () => modal.remove();
    modal.querySelector('#ui-confirm-ok').addEventListener('click', () => {
      close();
      onConfirm();
    });
    // No background-click or Escape dismissal — user must click OK
    document.body.appendChild(modal);
  },

};

window.UI = UI;
