const UI = {
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text == null ? '' : String(text);
    return div.innerHTML;
  },

  escapeAttr(text) {
    return String(text == null ? '' : text)
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  },

  header(title, subtitle) {
    return `
      <div class="page-header">
        <h1 class="page-title">${this.escapeHtml(title)}</h1>
        <p class="page-subtitle">${this.escapeHtml(subtitle || '')}</p>
      </div>`;
  },

  statusCard(icon, label, value, colorClass = '') {
    return `
      <div class="status-card">
        <div class="status-icon ${colorClass}">${this.escapeHtml(icon)}</div>
        <div class="status-info">
          <div class="status-label">${this.escapeHtml(label)}</div>
          <div class="status-value">${this.escapeHtml(value)}</div>
        </div>
      </div>`;
  },

  modelItem(model) {
    const badges = [];
    if (model.abaqus?.inp_count) badges.push(`<span class="badge has-inp">${model.abaqus.inp_count} INP</span>`);
    if (model.inc_analysis?.has_mat) badges.push(`<span class="badge has-mat">${model.inc_analysis.mat_files.length} MAT</span>`);
    if (model.shakedown?.has_results) badges.push(`<span class="badge has-sd">Shakedown</span>`);
    return `
      <div class="model-item" onclick="App.selectModel('${this.escapeAttr(model.source)}','${this.escapeAttr(model.name)}')">
        <span class="model-name">${this.escapeHtml(model.name)}</span>
        <div class="model-badges">${badges.join('')}</div>
        <span class="model-source">${model.source === 'examples' ? 'Examples' : 'Stored'}</span>
      </div>`;
  },

  taskItem(task) {
    const label = { pending: '等待', running: '运行中', success: '完成', failed: '失败' }[task.status] || task.status;
    return `
      <div class="task-item" onclick="App.viewTask('${this.escapeAttr(task.id)}')">
        <span class="task-status-badge ${this.escapeAttr(task.status)}">${this.escapeHtml(label)}</span>
        <span class="task-module">${this.escapeHtml(task.module)}</span>
        <span class="task-time">${this.escapeHtml(task.created_at || '')}</span>
      </div>`;
  },

  empty(title, text = '') {
    return `<div class="empty-state"><div class="empty-text">${this.escapeHtml(title)}</div><p>${this.escapeHtml(text)}</p></div>`;
  },

  button(label, onclick, cls = 'btn-primary') {
    return `<button class="${cls}" onclick="${onclick}">${this.escapeHtml(label)}</button>`;
  },
};
