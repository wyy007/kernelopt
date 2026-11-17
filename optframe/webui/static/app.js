/* optframe 控制台 · 只读前端（无框架、无 CDN） */
const state = {
  runs: [],
  run: null,
  diff: {},
  tab: 'list',
  filter: 'all',
  q: '',
  prevId: null,
};

const $ = (id) => document.getElementById(id);
const el = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };
const fmt = (v, d = 2) => (v === null || v === undefined || v === '' ? '—' : Number(v).toFixed(d));

async function getJSON(url) {
  if (window.SNAPSHOT && window.SNAPSHOT[url]) return window.SNAPSHOT[url];
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(r.status + ' ' + url);
  return r.json();
}

async function boot() {
  try {
    const data = await getJSON('/api/runs');
    state.runs = data.runs || [];
    $('chip-root').textContent = data.root || 'runs';
    renderRuns();
    const pick = state.runs.find((r) => (r.kinds || []).includes('analysis')) || state.runs[0];
    if (pick) await openRun(pick.id);
  } catch (e) {
    $('title').textContent = '无法加载数据';
    $('subtitle').textContent = String(e);
  }
}

function renderRuns() {
  const box = $('runs');
  box.innerHTML = '';
  for (const r of state.runs) {
    const row = el('div', 'run' + (state.run && state.run.id === r.id ? ' on' : ''));
    row.innerHTML = `<i></i><div class="t"><div class="n">${escapeHtml(r.id)}</div>
      <div class="m">${timeStr(r.mtime)} · ${(r.kinds || []).join(' / ')}</div></div>`;
    row.onclick = () => openRun(r.id);
    box.appendChild(row);
  }
}

function timeStr(ts) {
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, '0');
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

async function openRun(id) {
  try {
    state.run = await getJSON(`/api/runs/${encodeURIComponent(id)}`);
  } catch (e) {
    $('title').textContent = id;
    $('subtitle').textContent = '这个运行没有可展示的分析产物';
    $('rows').innerHTML = '<tr><td colspan="8" class="empty">没有 bottlenecks.json / ledger.jsonl</td></tr>';
    return;
  }
  state.diff = {};
  state.prevId = null;

  const prev = state.runs.find((r) => r.id !== id && (r.kinds || []).includes('analysis')
    && r.mtime < state.run.mtime);
  if (prev) {
    try {
      const d = await getJSON(`/api/diff?cur=${encodeURIComponent(id)}&prev=${encodeURIComponent(prev.id)}`);
      state.prevId = prev.id;
      for (const row of d.rows || []) {
        state.diff[row.op + '||' + row.shape] = row;
      }
    } catch (e) { /* 对比失败不影响主视图 */ }
  }

  renderHeader();
  renderTable();
  renderReport();
  renderRuns();
}

function renderHeader() {
  const r = state.run || {};
  $('title').textContent = r.id || '-';
  $('subtitle').textContent = (r.files && r.files.bottlenecks) || '';
  $('chip-run').textContent = r.id || '-';
  $('s-kinds').textContent = (r.files ? Object.keys(r.files).join(' / ') : '-');
  $('s-findings').textContent = (r.summary && r.summary.findings) || 0;
  $('s-elig').textContent = (r.summary && r.summary.eligible) || 0;
  $('s-busy').textContent = r.summary && r.summary.gpu_busy != null ? fmt(r.summary.gpu_busy, 1) + '%' : '—';
  $('s-prev').textContent = state.prevId || '无';
  $('foot-right').textContent = state.prevId ? `回退判定：SOL 下降 ≥ 2pp 且占比 ≥ 1%` : '无历史基线，Δ 列不可用';
}

function rowDelta(row) {
  return state.diff[row.op + '||' + row.shape];
}

function isRegression(row) {
  const d = rowDelta(row);
  return !!(d && d.d_sol != null && d.d_sol <= -2 && (row.pct || 0) >= 1);
}

function filteredRows() {
  const q = state.q.trim().toLowerCase();
  return (state.run.rows || []).filter((r) => {
    const isReg = isRegression(r);
    if (state.filter === 'elig' && !r.eligible) return false;
    if (state.filter === 'excl' && r.eligible) return false;
    if (state.filter === 'reg' && !isReg) return false;
    if (q) {
      const hay = `${r.op} ${r.kernel} ${r.shape}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function tagFor(r) {
  if (r.eligible) return ['keep', '进优化'];
  const map = {
    below_amdahl_threshold: ['', '低于阈值'],
    near_roofline: ['', '贴顶'],
    insufficient_evidence: ['', '无证据'],
    profiled_timing: ['', '待复核'],
    hand_optimized_kernel: ['', '手写优化'],
    system_memcpy: ['', 'memcpy'],
    system_collective: ['', '通信'],
  };
  return map[r.reason] || ['', r.reason || '—'];
}

function renderTable() {
  const rows = filteredRows();
  const total = (state.run.rows || []).length;
  const elig = (state.run.rows || []).filter((r) => r.eligible).length;
  let regs = 0;
  for (const r of state.run.rows || []) {
    if (isRegression(r)) regs += 1;
  }
  $('c-all').textContent = total;
  $('c-elig').textContent = elig;
  $('c-reg').textContent = regs;
  $('c-excl').textContent = total - elig;

  const tbody = $('rows');
  tbody.innerHTML = '';
  for (const r of rows.slice(0, 400)) {
    const d = rowDelta(r);
    const isReg = isRegression(r);
    const tr = el('tr', (r.eligible ? 'elig ' : '') + (isReg ? 'reg' : ''));
    const [tcls, ttext] = tagFor(r);
    const sol = r.sol == null ? '—' : `<span class="bar"><i style="width:${Math.min(100, r.sol)}%"></i></span> ${fmt(r.sol, 1)}`;
    const dText = (!d || d.d_sol == null) ? 'n/a'
      : (d.d_sol > 0 ? '+' : '▼ ') + fmt(d.d_sol, 1) + 'pp';
    tr.innerHTML = `<td>${r.rank}</td>
      <td title="${escapeHtml(r.shape || '')}">${escapeHtml(r.op || '')} ${escapeHtml(r.shape || '')}</td>
      <td>${escapeHtml(r.category || '')}</td><td>${sol}</td>
      <td class="num">${fmt(r.pct, 2)}%</td>
      <td><span class="tag ${r.bound === 'memory' ? 'mem' : r.bound === 'compute' ? 'cmp' : ''}">${escapeHtml(r.bound || '—')}</span></td>
      <td><span class="tag ${tcls}">${escapeHtml(ttext)}</span></td>
      <td class="num ${isReg ? 'd' : ''}">${dText}</td>`;
    tbody.appendChild(tr);
  }
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty">当前筛选无数据</td></tr>';
  }
  $('foot-left').textContent = `显示 ${Math.min(rows.length, 400)} / ${rows.length} · 共 ${total} 个算子`;
}

function renderReport() {
  const r = state.run || {};
  const s = r.summary || {};
  const box = $('report-metrics');
  box.innerHTML = `
    <div class="m"><div class="k">GPU 忙碌</div><div class="v">${s.gpu_busy != null ? fmt(s.gpu_busy, 1) + '%' : '—'}</div><div class="s">窗口内</div></div>
    <div class="m"><div class="k">可优化算子</div><div class="v">${s.eligible || 0} / ${s.findings || 0}</div><div class="s">最大占比 ${fmt(s.top_pct, 2)}%</div></div>
    <div class="m"><div class="k">覆盖份额</div><div class="v">${fmt(s.coverage, 1)}%</div><div class="s">可优化项合计</div></div>
    <div class="m"><div class="k">账本记录</div><div class="v">${(r.ledger || []).length}</div><div class="s">最近 50 条</div></div>`;

  const last = (r.ledger || []).filter((x) => x.stage === 'candidate').slice(-10);
  $('rp-verdict').textContent = last.length ? (last[last.length - 1].verdict || '—') : '无评测记录';
  $('rp-note').textContent = last.length
    ? `ratio ${fmt(last[last.length - 1].ratio, 4)} · CI [${fmt(last[last.length - 1].ratio_low, 4)}, ${fmt(last[last.length - 1].ratio_high, 4)}] · ${last[last.length - 1].cache_policy || ''}`
    : '这个 run 还没有候选评测记录';

  $('rp-ledger').innerHTML = (last || []).map((x) =>
    `<div>${escapeHtml(x.verdict)} · ratio ${fmt(x.ratio, 4)} · ${escapeHtml(x.generator || '')}</div>`).join('') || '—';

  const e = r.e2e;
  $('rp-e2e').textContent = e
    ? `延迟比 ${fmt(e.lat_ratio_median, 4)} · CI [${fmt((e.lat_ratio_ci || [])[0], 4)}, ${fmt((e.lat_ratio_ci || [])[1], 4)}]`
    : '无 e2e 结果';
}

async function loadActions() {
  const box = $('actions');
  try {
    const data = await getJSON('/api/actions');
    box.innerHTML = '';
    for (const a of data.actions || []) {
      const card = el('div', 'card');
      const fields = (a.params || []).map((p) => `
        <div class="field"><span>${escapeHtml(p.label)}</span>
          <input class="search" style="max-width:220px" data-p="${escapeHtml(p.name)}"
                 placeholder="${escapeHtml(p.placeholder || '')}"
                 value="${escapeHtml(p.default ?? '')}"></div>`).join('');
      card.innerHTML = `<div class="k">${escapeHtml(a.title)}${a.needs_gpu ? ' · 需 GPU' : ''}</div>
        <div class="s" style="margin:6px 0 8px">${escapeHtml(a.desc)}</div>${fields}
        <div style="margin-top:8px"><button class="btn primary" data-act="${escapeHtml(a.name)}">开始</button></div>`;
      card.querySelector('button').onclick = () => submitAction(a.name, card);
      box.appendChild(card);
    }
    if (!(data.actions || []).length) box.innerHTML = '<div class="empty">没有可用动作</div>';
  } catch (e) {
    box.innerHTML = `<div class="empty">无法加载动作（需要后端 serve.py）：${escapeHtml(String(e))}</div>`;
  }
}

async function submitAction(name, card) {
  const params = {};
  card.querySelectorAll('input[data-p]').forEach((i) => { if (i.value.trim()) params[i.dataset.p] = i.value.trim(); });
  $('job-head').textContent = '提交中…';
  try {
    const r = await fetch(`/api/actions/${encodeURIComponent(name)}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params),
    });
    const job = await r.json();
    if (!r.ok) throw new Error(job.error || r.status);
    watchJob(job.id);
  } catch (e) {
    $('job-head').textContent = '提交失败：' + String(e);
  }
}

let jobTimer = null;
function watchJob(id) {
  if (jobTimer) clearInterval(jobTimer);
  const tick = async () => {
    try {
      const j = await getJSON(`/api/jobs/${id}`);
      $('job-head').textContent = `${j.status} · rc=${j.returncode ?? '—'} · ${j.action} · ${id}`;
      $('job-log').textContent = j.log_tail || '（暂无输出）';
      $('job-cancel').disabled = !(j.status === 'running' || j.status === 'queued');
      $('job-cancel').onclick = async () => { await fetch(`/api/jobs/${id}/cancel`, { method: 'POST' }); };
      if (j.status === 'done' || j.status === 'failed' || j.status === 'canceled') {
        clearInterval(jobTimer); jobTimer = null;
      }
    } catch (e) { /* 忽略轮询错误 */ }
  };
  tick();
  jobTimer = setInterval(tick, 2000);
}

function bind() {
  document.querySelectorAll('.tab').forEach((t) => {
    t.onclick = () => {
      state.tab = t.dataset.tab;
      document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('on', x === t));
      $('view-list').classList.toggle('hidden', state.tab !== 'list');
      $('view-report').classList.toggle('hidden', state.tab !== 'report');
      $('view-run').classList.toggle('hidden', state.tab !== 'run');
      document.querySelectorAll('.fchip, #q').forEach((x) => {
        x.style.display = state.tab === 'list' ? '' : 'none';
      });
      if (state.tab === 'run') loadActions();
    };
  });
  document.querySelectorAll('.fchip').forEach((c) => {
    c.onclick = () => {
      state.filter = c.dataset.filter;
      document.querySelectorAll('.fchip').forEach((x) => x.classList.toggle('on', x === c));
      if (state.run) renderTable();
    };
  });
  $('q').oninput = (e) => { state.q = e.target.value; if (state.run) renderTable(); };
  $('btn-reload').onclick = boot;
}

bind();
// 支持 ?tab=run 直达（便于分享/截图）
const m = new URLSearchParams(location.search).get('tab');
if (m) {
  const t = document.querySelector(`.tab[data-tab="${m}"]`);
  if (t) t.click();
}
boot();
