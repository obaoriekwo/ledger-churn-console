const COLORS = {
  existing: '#2f4a3e',
  attrited: '#a23f34',
  brass: '#a9793e',
  grid: '#cdc3a8',
};

Chart.defaults.font.family = "'IBM Plex Mono', monospace";
Chart.defaults.font.size = 11;
Chart.defaults.color = '#5a5442';

const charts = {};

function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

// ── Tabs ────────────────────────────────────────────────────────────────

const loaded = new Set();

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    const name = tab.dataset.tab;
    document.getElementById('panel-' + name).classList.add('active');
    loadPanel(name);
  });
});

function loadPanel(name) {
  if (loaded.has(name)) return;
  loaded.add(name);
  if (name === 'overview') loadOverview();
  if (name === 'models') loadModels();
  if (name === 'demographics') loadDemographics();
  if (name === 'behavior') loadBehavior();
  if (name === 'segments') loadSegments();
}

// ── Masthead ────────────────────────────────────────────────────────────

fetch('/api/overview').then(r => r.json()).then(d => {
  document.querySelector('#masthead-churn .masthead-stat-value').textContent = d.churn_rate + '%';
});

// ── Overview ────────────────────────────────────────────────────────────

function loadOverview() {
  fetch('/api/overview').then(r => r.json()).then(d => {
    const rows = [
      ['Total customers', d.total_customers.toLocaleString()],
      ['Existing customers', d.existing.toLocaleString()],
      ['Attrited customers', d.attrited.toLocaleString()],
      ['Churn rate', d.churn_rate + '%'],
      ['Average age', d.avg_age + ' yrs'],
      ['Average credit limit', '$' + d.avg_credit.toLocaleString()],
      ['Average transaction amount', '$' + d.avg_trans_amt.toLocaleString()],
    ];
    const grid = document.getElementById('overview-grid');
    grid.innerHTML = rows.map(([label, value]) => `
      <div class="ledger-item"><span class="label">${label}</span><span class="value">${value}</span></div>
    `).join('');
  });
}

// ── Models ──────────────────────────────────────────────────────────────

function loadModels() {
  fetch('/api/models').then(r => r.json()).then(data => {
    const tbody = document.querySelector('#models-table tbody');
    tbody.innerHTML = Object.entries(data).map(([name, m]) => `
      <tr>
        <td>${name}</td>
        <td>${(m.accuracy * 100).toFixed(2)}%</td>
        <td>${m.auc.toFixed(4)}</td>
        <td>${m.cv_auc.toFixed(4)}</td>
      </tr>
    `).join('');

    // Model selector for feature importance
    const select = document.getElementById('fi-model-select');
    select.innerHTML = Object.keys(data).map(n => `<option value="${n}">${n}</option>`).join('');
    select.onchange = () => renderFeatureImportance(select.value);
    renderFeatureImportance(select.value);

    // ROC curves — all three models
    destroyChart('roc');
    const ctx = document.getElementById('chart-roc');
    const palette = ['#2f4a3e', '#a9793e', '#a23f34'];
    charts.roc = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: Object.entries(data).map(([name, m], i) => ({
          label: name,
          data: m.roc.fpr.map((f, j) => ({ x: f, y: m.roc.tpr[j] })),
          borderColor: palette[i % palette.length],
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.15,
        })).concat([{
          label: 'Chance',
          data: [{x:0,y:0},{x:1,y:1}],
          borderColor: '#cdc3a8',
          borderDash: [4,4],
          borderWidth: 1,
          pointRadius: 0,
        }])
      },
      options: {
        responsive: true,
        scales: {
          x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'False positive rate' }, grid: { color: COLORS.grid } },
          y: { min: 0, max: 1, title: { display: true, text: 'True positive rate' }, grid: { color: COLORS.grid } },
        },
        plugins: { legend: { position: 'bottom', labels: { boxWidth: 12 } } }
      }
    });
  });
}

function renderFeatureImportance(modelName) {
  fetch('/api/feature_importance?model=' + encodeURIComponent(modelName)).then(r => r.json()).then(fi => {
    const entries = Object.entries(fi).slice(0, 10).reverse();
    destroyChart('fi');
    const ctx = document.getElementById('chart-feature-importance');
    charts.fi = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: entries.map(([k]) => k),
        datasets: [{
          data: entries.map(([, v]) => v),
          backgroundColor: COLORS.brass,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        scales: {
          x: { grid: { color: COLORS.grid } },
          y: { grid: { display: false } },
        },
        plugins: { legend: { display: false } }
      }
    });
  });
}

// ── Demographics ────────────────────────────────────────────────────────

function barPair(ctxId, labels, existingData, attritedData, horizontal=false) {
  destroyChart(ctxId);
  const ctx = document.getElementById(ctxId);
  charts[ctxId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Existing', data: existingData, backgroundColor: COLORS.existing },
        { label: 'Attrited', data: attritedData, backgroundColor: COLORS.attrited },
      ]
    },
    options: {
      indexAxis: horizontal ? 'y' : 'x',
      responsive: true,
      scales: {
        x: { grid: { color: COLORS.grid } },
        y: { grid: { color: COLORS.grid } },
      },
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 12 } } }
    }
  });
}

function loadDemographics() {
  fetch('/api/demographics').then(r => r.json()).then(d => {
    barPair('chart-age', d.age_bins.labels, d.age_bins.existing_vals, d.age_bins.attrited_vals);

    const incomeLabels = Object.keys(d.income.existing);
    barPair('chart-income', incomeLabels,
      incomeLabels.map(k => d.income.existing[k] || 0),
      incomeLabels.map(k => d.income.attrited[k] || 0), true);

    const cardLabels = Object.keys(d.card.existing);
    barPair('chart-card', cardLabels,
      cardLabels.map(k => d.card.existing[k] || 0),
      cardLabels.map(k => d.card.attrited[k] || 0));
  });
}

// ── Behavior ────────────────────────────────────────────────────────────

function loadBehavior() {
  fetch('/api/behavioral').then(r => r.json()).then(d => {
    // Table
    const metrics = ['Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Revolving_Bal',
                      'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Avg_Utilization_Ratio'];
    const tbody = document.querySelector('#behavior-table tbody');
    tbody.innerHTML = metrics.filter(m => d[m]).map(m => {
      const v = d[m];
      return `<tr><td>${m.replace(/_/g,' ')}</td><td>${v.attrited_mean}</td><td>${v.existing_mean}</td><td>${v.attrited_median}</td><td>${v.existing_median}</td></tr>`;
    }).join('');

    // Scatter
    destroyChart('scatter');
    const scatterCtx = document.getElementById('chart-scatter');
    const existingPts = [], attritedPts = [];
    d.scatter.x.forEach((x, i) => {
      const pt = { x, y: d.scatter.y[i] };
      if (d.scatter.label[i] === 'Attrited Customer') attritedPts.push(pt);
      else existingPts.push(pt);
    });
    charts.scatter = new Chart(scatterCtx, {
      type: 'scatter',
      data: {
        datasets: [
          { label: 'Existing', data: existingPts, backgroundColor: COLORS.existing + 'aa', pointRadius: 3 },
          { label: 'Attrited', data: attritedPts, backgroundColor: COLORS.attrited + 'cc', pointRadius: 3 },
        ]
      },
      options: {
        responsive: true,
        scales: {
          x: { title: { display: true, text: 'Transaction count' }, grid: { color: COLORS.grid } },
          y: { title: { display: true, text: 'Transaction amount ($)' }, grid: { color: COLORS.grid } },
        },
        plugins: { legend: { position: 'bottom', labels: { boxWidth: 12 } } }
      }
    });

    // Inactivity distribution
    const months = Array.from(new Set([
      ...Object.keys(d.inactivity_dist.existing),
      ...Object.keys(d.inactivity_dist.attrited)
    ])).sort((a,b) => a - b);
    barPair('chart-inactivity', months,
      months.map(m => d.inactivity_dist.existing[m] || 0),
      months.map(m => d.inactivity_dist.attrited[m] || 0));
  });
}

// ── Segments ────────────────────────────────────────────────────────────

function segChart(ctxId, dataObj) {
  destroyChart(ctxId);
  const ctx = document.getElementById(ctxId);
  const labels = Object.keys(dataObj);
  charts[ctxId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label: 'Churn rate %', data: labels.map(k => dataObj[k]), backgroundColor: COLORS.brass }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      scales: {
        x: { title: { display: true, text: 'Churn rate (%)' }, grid: { color: COLORS.grid } },
        y: { grid: { display: false } },
      },
      plugins: { legend: { display: false } }
    }
  });
}

function loadSegments() {
  fetch('/api/churn_by_segment').then(r => r.json()).then(d => {
    if (d.Income_Category) segChart('chart-seg-income', d.Income_Category);
    if (d.Card_Category) segChart('chart-seg-card', d.Card_Category);
    if (d.Education_Level) segChart('chart-seg-edu', d.Education_Level);
  });
}

// ── Assess form ─────────────────────────────────────────────────────────

document.getElementById('assess-form').addEventListener('submit', e => {
  e.preventDefault();
  const form = e.target;
  const data = {};
  new FormData(form).forEach((v, k) => { data[k] = v; });

  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  })
  .then(r => r.json())
  .then(res => {
    const resultEl = document.getElementById('assess-result');
    resultEl.classList.remove('hidden');

    document.getElementById('predict-label').textContent = res.prediction;
    document.getElementById('churn-prob').textContent = res.churn_probability + '%';
    document.getElementById('retain-prob').textContent = res.retain_probability + '%';

    const stamp = document.getElementById('risk-stamp');
    stamp.classList.remove('low', 'medium', 'high');
    stamp.classList.add(res.risk_level.toLowerCase());
    document.getElementById('risk-label').textContent = res.risk_level + ' risk';
  })
  .catch(() => {
    alert('Prediction failed — check the values and try again.');
  });
});

// ── Init ────────────────────────────────────────────────────────────────

loadPanel('overview');
