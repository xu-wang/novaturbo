/**
 * NovaTurbo Engineering Dashboard
 * Interactive Brayton cycle charts, Pareto front, and design parameter controls.
 * Uses Chart.js for plotting.
 */

import { Chart, registerables } from 'chart.js';
import { zhComponent, zhName, zhState, zhVariable, zhIdentifier, zhUnit, zhMessage } from './zh-cn.js';
Chart.register(...registerables);

// Chart color theme
const C = {
    accent: '#00E5FF',
    accentDim: 'rgba(0,229,255,0.15)',
    danger: '#ef4444',
    warning: '#f59e0b',
    success: '#22c55e',
    blue: '#3b82f6',
    purple: '#a78bfa',
    grid: 'rgba(255,255,255,0.06)',
    gridLabel: '#64748b',
    bg: '#151c2c',
    surface: '#1e293b',
    text: '#e2e8f0',
};

let tsChart = null;
let stationChart = null;
let paretoChart = null;
let currentParams = {
    pr: 3.5, tit: 1100, eta_c: 0.78, eta_t: 0.82, mdot: 0.15, mach: 0,
    nozzle_exit_mm: 55, nozzle_length_mm: 50,
    combustor_outer_mm: 110, combustor_inner_mm: 60, combustor_length_mm: 80,
};
let currentTargets = {
    target_thrust_N: 100,
    target_tsfc_g_kNs: 35,
};
let debounceTimer = null;
let learningPollHandle = null;
let geometry19Schema = null;
let geometry19Values = {};
let geometry19Initialized = false;
let geometry19V22Epoch = 0;
let geometry19V22ActiveJob = null;
const cycleParamKeys = ['pr', 'tit', 'eta_c', 'eta_t', 'mdot', 'mach'];

// ─── Public API ───
export function openDashboard() {
    const overlay = document.getElementById('dashboard-overlay');
    overlay.classList.add('visible');
    const btn = document.getElementById('btn-dashboard');
    if (btn) btn.classList.add('active');
    // Load data on first open
    if (!tsChart) {
        initCharts();
        fetchBrayton();
        fetchPareto();
    }
    if (!geometry19Initialized) initGeometry19Panel();
    refreshLearningStatus();
}

export function closeDashboard() {
    document.getElementById('dashboard-overlay').classList.remove('visible');
    const btn = document.getElementById('btn-dashboard');
    if (btn) btn.classList.remove('active');
}

export function toggleDashboard() {
    const overlay = document.getElementById('dashboard-overlay');
    if (overlay.classList.contains('visible')) closeDashboard();
    else openDashboard();
}

// ─── Chart Setup ───
function chartDefaults() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        plugins: {
            legend: { labels: { color: C.gridLabel, font: { size: 10, family: 'Inter' } } },
            tooltip: {
                backgroundColor: C.bg,
                borderColor: C.accent,
                borderWidth: 1,
                titleFont: { family: 'Inter', size: 11 },
                bodyFont: { family: 'JetBrains Mono', size: 10 },
                titleColor: C.text,
                bodyColor: C.gridLabel,
            },
        },
        scales: {
            x: { ticks: { color: C.gridLabel, font: { size: 10 } }, grid: { color: C.grid } },
            y: { ticks: { color: C.gridLabel, font: { size: 10 } }, grid: { color: C.grid } },
        },
    };
}

function initCharts() {
    // T-s Diagram
    const tsCtx = document.getElementById('chart-ts').getContext('2d');
    tsChart = new Chart(tsCtx, {
        type: 'line',
        data: {
            datasets: [{
                label: '布雷顿循环',
                data: [],
                borderColor: C.accent,
                backgroundColor: C.accentDim,
                fill: true,
                tension: 0.3,
                pointRadius: 5,
                pointBackgroundColor: [C.blue, C.blue, C.success, C.danger, C.warning, C.purple],
                pointBorderColor: '#fff',
                pointBorderWidth: 1,
                borderWidth: 2,
            }],
        },
        options: {
            ...chartDefaults(),
            plugins: {
                ...chartDefaults().plugins,
                legend: { display: false },
                tooltip: {
                    ...chartDefaults().plugins.tooltip,
                    callbacks: {
                        label: ctx => {
                            const p = ctx.raw;
                            return `${p.label}: T=${p.y} K, s=${p.x} J/kg·K`;
                        }
                    }
                },
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: '熵 (J/kg·K)', color: C.gridLabel, font: { size: 10 } },
                    ticks: { color: C.gridLabel, font: { size: 9 } },
                    grid: { color: C.grid },
                },
                y: {
                    title: { display: true, text: '温度 (K)', color: C.gridLabel, font: { size: 10 } },
                    ticks: { color: C.gridLabel, font: { size: 9 } },
                    grid: { color: C.grid },
                },
            },
        },
    });

    // Station bar chart (dual-axis: Temp + Pressure)
    const stCtx = document.getElementById('chart-stations').getContext('2d');
    stationChart = new Chart(stCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                {
                    label: '温度 (K)',
                    data: [],
                    backgroundColor: 'rgba(239,68,68,0.6)',
                    borderColor: C.danger,
                    borderWidth: 1,
                    yAxisID: 'y',
                    borderRadius: 3,
                },
                {
                    label: '压力 (kPa)',
                    data: [],
                    backgroundColor: 'rgba(59,130,246,0.6)',
                    borderColor: C.blue,
                    borderWidth: 1,
                    yAxisID: 'y1',
                    borderRadius: 3,
                },
            ],
        },
        options: {
            ...chartDefaults(),
            scales: {
                x: { ticks: { color: C.gridLabel, font: { size: 9 } }, grid: { display: false } },
                y: {
                    position: 'left',
                    title: { display: true, text: '温度 (K)', color: '#ef4444', font: { size: 9 } },
                    ticks: { color: '#ef4444', font: { size: 9 } },
                    grid: { color: C.grid },
                },
                y1: {
                    position: 'right',
                    title: { display: true, text: '压力 (kPa)', color: '#3b82f6', font: { size: 9 } },
                    ticks: { color: '#3b82f6', font: { size: 9 } },
                    grid: { drawOnChartArea: false },
                },
            },
        },
    });

    // Pareto scatter
    const pCtx = document.getElementById('chart-pareto').getContext('2d');
    paretoChart = new Chart(pCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: '设计（颜色表示推重比）',
                data: [],
                backgroundColor: [],
                pointRadius: 3.5,
                pointHoverRadius: 6,
            }],
        },
        options: {
            ...chartDefaults(),
            plugins: {
                ...chartDefaults().plugins,
                legend: { display: false },
                tooltip: {
                    ...chartDefaults().plugins.tooltip,
                    callbacks: {
                        label: ctx => {
                            const p = ctx.raw;
                            return `推力：${p.x} N | 质量：${p.y} kg | 推重比：${p.tw}`;
                        }
                    }
                },
            },
            scales: {
                x: {
                    title: { display: true, text: '推力 (N)', color: C.gridLabel, font: { size: 10 } },
                    ticks: { color: C.gridLabel, font: { size: 9 } },
                    grid: { color: C.grid },
                },
                y: {
                    title: { display: true, text: '质量 (kg)', color: C.gridLabel, font: { size: 10 } },
                    ticks: { color: C.gridLabel, font: { size: 9 } },
                    grid: { color: C.grid },
                },
            },
        },
    });

    // Bind sliders
    document.querySelectorAll('.dash-slider').forEach(slider => {
        slider.addEventListener('input', onSliderChange);
    });

    // Close button
    document.getElementById('dash-close').addEventListener('click', closeDashboard);
    document.getElementById('btn-apply-model').addEventListener('click', applyModelChanges);
    document.getElementById('btn-save-variant').addEventListener('click', saveVariantSample);
    document.getElementById('btn-run-cfd').addEventListener('click', runCfdCalibration);
    document.getElementById('btn-train-ui').addEventListener('click', startUiTraining);
    document.getElementById('btn-inverse-design').addEventListener('click', runInverseDesign);
    document.getElementById('target-thrust').addEventListener('input', onTargetChange);
    document.getElementById('target-tsfc').addEventListener('input', onTargetChange);
    document.getElementById('btn-run-validation')?.addEventListener('click', runValidation);
    document.getElementById('btn-run-blade')?.addEventListener('click', runBladeAnalysis);
    document.getElementById('btn-run-tbc')?.addEventListener('click', runTBCAnalysis);
    document.getElementById('btn-compare-tbc')?.addEventListener('click', compareTBCCoatings);
    onTargetChange();
    // ESC key
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closeDashboard();
    });

    if (!learningPollHandle) {
        learningPollHandle = setInterval(refreshLearningStatus, 3000);
    }
}

// ─── Data Fetching ───
async function fetchBrayton() {
    const cycleParams = {};
    cycleParamKeys.forEach(k => { cycleParams[k] = currentParams[k]; });
    const qs = new URLSearchParams(cycleParams).toString();
    try {
        const resp = await fetch('/api/brayton/compute?' + qs);
        const data = await resp.json();
        updateTsChart(data.ts_diagram);
        updateStationChart(data.labels, data.temperatures, data.pressures);
        updateKpis(data.performance, data.warnings, data.valid);
    } catch (e) {
        console.warn('Brayton fetch failed:', e);
    }
}

async function fetchPareto() {
    try {
        const resp = await fetch('/api/pareto');
        const data = await resp.json();
        updateParetoChart(data.points, data.stats);
    } catch (e) {
        console.warn('Pareto fetch failed:', e);
    }
}

// ─── Chart Updates ───
function updateTsChart(tsPoints) {
    if (!tsChart || !tsPoints) return;
    // Close the cycle loop: add first point at end
    const pts = tsPoints.map(p => ({ x: p.s, y: p.T, label: zhName(p.label) }));
    if (pts.length > 1) pts.push({ ...pts[0] });

    tsChart.data.datasets[0].data = pts;
    tsChart.data.datasets[0].pointBackgroundColor =
        pts.map((_, i) => [C.blue, C.blue, C.success, C.danger, C.warning, C.purple, C.blue][i] || C.accent);
    tsChart.update();
}

function updateStationChart(labels, temps, pressures) {
    if (!stationChart) return;
    stationChart.data.labels = labels.map(zhName);
    stationChart.data.datasets[0].data = temps;
    stationChart.data.datasets[1].data = pressures;
    stationChart.update();
}

function updateKpis(perf, warnings, valid) {
    if (!perf) return;
    const set = (id, val, unit) => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = `<span class="kpi-val">${val}</span><span class="kpi-unit">${unit}</span>`;
    };
    set('kpi-thrust', perf.thrust_N, 'N');
    set('kpi-tsfc', perf.tsfc_g_kNs, 'g/kN·s');
    set('kpi-eff', perf.thermal_eff, '%');
    set('kpi-exhaust', perf.exhaust_vel, 'm/s');
    set('kpi-fuel', perf.fuel_flow_g_hr, 'g/hr');
    set('kpi-afr', perf.air_fuel_ratio, ':1');

    // Status indicator
    const statusEl = document.getElementById('dash-status');
    if (statusEl) {
        if (!valid) {
            statusEl.textContent = '⚠ ' + zhMessage(warnings[0] || '无效设计');
            statusEl.className = 'dash-status warn';
        } else if (warnings.length > 0) {
            statusEl.textContent = '⚡ ' + zhMessage(warnings[0]);
            statusEl.className = 'dash-status caution';
        } else {
            statusEl.textContent = '✓ 设计满足限制';
            statusEl.className = 'dash-status ok';
        }
    }
}

function setDashStatus(text, cls) {
    const statusEl = document.getElementById('dash-status');
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.className = 'dash-status ' + cls;
}

function twColor(tw, minTw, maxTw) {
    const t = Math.max(0, Math.min(1, (tw - minTw) / (maxTw - minTw + 0.01)));
    // Blue (low T/W) → Cyan → Green → Yellow → Red (high T/W)
    if (t < 0.25) return `rgba(59,130,246,0.7)`;
    if (t < 0.5)  return `rgba(0,229,255,0.7)`;
    if (t < 0.75) return `rgba(34,197,94,0.7)`;
    return `rgba(245,158,11,0.7)`;
}

function updateParetoChart(points, stats) {
    if (!paretoChart || !points) return;
    const minTw = stats.tw_range[0];
    const maxTw = stats.tw_range[1];

    paretoChart.data.datasets[0].data = points.map(p => ({
        x: p.thrust, y: p.mass, tw: p.tw,
    }));
    paretoChart.data.datasets[0].backgroundColor = points.map(p => twColor(p.tw, minTw, maxTw));
    paretoChart.update();

    // Update Pareto legend counts
    const infoEl = document.getElementById('pareto-info');
    if (infoEl) {
        infoEl.textContent = `${points.length} 个采样设计 | 推重比范围 ${stats.tw_range[0]} - ${stats.tw_range[1]}`;
    }
}

async function saveVariantSample() {
    const btn = document.getElementById('btn-save-variant');
    if (!btn) return;
    const old = btn.textContent;
    btn.disabled = true;
    btn.textContent = '保存中…';
    try {
        const resp = await fetch('/api/learning/add_variant', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentParams),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) throw new Error(data.error || '保存失败');
        setDashStatus('✓ 设计样本已保存', 'ok');
        const info = document.getElementById('learning-status');
        if (info) {
            info.textContent = `已保存样本：${data.variant_count} | 最新推力 ${data.sample.thrust_N} N`;
        }
    } catch (err) {
        setDashStatus('⚠ ' + err.message, 'warn');
    } finally {
        btn.disabled = false;
        btn.textContent = old;
    }
}

async function startUiTraining() {
    const btn = document.getElementById('btn-train-ui');
    if (!btn) return;
    const old = btn.textContent;
    btn.disabled = true;
    btn.textContent = '启动中…';
    try {
        const useCfd = document.getElementById('chk-use-cfd')?.checked ?? false;
        const resp = await fetch('/api/learning/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ epochs: 80, use_cfd: useCfd }),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) throw new Error(data.error || '训练启动失败');
        setDashStatus('⏳ 界面样本训练已开始', 'caution');
        const info = document.getElementById('learning-status');
        if (info) {
            const mode = useCfd ? '经 CFD 标定' : '仅物理计算';
            info.textContent = `正在训练（${mode}）…样本：${data.variant_count}，轮次：${data.epochs}`;
        }
    } catch (err) {
        setDashStatus('⚠ ' + err.message, 'warn');
    } finally {
        btn.disabled = false;
        btn.textContent = old;
    }
}

async function refreshLearningStatus() {
    try {
        const resp = await fetch('/api/learning/status');
        const data = await resp.json();
        const info = document.getElementById('learning-status');
        if (info) {
            const cfd = data.cfd || {};
            let trainText = `已保存界面样本：${data.variant_count}`;
            if (data.state === 'running') {
                trainText = `正在训练…样本：${data.variant_count}`;
            } else if (data.state === 'done') {
                trainText = `训练完成。样本：${data.variant_count}`;
            } else if (data.state === 'failed') {
                trainText = `训练失败：${zhMessage(data.message)}`;
            }

            let cfdText = '';
            if (cfd.state === 'running') {
                cfdText = ` | CFD 正在运行（${cfd.solver || '求解器'}），已处理 ${cfd.processed || 0} 个案例`;
            } else if (cfd.state === 'done') {
                cfdText = ` | CFD 已就绪（${cfd.sample_count || 0} 个样本）`;
            } else if (cfd.state === 'failed') {
                cfdText = ` | CFD 失败：${zhMessage(cfd.message)}`;
            } else if (data.has_cfd_calibration) {
                cfdText = ' | CFD 标定可用';
            }
            info.textContent = trainText + cfdText;
        }
    } catch (_e) {
        // Keep UI silent if status polling fails intermittently.
    }
}

function onTargetChange() {
    const thrustEl = document.getElementById('target-thrust');
    const tsfcEl = document.getElementById('target-tsfc');
    currentTargets.target_thrust_N = parseFloat(thrustEl.value);
    currentTargets.target_tsfc_g_kNs = parseFloat(tsfcEl.value);
    document.getElementById('target-thrust-val').textContent = String(Math.round(currentTargets.target_thrust_N));
    document.getElementById('target-tsfc-val').textContent = String(Math.round(currentTargets.target_tsfc_g_kNs));
}

function setUiParamValue(paramKey, value) {
    if (value === undefined || value === null) return;
    currentParams[paramKey] = value;
    const slider = document.querySelector(`.dash-slider[data-param="${paramKey}"]`);
    if (slider) slider.value = String(value);
    const label = document.getElementById('val-' + paramKey);
    if (label) {
        const v = parseFloat(value);
        label.textContent = Number.isInteger(v) ? String(v) : v.toFixed(2).replace(/\.00$/, '');
    }
}

async function runCfdCalibration() {
    const btn = document.getElementById('btn-run-cfd');
    if (!btn) return;
    const solver = document.getElementById('cfd-solver')?.value || 'openfoam';
    const old = btn.textContent;
    btn.disabled = true;
    btn.textContent = '正在启动 CFD…';
    try {
        const resp = await fetch('/api/learning/cfd/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ solver, max_cases: 5, timeout_s: 900 }),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) throw new Error(data.error || 'CFD 标定启动失败');
        setDashStatus(`⏳ ${solver.toUpperCase()} 标定已启动`, 'caution');
        const info = document.getElementById('learning-status');
        if (info) info.textContent = `正在使用 ${solver} 进行 CFD 标定…`;
    } catch (err) {
        setDashStatus('⚠ ' + err.message, 'warn');
    } finally {
        btn.disabled = false;
        btn.textContent = old;
    }
}

async function runInverseDesign() {
    const btn = document.getElementById('btn-inverse-design');
    if (!btn) return;
    onTargetChange();

    const old = btn.textContent;
    btn.disabled = true;
    btn.textContent = '搜索中…';
    try {
        const resp = await fetch('/api/inverse_design', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_thrust_N: currentTargets.target_thrust_N,
                target_tsfc_g_kNs: currentTargets.target_tsfc_g_kNs,
                n_samples: 3000,
                top_k: 5,
            }),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) throw new Error(data.error || '逆向设计失败');

        if (data.best_ui_params) {
            Object.entries(data.best_ui_params).forEach(([k, v]) => setUiParamValue(k, v));
            fetchBrayton();
        }

        const inv = document.getElementById('inverse-status');
        if (inv && data.best && data.best.performance) {
            const p = data.best.performance;
            inv.textContent = `最佳匹配：推力 ${p.thrust_N.toFixed(1)} N，单位推力燃油消耗率 ${p.tsfc_g_kNs.toFixed(1)} g/kN·s`;
        }
        setDashStatus('✓ 逆向设计建议已就绪', 'ok');
    } catch (err) {
        setDashStatus('⚠ ' + err.message, 'warn');
    } finally {
        btn.disabled = false;
        btn.textContent = old;
    }
}

// ─── Canonical Geometry19 workflow ───
const geometry19GroupMeta = {
    compressor: { label: '压气机', color: '#4FC3F7' },
    combustor: { label: '燃烧室', color: '#FF8A65' },
    turbine: { label: '涡轮', color: '#E57373' },
    nozzle: { label: '喷管', color: '#FFD54F' },
    operating_point: { label: '运行点', color: '#A78BFA' },
    lattice: { label: '点阵', color: '#81C784' },
};

function geometry19DisplayScale(variable) {
    return variable.display_unit === 'mm' ? 1000 : 1;
}

function geometry19DisplayValue(variable, value) {
    return Number(value) * geometry19DisplayScale(variable);
}

function geometry19Step(variable) {
    if (String(variable.dtype).startsWith('int')) return 1;
    const scale = geometry19DisplayScale(variable);
    const span = (Number(variable.bounds[1]) - Number(variable.bounds[0])) * scale;
    return Number((span / 200).toPrecision(4));
}

function geometry19HumanName(name) {
    return zhVariable(name);
}

function setGeometry19Message(text, kind = '') {
    const el = document.getElementById('geometry19-message');
    if (!el) return;
    el.textContent = text;
    el.className = 'geometry19-message' + (kind ? ' ' + kind : '');
}

async function initGeometry19Panel() {
    geometry19Initialized = true;
    document.getElementById('geometry19-model-mode')?.addEventListener('change', syncGeometry19ModelMode);
    document.getElementById('geometry19-reset')?.addEventListener('click', resetGeometry19Reference);
    document.getElementById('geometry19-load-json')?.addEventListener('click', loadGeometry19Json);
    document.getElementById('geometry19-v22-generate')?.addEventListener('click', submitGeometry19V22Target);
    document.getElementById('geometry19-solve')?.addEventListener('click', () => submitGeometry19(false));
    document.getElementById('geometry19-regenerate')?.addEventListener('click', () => submitGeometry19(true));
    try {
        const resp = await fetch('/api/geometry19/schema');
        const data = await resp.json();
        if (!resp.ok || data.error) throw new Error(data.error || '参数结构请求失败');
        geometry19Schema = data;
        geometry19Values = Object.fromEntries(
            data.schema.design_variables.map(variable => [variable.name, variable.reference_value])
        );
        renderGeometry19Controls();
        document.getElementById('geometry19-json').value = JSON.stringify(geometry19Values, null, 2);
        syncGeometry19ModelMode();
        setGeometry19Message(`就绪 · ${data.canonical_names.length} 个标准变量 · 参数结构 ${data.schema_hash.slice(0, 12)}`, 'ok');
    } catch (error) {
        geometry19Initialized = false;
        setGeometry19Message('参数结构不可用：' + error.message, 'warn');
    }
}

function syncGeometry19ModelMode() {
    const mode = document.getElementById('geometry19-model-mode')?.value;
    const completedV22 = mode === 'v2_2_new';
    geometry19V22Epoch += 1;
    geometry19V22ActiveJob = null;
    clearGeometry19Display();
    const subtitle = document.getElementById('geometry19-dashboard-subtitle');
    if (subtitle) subtitle.textContent = completedV22
        ? 'Geometry19 · V2.2 开发版物理计算 · 网格诊断'
        : 'Geometry19 · V2.1 物理计算 · 网格诊断';
    document.getElementById('dashboard-overlay').classList.toggle('v22-mode', completedV22);
    document.getElementById('geometry19-solve').disabled = completedV22;
    document.getElementById('geometry19-regenerate').disabled = completedV22;
    document.getElementById('geometry19-profile').disabled = completedV22;
    document.getElementById('geometry19-reset').disabled = completedV22;
    document.getElementById('geometry19-load-json').disabled = completedV22;
    document.getElementById('geometry19-v22-targets').hidden = !completedV22;
    document.getElementById('geometry19-v22-generate').disabled = !completedV22;
    document.getElementById('geometry19-groups').hidden = completedV22;
    document.querySelector('.geometry19-json-panel').hidden = completedV22;
    document.querySelectorAll('#geometry19-groups input, #geometry19-groups button').forEach(
        element => { element.disabled = completedV22; }
    );
    if (completedV22) document.getElementById('geometry19-json').value = '';
    else syncGeometry19Json();
    for (const id of ['g19-target-thrust', 'g19-target-tsfc', 'g19-target-mass']) {
        document.getElementById(id).disabled = false;
    }
    if (geometry19Initialized) {
        setGeometry19Message(
            completedV22
                ? 'V2.2 新设计模式：输入三个性能目标后生成。'
                : 'V2.1 交互模式：求解或生成新设计。',
            completedV22 ? 'busy' : 'ok'
        );
    }
    window.dispatchEvent(new CustomEvent("geometry19-mode-synced", { detail: { mode } }));
}

function clearGeometry19Display() {
    if (document.getElementById('geometry19-model-mode')?.value === 'v2_2_new') {
        document.getElementById('geometry19-groups').hidden = true;
        document.querySelector('.geometry19-json-panel').hidden = true;
        document.getElementById('geometry19-json').value = '';
    }
    for (const id of ['g19-thrust', 'g19-tsfc', 'g19-mass', 'g19-solver',
        'g19-constraints', 'g19-mesh']) {
        const element = document.getElementById(id);
        if (element) element.textContent = '—';
    }
    for (const id of ['g19-solver-detail', 'g19-constraint-detail', 'g19-mesh-detail',
        'geometry19-violations', 'geometry19-mesh-diagnostics', 'geometry19-provenance']) {
        const element = document.getElementById(id);
        if (element) element.textContent = '';
    }
    window.dispatchEvent(new CustomEvent('geometry19-clear-model'));
}

async function submitGeometry19V22Target() {
    if (document.getElementById('geometry19-model-mode').value !== 'v2_2_new') return;
    if (geometry19V22ActiveJob) return;
    const target = {
        thrust_N: Number(document.getElementById('g19-target-thrust').value),
        tsfc_kg_N_s: Number(document.getElementById('g19-target-tsfc').value),
        mass_kg: Number(document.getElementById('g19-target-mass').value),
    };
    if (['g19-target-thrust', 'g19-target-tsfc', 'g19-target-mass'].some(
        id => !document.getElementById(id).value.trim()
    ) || Object.values(target).some(value => !Number.isFinite(value) || value <= 0)) {
        setGeometry19Message('三个目标都必须是有限正数。', 'warn');
        return;
    }
    const epoch = ++geometry19V22Epoch;
    clearGeometry19Display();
    const button = document.getElementById('geometry19-v22-generate');
    button.disabled = true;
    const targetInputs = ['g19-target-thrust', 'g19-target-tsfc', 'g19-target-mass']
        .map(id => document.getElementById(id));
    targetInputs.forEach(input => { input.disabled = true; });
    try {
        const response = await fetch('/api/geometry19/v2-2/jobs', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(target),
        });
        const started = await response.json();
        if (!response.ok || started.error) throw new Error(started.error || '无法启动 V2.2 任务');
        if (epoch !== geometry19V22Epoch) return;
        geometry19V22ActiveJob = started.job_id;
        setGeometry19Message('V2.2 正在运行…', 'busy');
        while (epoch === geometry19V22Epoch) {
            await new Promise(resolve => setTimeout(resolve, 2000));
            if (epoch !== geometry19V22Epoch) return;
            const statusResponse = await fetch(`/api/geometry19/v2-2/jobs/${started.job_id}`);
            const status = await statusResponse.json();
            if (!statusResponse.ok || status.error) throw new Error(status.error || 'V2.2 任务失败');
            if (status.state === 'failed') throw new Error(status.error || 'V2.2 任务失败');
            if (status.state === 'complete') break;
            setGeometry19Message(`V2.2 ${zhState(status.state)}…`, 'busy');
        }
        if (epoch !== geometry19V22Epoch) return;
        const quality = document.getElementById('geometry19-quality').value;
        const resultResponse = await fetch(
            `/api/geometry19/v2-2/jobs/${started.job_id}/result?quality=${encodeURIComponent(quality)}`
        );
        const data = await resultResponse.json();
        if (epoch !== geometry19V22Epoch) return;
        if (!resultResponse.ok || data.error) throw new Error(data.error || 'V2.2 结果不可用');
        geometry19Schema.schema.design_variables.forEach(variable => {
            setGeometry19Variable(variable, Number(data.geometry19[variable.name]));
        });
        document.getElementById('geometry19-groups').hidden = false;
        document.querySelector('.geometry19-json-panel').hidden = false;
        updateGeometry19Results(data, true);
        window.dispatchEvent(new CustomEvent('geometry19-model-ready', {
            detail: { component_urls: data.component_urls, design_id: data.design_id,
                simulation: null, overlay_parameters: null,
                violated_components: geometry19ViolationComponents(
                    data.constraints?.violated_constraints || []), },
        }));
        setGeometry19Message(`新的 V2.2 设计 ${data.design_id.slice(0, 12)} 已加载。`, 'ok');
    } catch (error) {
        if (epoch === geometry19V22Epoch) setGeometry19Message(error.message, 'warn');
    } finally {
        if (epoch === geometry19V22Epoch) {
            geometry19V22ActiveJob = null;
            button.disabled = false;
        }
        targetInputs.forEach(input => { input.disabled = false; });
    }
}

function renderGeometry19Controls() {
    const container = document.getElementById('geometry19-groups');
    if (!container || !geometry19Schema) return;
    container.innerHTML = '';
    const variables = geometry19Schema.schema.design_variables;
    for (const [groupName, meta] of Object.entries(geometry19GroupMeta)) {
        const members = variables.filter(variable => variable.component === groupName);
        if (!members.length) continue;
        const group = document.createElement('section');
        group.className = 'geometry19-group';
        group.dataset.component = groupName;
        group.style.setProperty('--group-color', meta.color);
        const title = document.createElement('div');
        title.className = 'geometry19-group-title';
        title.innerHTML = `<span>${meta.label}</span><small>${members.length}D</small>`;
        group.appendChild(title);
        members.forEach(variable => group.appendChild(makeGeometry19Control(variable)));
        container.appendChild(group);
    }
}

function makeGeometry19Control(variable) {
    const scale = geometry19DisplayScale(variable);
    const lower = Number(variable.bounds[0]) * scale;
    const upper = Number(variable.bounds[1]) * scale;
    const value = geometry19DisplayValue(variable, geometry19Values[variable.name]);
    const step = geometry19Step(variable);
    const row = document.createElement('div');
    row.className = 'geometry19-param';
    row.dataset.variable = variable.name;

    const label = document.createElement('label');
    label.title = `${zhVariable(variable.name)} · 范围 ${lower}–${upper} ${zhUnit(variable.display_unit || variable.unit)}`;
    const channel = variable.component === 'operating_point' ? '覆盖层' : (variable.component === 'lattice' ? '点阵' : '实体');
    const type = String(variable.dtype).startsWith('int') ? '离散' : '连续';
    label.innerHTML = `${geometry19HumanName(variable.name)}<code>${variable.symbol} · ${zhUnit(variable.display_unit || variable.unit)} · ${type} · ${channel}</code><code>范围 ${lower}–${upper} · 参考值 ${value}</code>`;

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.className = 'dash-slider geometry19-slider';
    slider.min = String(lower);
    slider.max = String(upper);
    slider.step = String(step);
    slider.value = String(value);
    slider.dataset.variable = variable.name;

    const number = document.createElement('input');
    number.type = 'number';
    number.className = 'geometry19-number';
    number.min = String(lower);
    number.max = String(upper);
    number.step = String(step);
    number.value = String(value);
    number.dataset.variable = variable.name;
    number.setAttribute('aria-label', zhVariable(variable.name));

    const shortcuts = document.createElement('div');
    shortcuts.className = 'geometry19-shortcuts';
    [['下', variable.bounds[0]], ['参', variable.reference_value], ['上', variable.bounds[1]]].forEach(([text, siValue]) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = text;
        button.title = { 下: '下限', 参: '参考值', 上: '上限' }[text];
        button.addEventListener('click', () => setGeometry19Variable(variable, Number(siValue)));
        shortcuts.appendChild(button);
    });

    slider.addEventListener('input', () => {
        number.value = slider.value;
        geometry19Values[variable.name] = Number(slider.value) / scale;
        syncGeometry19Json();
    });
    number.addEventListener('input', () => {
        slider.value = number.value;
        geometry19Values[variable.name] = Number(number.value) / scale;
        syncGeometry19Json();
    });
    row.append(label, slider, number, shortcuts);
    return row;
}

function setGeometry19Variable(variable, siValue) {
    geometry19Values[variable.name] = siValue;
    const row = document.querySelector(`.geometry19-param[data-variable="${variable.name}"]`);
    if (!row) return;
    const display = geometry19DisplayValue(variable, siValue);
    row.querySelector('.geometry19-slider').value = String(display);
    row.querySelector('.geometry19-number').value = String(display);
    syncGeometry19Json();
}

function syncGeometry19Json() {
    const textarea = document.getElementById('geometry19-json');
    if (textarea && document.activeElement !== textarea) {
        textarea.value = JSON.stringify(geometry19Values, null, 2);
    }
}

function resetGeometry19Reference() {
    if (!geometry19Schema) return;
    geometry19Schema.schema.design_variables.forEach(variable => {
        setGeometry19Variable(variable, Number(variable.reference_value));
    });
    clearGeometry19Warnings();
    setGeometry19Message('已恢复参考设计。', 'ok');
}

function loadGeometry19Json() {
    if (!geometry19Schema) return;
    try {
        const parsed = JSON.parse(document.getElementById('geometry19-json').value);
        const source = parsed.geometry19 || parsed.design || parsed.best?.geometry19 || parsed.best?.design || parsed;
        const expected = geometry19Schema.canonical_names;
        const missing = expected.filter(name => !(name in source));
        if (missing.length) throw new Error('缺少：' + missing.join(', '));
        geometry19Schema.schema.design_variables.forEach(variable => {
            setGeometry19Variable(variable, Number(source[variable.name]));
        });
        validateGeometry19Inputs();
        setGeometry19Message('JSON 已载入全部 19 个标准参数控件。', 'ok');
    } catch (error) {
        setGeometry19Message('无法加载 JSON：' + error.message, 'warn');
    }
}

function validateGeometry19Inputs() {
    if (!geometry19Schema) throw new Error('参数结构尚未加载');
    for (const variable of geometry19Schema.schema.design_variables) {
        const value = Number(geometry19Values[variable.name]);
        const [lower, upper] = variable.bounds.map(Number);
        if (!Number.isFinite(value)) throw new Error(`${zhVariable(variable.name)} 必须是有限数值`);
        if (value < lower || value > upper) {
            throw new Error(`${zhVariable(variable.name)} 超出范围 [${lower}, ${upper}] ${variable.unit}`);
        }
        if (String(variable.dtype).startsWith('int') && !Number.isInteger(value)) {
            throw new Error(`${zhVariable(variable.name)} 必须是整数`);
        }
    }
    return { ...geometry19Values };
}

async function submitGeometry19(regenerate) {
    if (document.getElementById('geometry19-model-mode')?.value !== 'v2_1_interactive') return;
    const epoch = geometry19V22Epoch;
    const button = document.getElementById(regenerate ? 'geometry19-regenerate' : 'geometry19-solve');
    const oldText = button.textContent;
    try {
        const design = validateGeometry19Inputs();
        button.disabled = true;
        button.textContent = regenerate ? '正在生成网格…' : '正在求解…';
        setGeometry19Message(regenerate ? '正在求解 V2.1 并生成独立网格产物…' : '正在求解 V2.1 并评估约束…', 'busy');
        const payload = {
            geometry19: design,
            constraint_profile: document.getElementById('geometry19-profile').value,
            mesh_quality: document.getElementById('geometry19-quality').value,
        };
        const endpoint = regenerate ? '/api/geometry19/regenerate' : '/api/geometry19/solve';
        const resp = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (epoch !== geometry19V22Epoch ||
            document.getElementById('geometry19-model-mode').value !== 'v2_1_interactive') return;
        if (!resp.ok || data.error) throw new Error(data.error || 'Geometry19 请求失败');
        updateGeometry19Results(data, regenerate);
        if (regenerate) {
            window.dispatchEvent(new CustomEvent('geometry19-model-ready', {
                detail: {
                    component_urls: data.component_urls,
                    design_id: data.design_id,
                    simulation: data.simulation,
                    overlay_parameters: data.overlay_parameters,
                    violated_components: geometry19ViolationComponents(data.constraints?.violated_constraints || []),
                },
            }));
        }
    } catch (error) {
        if (epoch === geometry19V22Epoch) setGeometry19Message(error.message, 'warn');
    } finally {
        button.disabled = document.getElementById('geometry19-model-mode')?.value !== 'v2_1_interactive';
        button.textContent = oldText;
    }
}

function setGeometry19Metric(id, value, digits = 4) {
    const el = document.getElementById(id);
    if (!el) return;
    const numeric = Number(value);
    el.textContent = Number.isFinite(numeric) ? numeric.toPrecision(digits) : '—';
}

function updateGeometry19Results(data, regenerated) {
    const performance = data.physics?.performance || {};
    const solver = data.solver || {};
    const constraints = data.constraints || {};
    setGeometry19Metric('g19-thrust', performance.thrust_N, 5);
    setGeometry19Metric('g19-tsfc', performance.tsfc_kg_N_s, 4);
    setGeometry19Metric('g19-mass', performance.mass_kg, 5);
    document.getElementById('g19-solver').textContent = solver.converged ? '已收敛' : '失败';
    const workingPoints = data.physics?.operating_point?.flow_root_count
        ?? (Array.isArray(data.physics?.operating_points) ? data.physics.operating_points.length : 1);
    document.getElementById('g19-solver-detail').textContent = `${workingPoints} 个运行点 · ${solver.iterations ?? 0} 次迭代 · 残差 ${Number(solver.flow_residual).toExponential(2)}`;
    const solverCard = document.getElementById('g19-solver').parentElement;
    solverCard.classList.toggle('pass', Boolean(solver.converged));
    solverCard.classList.toggle('fail', !solver.converged);

    document.getElementById('g19-constraints').textContent = constraints.accepted ? '已接受' : '已拒绝';
    document.getElementById('g19-constraint-detail').textContent = `${constraints.violated_constraints?.length || 0} 个违反项`;
    const constraintCard = document.getElementById('g19-constraints').parentElement;
    constraintCard.classList.toggle('pass', Boolean(constraints.accepted));
    constraintCard.classList.toggle('fail', !constraints.accepted);
    renderGeometry19Violations(constraints.violated_constraints || []);

    if (data.mesh_diagnostics) renderGeometry19MeshDiagnostics(data.mesh_diagnostics);
    else {
        document.getElementById('g19-mesh').textContent = '尚未生成';
        document.getElementById('g19-mesh-detail').textContent = '仅求解';
    }
    const idText = data.design_id ? ` · 设计 ${data.design_id.slice(0, 12)}` : '';
    setGeometry19Message(`${regenerated ? '已生成并发送至三维查看器' : '求解完成'}${idText}`, constraints.accepted ? 'ok' : 'busy');
    document.getElementById('geometry19-provenance').textContent = [
        data.geometry_hash && `几何 ${data.geometry_hash.slice(0, 12)}`,
        data.lattice_hash && `点阵 ${data.lattice_hash.slice(0, 12)}`,
        data.schema_hash && `参数结构 ${data.schema_hash.slice(0, 12)}`,
        data.calibration_hash && `标定 ${String(data.calibration_hash).slice(0, 12)}`,
        constraints.constraint_profile_hash && `约束配置 ${constraints.constraint_profile_hash.slice(0, 12)}`,
    ].filter(Boolean).join(' · ');
}

function clearGeometry19Warnings() {
    document.querySelectorAll('.geometry19-group').forEach(group => group.classList.remove('constraint-warning'));
    document.getElementById('geometry19-violations').textContent = '';
}

function geometry19ViolationComponents(violations) {
    const found = new Set();
    for (const violation of violations) {
        const text = String(violation).toLowerCase();
        for (const component of Object.keys(geometry19GroupMeta)) {
            const token = component === 'operating_point' ? 'operat' : component;
            if (text.includes(token)) found.add(component);
        }
    }
    return [...found];
}

function renderGeometry19Violations(violations) {
    clearGeometry19Warnings();
    const target = document.getElementById('geometry19-violations');
    if (!violations.length) {
        target.textContent = '约束诊断：没有违反项。';
        return;
    }
    target.innerHTML = '<span class="failure">约束诊断：' + violations.map(zhIdentifier).join(' · ') + '</span>';
    geometry19ViolationComponents(violations).forEach(component => {
        document.querySelector(`.geometry19-group[data-component="${component}"]`)?.classList.add('constraint-warning');
    });
}

function renderGeometry19MeshDiagnostics(diagnostics) {
    const components = Object.values(diagnostics.component_diagnostics || {});
    const passed = Boolean(diagnostics.passed);
    const mesh = document.getElementById('g19-mesh');
    mesh.textContent = passed ? '通过' : '失败';
    mesh.parentElement.classList.toggle('pass', passed);
    mesh.parentElement.classList.toggle('fail', !passed);
    document.getElementById('g19-mesh-detail').textContent = `${components.length} 个组件 · ${diagnostics.quality === 'preview_mesh' ? '预览网格' : diagnostics.quality === 'export_mesh' ? '导出网格' : ''}`;
    const failures = components.filter(item => !item.passed).map(item => item.name);
    const vertexCount = components.reduce((sum, item) => sum + Number(item.vertex_count || 0), 0);
    const faceCount = components.reduce((sum, item) => sum + Number(item.face_count || 0), 0);
    document.getElementById('geometry19-mesh-diagnostics').innerHTML =
        `网格诊断：${vertexCount.toLocaleString()} 个顶点 · ${faceCount.toLocaleString()} 个面片` +
        (failures.length ? ` · <span class="failure">失败：${failures.map(zhIdentifier).join(', ')}</span>` : ' · 所有必需检查均通过');
}

// ─── Slider Handling ───
function onSliderChange(e) {
    const slider = e.target;
    const key = slider.dataset.param;
    const val = parseFloat(slider.value);
    currentParams[key] = val;

    // Update label
    const label = document.getElementById('val-' + key);
    if (label) label.textContent = Number.isInteger(val) ? val : val.toFixed(2).replace(/\.00$/, '');

    // Update Brayton charts only for cycle parameters
    if (cycleParamKeys.includes(key)) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => fetchBrayton(), 200);
    }
}

async function applyModelChanges() {
    const btn = document.getElementById('btn-apply-model');
    if (!btn) return;

    const prevText = btn.textContent;
    btn.disabled = true;
    btn.textContent = '应用中…';

    try {
        const resp = await fetch('/api/model/regenerate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentParams),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) {
            throw new Error(data.error || '模型重新生成失败');
        }

        const statusEl = document.getElementById('dash-status');
        if (statusEl) {
            statusEl.textContent = '✓ 三维模型已重新生成';
            statusEl.className = 'dash-status ok';
        }

        // Ensure KPIs reflect the regenerated configuration before reload.
        if (data.performance) {
            updateKpis(
                {
                    thrust_N: data.performance.thrust_N,
                    tsfc_g_kNs: data.performance.tsfc_g_kNs,
                    thermal_eff: data.performance.thermal_eff,
                    exhaust_vel: '-',
                    fuel_flow_g_hr: '-',
                    air_fuel_ratio: '-',
                },
                data.warnings || [],
                true
            );
        }

        setTimeout(() => window.location.reload(), 300);
    } catch (err) {
        const statusEl = document.getElementById('dash-status');
        if (statusEl) {
            statusEl.textContent = '⚠ ' + err.message;
            statusEl.className = 'dash-status warn';
        }
        btn.disabled = false;
        btn.textContent = prevText;
    }
}

// ─── Brayton Validation ───
let bladeCpChart = null;

async function runValidation() {
    const btn = document.getElementById('btn-run-validation');
    const out = document.getElementById('validation-results');
    if (!btn || !out) return;
    btn.disabled = true;
    btn.textContent = '运行中…';
    out.textContent = '正在使用公开发动机数据验证…';
    try {
        const resp = await fetch('/api/validation');
        const data = await resp.json();
        let txt = `评级：${data.grade}  |  发动机：${data.valid_count}/${data.engine_count}\n`;
        txt += `平均推力误差：${data.avg_thrust_error_pct}%  |  平均单位推力燃油消耗率误差：${data.avg_tsfc_error_pct}%\n`;
        txt += '─'.repeat(60) + '\n';
        for (const eng of data.engines) {
            const st = eng.valid ? '✓' : '✗';
            txt += `\n[${st}] ${eng.name} (${eng.source})\n`;
            txt += `  推力：${eng.predicted.thrust_N} N  对比  ${eng.published.thrust_N} N  (${eng.errors.thrust_pct > 0 ? '+' : ''}${eng.errors.thrust_pct}%)\n`;
            txt += `  排气温度：${eng.predicted.exhaust_temp_K} K  对比  ${eng.published.exhaust_temp_K} K  (${eng.errors.exhaust_temp_pct > 0 ? '+' : ''}${eng.errors.exhaust_temp_pct}%)\n`;
            txt += `  单位推力燃油消耗率：${eng.predicted.tsfc_kg_kgf_h} kg/kgf·h  对比  ${eng.published.tsfc_kg_kgf_h} kg/kgf·h  (${eng.errors.tsfc_pct > 0 ? '+' : ''}${eng.errors.tsfc_pct}%)\n`;
            for (const n of eng.notes) txt += `  ⚠ ${zhMessage(n)}\n`;
        }
        out.textContent = txt;
        setDashStatus(`验证评级：${data.grade}`, data.grade === 'A' || data.grade === 'B' ? 'ok' : 'warn');
    } catch (err) {
        out.textContent = '错误：' + err.message;
    } finally {
        btn.disabled = false;
        btn.textContent = '运行布雷顿循环验证';
    }
}

// ─── 叶片型线分析 ───
async function runBladeAnalysis() {
    const btn = document.getElementById('btn-run-blade');
    const metricsEl = document.getElementById('blade-metrics');
    if (!btn) return;
    btn.disabled = true;
    btn.textContent = '分析中…';
    try {
        const resp = await fetch('/api/blade_analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rpm: 100000 }),
        });
        const data = await resp.json();
        const c = data.compressor;
        const t = data.turbine;

        // Metrics text
        if (metricsEl) {
            let txt = '── 压气机 ──\n';
            txt += `扩压因子：${c.metrics.diffusion_factor}\n`;
            txt += `De Haller 数：${c.metrics.de_haller_number}\n`;
            txt += `做功系数：${c.metrics.work_coefficient}\n`;
            txt += `流量系数：${c.metrics.flow_coefficient}\n`;
            txt += `反力度：${c.metrics.degree_of_reaction}\n`;
            txt += `效率：${(c.efficiency_estimate * 100).toFixed(1)}%\n`;
            txt += `失速风险：${{low:'低', moderate:'中', high:'高'}[c.stall_risk] || c.stall_risk}\n`;
            txt += `总损失：${c.losses.total}\n`;
            for (const n of c.notes) txt += `⚠ ${zhMessage(n)}\n`;
            txt += '\n── 涡轮 ──\n';
            txt += `Zweifel 系数：${t.metrics.zweifel_coefficient}\n`;
            txt += `做功系数：${t.metrics.work_coefficient}\n`;
            txt += `流量系数：${t.metrics.flow_coefficient}\n`;
            txt += `反力度：${t.metrics.degree_of_reaction}\n`;
            txt += `效率：${(t.efficiency_estimate * 100).toFixed(1)}%\n`;
            txt += `总损失：${t.losses.total}\n`;
            for (const n of t.notes) txt += `⚠ ${zhMessage(n)}\n`;
            metricsEl.textContent = txt;
        }

        // Blade Cp chart
        const canvas = document.getElementById('blade-cp-chart');
        if (canvas) {
            if (bladeCpChart) bladeCpChart.destroy();
            bladeCpChart = new Chart(canvas.getContext('2d'), {
                type: 'line',
                data: {
                    labels: c.pressure_distribution.chord.map(v => v.toFixed(2)),
                    datasets: [
                        {
                            label: '压气机吸力面',
                            data: c.pressure_distribution.Cp_suction,
                            borderColor: C.accent,
                            borderWidth: 1.5,
                            pointRadius: 0,
                            tension: 0.3,
                        },
                        {
                            label: '压气机压力面',
                            data: c.pressure_distribution.Cp_pressure,
                            borderColor: C.danger,
                            borderWidth: 1.5,
                            pointRadius: 0,
                            tension: 0.3,
                        },
                        {
                            label: '涡轮吸力面',
                            data: t.pressure_distribution.Cp_suction,
                            borderColor: C.success,
                            borderWidth: 1.5,
                            pointRadius: 0,
                            borderDash: [4, 3],
                            tension: 0.3,
                        },
                        {
                            label: '涡轮压力面',
                            data: t.pressure_distribution.Cp_pressure,
                            borderColor: C.warning,
                            borderWidth: 1.5,
                            pointRadius: 0,
                            borderDash: [4, 3],
                            tension: 0.3,
                        },
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: '弦长比例', color: C.gridLabel }, ticks: { color: C.gridLabel, maxTicksLimit: 6 }, grid: { color: C.grid } },
                        y: { title: { display: true, text: 'Cp', color: C.gridLabel }, ticks: { color: C.gridLabel }, grid: { color: C.grid }, reverse: true },
                    },
                    plugins: {
                        legend: { labels: { color: C.text, boxWidth: 12 } },
                        title: { display: true, text: '叶片表面压力系数 (Cp)', color: C.text },
                    }
                }
            });
        }
        setDashStatus('✓ 叶片分析完成', 'ok');
    } catch (err) {
        if (metricsEl) metricsEl.textContent = '错误：' + err.message;
        setDashStatus('⚠ ' + err.message, 'warn');
    } finally {
        btn.disabled = false;
        btn.textContent = '叶片型线分析';
    }
}

// ============================================================
// Thermal Barrier Coating (TBC) Analysis
// ============================================================

let tbcChart = null;

async function runTBCAnalysis() {
    const btn = document.getElementById('btn-run-tbc');
    const resultsEl = document.getElementById('tbc-results');
    btn.disabled = true;
    btn.textContent = '分析中…';
    setDashStatus('正在分析热障涂层…', 'ok');

    try {
        const coating = document.getElementById('tbc-coating-select').value;
        const resp = await fetch('/api/tbc_analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'single', coating: coating })
        });
        const data = await resp.json();

        let txt = `🛡️ ${zhName(data.coating_used)}\n`;
        txt += `类别：${data.coating_category === 'bio_inspired' ? '仿生' : '常规'}\n`;
        txt += `导热系数：${data.coating_conductivity_W_mK} W/mK\n`;
        txt += `厚度：${data.coating_thickness_mm} mm\n\n`;

        for (const c of data.components) {
            const arrow = c.substrate_ok ? '✓' : '⚠';
            txt += `${arrow} ${zhComponent(c.component)}\n`;
            txt += `  壁温：${c.wall_no_tbc_K.toFixed(0)} K → ${c.substrate_temp_K.toFixed(0)} K (↓${c.improvement_K.toFixed(0)} K)\n`;
            txt += `  裕度：${c.margin_before_K > 0 ? '+' : ''}${c.margin_before_K.toFixed(0)} K → +${c.margin_after_K.toFixed(0)} K\n`;
            txt += `  质量：+${c.mass_added_g.toFixed(1)} g\n\n`;
        }
        txt += `━━━━━━━━━━━━━━━━━━━━━━━\n`;
        txt += `总增重：+${data.total_mass_added_g} g\n`;
        txt += `平均降温：${data.avg_temp_improvement_K} K\n`;
        txt += `全部安全：${data.all_within_limits ? '✓ 是' : '⚠ 否'}`;
        resultsEl.textContent = txt;

        renderTBCChart(data);
        setDashStatus('✓ 热障涂层分析完成', 'ok');
    } catch (err) {
        resultsEl.textContent = '错误：' + err.message;
        setDashStatus('⚠ ' + err.message, 'warn');
    } finally {
        btn.disabled = false;
        btn.textContent = '分析涂层';
    }
}

async function compareTBCCoatings() {
    const btn = document.getElementById('btn-compare-tbc');
    const resultsEl = document.getElementById('tbc-results');
    btn.disabled = true;
    btn.textContent = '比较中…';
    setDashStatus('正在比较所有热障涂层…', 'ok');

    try {
        const resp = await fetch('/api/tbc_analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'compare' })
        });
        const data = await resp.json();

        let txt = '📊 热障涂层对比\n\n';
        const entries = Object.entries(data.coatings);
        entries.sort((a, b) => b[1].avg_improvement_K - a[1].avg_improvement_K);

        for (const [key, c] of entries) {
            const cat = c.category === 'bio_inspired' ? '🧬' : '⚙️';
            const ok = c.all_ok ? '✓' : '⚠';
            txt += `${ok} ${cat} ${zhName(c.name)}\n`;
            txt += `  k=${c.k_W_mK}W/mK | ΔT=${c.avg_improvement_K}K | +${c.total_mass_g}g\n\n`;
        }
        resultsEl.textContent = txt;

        renderTBCCompareChart(entries);
        setDashStatus('✓ 对比完成', 'ok');
    } catch (err) {
        resultsEl.textContent = '错误：' + err.message;
        setDashStatus('⚠ ' + err.message, 'warn');
    } finally {
        btn.disabled = false;
        btn.textContent = '全部比较';
    }
}

function renderTBCChart(data) {
    const canvas = document.getElementById('tbc-chart');
    if (!canvas) return;
    if (tbcChart) tbcChart.destroy();

    const labels = data.components.map(c => zhComponent(c.component));
    const before = data.components.map(c => c.wall_no_tbc_K);
    const after = data.components.map(c => c.substrate_temp_K);
    const limits = data.components.map(c => {
        const lims = { combustor: 973, turbine: 973, nozzle: 1253 };
        return lims[c.component] || 973;
    });

    tbcChart = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                { label: '无涂层（壁面）', data: before, backgroundColor: 'rgba(255,99,71,0.7)', borderWidth: 0 },
                { label: '有涂层（基体）', data: after, backgroundColor: 'rgba(0,229,255,0.7)', borderWidth: 0 },
                { label: '材料限值', data: limits, type: 'line', borderColor: '#ff6d00', borderWidth: 2, borderDash: [5,5], pointRadius: 4, fill: false },
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#B0BEC5', font: { size: 10 } } },
                       title: { display: true, text: '壁温：使用热障涂层前后', color: '#ECEFF1' } },
            scales: {
                y: { title: { display: true, text: '温度 (K)', color: '#78909C' }, ticks: { color: '#78909C' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { ticks: { color: '#78909C' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}

function renderTBCCompareChart(entries) {
    const canvas = document.getElementById('tbc-chart');
    if (!canvas) return;
    if (tbcChart) tbcChart.destroy();

    const labels = entries.map(([k, c]) => zhName(c.name));
    const improvements = entries.map(([k, c]) => c.avg_improvement_K);
    const masses = entries.map(([k, c]) => c.total_mass_g);
    const colors = entries.map(([k, c]) => c.category === 'bio_inspired' ? 'rgba(0,229,255,0.7)' : 'rgba(255,152,0,0.7)');

    tbcChart = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                { label: '降温幅度 (K)', data: improvements, backgroundColor: colors, borderWidth: 0, yAxisID: 'y' },
                { label: '增加质量 (g)', data: masses, type: 'line', borderColor: '#ff6d00', borderWidth: 2, pointRadius: 4, yAxisID: 'y1', fill: false },
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#B0BEC5', font: { size: 10 } } },
                       title: { display: true, text: '热障涂层对比：降温与增重', color: '#ECEFF1' } },
            scales: {
                y: { position: 'left', title: { display: true, text: 'ΔT (K)', color: '#00E5FF' }, ticks: { color: '#78909C' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y1: { position: 'right', title: { display: true, text: '质量 (g)', color: '#ff6d00' }, ticks: { color: '#78909C' }, grid: { drawOnChartArea: false } },
                x: { ticks: { color: '#78909C', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}
