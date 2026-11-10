/**
 * NovaTurbo 3D Engine Viewer + Simulation Suite
 * Three.js ES Module viewer with thermal heatmap, airflow particles,
 * stress analysis, component controls, section cuts, and explode view.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { STLExporter } from 'three/addons/exporters/STLExporter.js';
import { openDashboard, toggleDashboard } from './dashboard.js';
import { zhComponent, zhMessage } from './zh-cn.js';

// ──── Color Maps ────
function thermalColorMap(t) {
    if (t < 0.25) { const s = t / 0.25; return new THREE.Color(0, s, 1); }
    if (t < 0.5)  { const s = (t - 0.25) / 0.25; return new THREE.Color(0, 1, 1 - s); }
    if (t < 0.75) { const s = (t - 0.5) / 0.25; return new THREE.Color(s, 1, 0); }
    const s = (t - 0.75) / 0.25; return new THREE.Color(1, 1 - s, 0);
}
function stressColorMap(t) {
    if (t < 0.33) { const s = t / 0.33; return new THREE.Color(0.1, 0.8 - s * 0.3, 0.1); }
    if (t < 0.66) { const s = (t - 0.33) / 0.33; return new THREE.Color(0.5 + s * 0.5, 0.5, 0); }
    const s = (t - 0.66) / 0.34; return new THREE.Color(1, 0.3 - s * 0.3, 0);
}
function flowColorMap(t) {
    if (t < 0.5) { const s = t / 0.5; return new THREE.Color(0.05, 0.1 + s * 0.5, 0.4 + s * 0.6); }
    const s = (t - 0.5) / 0.5; return new THREE.Color(s, 0.6 + s * 0.4, 1);
}
function flameColorMap(t) {
    if (t < 0.25) { const s = t / 0.25; return new THREE.Color(0.9 + 0.1 * s, 0.1, 0); }
    if (t < 0.5)  { const s = (t - 0.25) / 0.25; return new THREE.Color(1, 0.2 + 0.5 * s, 0); }
    if (t < 0.75) { const s = (t - 0.5) / 0.25; return new THREE.Color(1, 0.7 + 0.3 * s, 0.1 + 0.3 * s); }
    const s = (t - 0.75) / 0.25; return new THREE.Color(1, 1, 0.4 + 0.6 * s);
}

// State
const state = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    components: {},
    clippingPlane: null,
    wireframe: false,
    sectionActive: false,
    explodeActive: false,
    explodeFactor: 0,
    allVisible: true,
    totalVerts: 0,
    totalFaces: 0,
    engineCenter: new THREE.Vector3(),
    engineLength: 200,
    // Simulation
    simMode: null,
    simData: null,
    geometry19Simulation: null,
    originalMaterials: {},
    flowParticles: null,
    flowArrows: [],
    flameParticles: null,
    flameVolumes: [],
    flameMeta: null,
    clock: new THREE.Clock(),
    // Lattice
    latticeVisible: false,
    latticeMeshes: {},
    currentVariation: 'v1_gyroid_standard',
    exportName: 'engine_geometry',
};

// --- Init ---
function init() {
    setupScene();
    setupLights();
    setupGrid();
    loadComponents();
    loadEngineData();
    bindControls();
    animate();
}

function setupScene() {
    const canvas = document.getElementById('three-canvas');
    state.renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: false,
        preserveDrawingBuffer: true
    });
    state.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    state.renderer.setClearColor(0x0a0e17, 1);
    state.renderer.setSize(canvas.parentElement.clientWidth, canvas.parentElement.clientHeight);
    state.renderer.localClippingEnabled = true;
    state.renderer.shadowMap.enabled = true;
    state.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    state.renderer.toneMappingExposure = 1.2;

    state.scene = new THREE.Scene();

    state.camera = new THREE.PerspectiveCamera(
        45,
        canvas.parentElement.clientWidth / canvas.parentElement.clientHeight,
        0.1,
        10000
    );
    state.camera.position.set(150, 100, 200);

    state.controls = new OrbitControls(state.camera, canvas);
    state.controls.enableDamping = true;
    state.controls.dampingFactor = 0.08;
    state.controls.rotateSpeed = 0.8;
    state.controls.target.set(0, 0, 100);

    state.clippingPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 200);

    window.addEventListener('resize', onResize);
    onResize();
}

function setupLights() {
    state.scene.add(new THREE.AmbientLight(0x404060, 0.6));
    state.scene.add(new THREE.HemisphereLight(0x88aacc, 0x223344, 0.5));

    const key = new THREE.DirectionalLight(0xffffff, 1.0);
    key.position.set(100, 150, 200);
    key.castShadow = true;
    state.scene.add(key);

    const fill = new THREE.DirectionalLight(0x6688aa, 0.4);
    fill.position.set(-100, 50, -100);
    state.scene.add(fill);

    const rim = new THREE.DirectionalLight(0x00E5FF, 0.2);
    rim.position.set(0, -50, -200);
    state.scene.add(rim);
}

function setupGrid() {
    const grid = new THREE.GridHelper(400, 40, 0x1e293b, 0x111827);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 100;
    state.scene.add(grid);
}

// --- Load STL Components ---
let geometry19LoadEpoch = 0;
async function loadComponents(componentList = null, requireComplete = false) {
    const loadEpoch = geometry19LoadEpoch;
    const generatedGeometry19 = componentList !== null;
    let components = componentList;
    if (!components) {
        const resp = await fetch('/api/components');
        components = await resp.json();
        if (loadEpoch !== geometry19LoadEpoch) return false;
    }
    const loader = new STLLoader();
    let loaded = 0;

    const progress = document.getElementById('load-progress');
    progress.textContent = '0 / ' + components.length + ' 个组件';

    for (const comp of components) {
        // Skip lattice meshes from auto-load (loaded on-demand via Lattice button)
        if (!generatedGeometry19 && comp.id.includes('lattice')) {
            loaded++;
            continue;
        }
        try {
            const geometry = await new Promise((resolve, reject) => {
                loader.load(comp.file, resolve, undefined, reject);
            });
            if (loadEpoch !== geometry19LoadEpoch) {
                geometry.dispose();
                return false;
            }

            geometry.computeVertexNormals();

            const material = new THREE.MeshPhysicalMaterial({
                color: new THREE.Color(comp.color),
                metalness: 0.7,
                roughness: 0.3,
                clearcoat: 0.1,
                clearcoatRoughness: 0.4,
                side: THREE.DoubleSide,
                clippingPlanes: [],
            });

            const mesh = new THREE.Mesh(geometry, material);
            mesh.castShadow = true;
            mesh.receiveShadow = true;

            state.scene.add(mesh);
            if (generatedGeometry19 && comp.id.includes('lattice')) {
                mesh.visible = false;
                state.latticeMeshes[comp.id] = mesh;
            } else {
                state.components[comp.id] = {
                    mesh,
                    data: comp,
                    visible: true,
                    originalPosition: mesh.position.clone()
                };
                state.originalMaterials[comp.id] = material.clone();
            }

            state.totalVerts += geometry.attributes.position.count;
            state.totalFaces += geometry.index ? geometry.index.count / 3 : geometry.attributes.position.count / 3;

            loaded++;
            progress.textContent = loaded + ' / ' + components.length + ' 个组件';

        } catch(e) {
            if (loadEpoch !== geometry19LoadEpoch) return false;
            if (requireComplete) {
                disposeCurrentComponents();
                document.getElementById('component-list').innerHTML = '';
                document.getElementById('stat-verts').textContent = '0';
                document.getElementById('stat-faces').textContent = '0';
                document.getElementById('status-left').textContent = '投影预览加载失败';
                window.dispatchEvent(new CustomEvent('geometry19-model-error', {detail: {error: String(e)}}));
                return false;
            }
            console.warn('Failed to load ' + comp.id, e);
            loaded++;
        }
    }

    // Compute center for camera target
    const box = new THREE.Box3();
    Object.values(state.components).forEach(c => box.expandByObject(c.mesh));
    if (!box.isEmpty()) {
        box.getCenter(state.engineCenter);
        state.engineLength = box.max.z - box.min.z;
        if (requireComplete) {
            // Keep the entire projection visible, including large radial extents.
            const size = box.getSize(new THREE.Vector3());
            const fov = THREE.MathUtils.degToRad(state.camera.fov);
            state.engineLength = Math.max(size.x, size.y, size.z) / Math.sin(fov / 2);
        }
    }

    state.controls.target.copy(state.engineCenter);
    state.camera.position.set(
        state.engineCenter.x + state.engineLength * 0.8,
        state.engineCenter.y + state.engineLength * 0.5,
        state.engineCenter.z + state.engineLength * 0.3
    );

    document.getElementById('stat-verts').textContent = formatNum(state.totalVerts);
    document.getElementById('stat-faces').textContent = formatNum(state.totalFaces);
    document.getElementById('status-left').textContent = '已加载 ' + loaded + ' 个组件';

    buildComponentList(components);

    setTimeout(() => {
        document.getElementById('loading-screen').classList.add('hidden');
    }, 300);
    return true;
}

function geometry19ComponentMetadata(componentUrls) {
    const styles = {
        inlet: ['#4FC3F7', 'Air Inlet', 0],
        compressor: ['#81C784', 'Compressor', 1],
        combustor: ['#FF8A65', 'Combustion Chamber', 2],
        turbine: ['#FFD54F', 'Turbine Stage', 3],
        nozzle: ['#E57373', 'Exhaust Nozzle', 4],
        shaft: ['#B0BEC5', 'Main Shaft', 5],
        casing: ['#78909C', 'Outer Casing', 6],
        transitions: ['#9575CD', 'Transitions', 7],
        combustor_lattice: ['#FF6D00', 'Combustor Lattice', 8],
        nozzle_lattice: ['#FF3D00', 'Nozzle Lattice', 9],
    };
    return Object.entries(componentUrls).map(([id, file], index) => {
        const style = styles[id] || ['#00E5FF', id.replaceAll('_', ' '), 50 + index];
        return { id, file, color: style[0], label: zhComponent(id, style[1]), order: style[2], size_kb: 0 };
    }).sort((a, b) => a.order - b.order);
}

function disposeCurrentComponents() {
    if (state.simMode || state.flowParticles || state.flameParticles || state.flowArrows.length) {
        clearSimulation();
    }
    for (const component of Object.values(state.components)) {
        state.scene.remove(component.mesh);
        component.mesh.geometry?.dispose();
        component.mesh.material?.dispose();
    }
    for (const mesh of Object.values(state.latticeMeshes)) {
        state.scene.remove(mesh);
        mesh.geometry?.dispose();
        mesh.material?.dispose();
    }
    for (const material of Object.values(state.originalMaterials)) material.dispose?.();
    state.components = {};
    state.latticeMeshes = {};
    state.originalMaterials = {};
    state.latticeVisible = false;
    document.getElementById('btn-lattice')?.classList.remove('active');
    state.totalVerts = 0;
    state.totalFaces = 0;
    state.exportName = 'engine_geometry';
}

async function showGeometry19Components(event) {
    const urls = event.detail?.component_urls;
    if (!urls || typeof urls !== 'object') return;
    const modelLabel = event.detail.model_label || 'Geometry19';
    document.getElementById('status-left').textContent = '正在加载 ' + modelLabel + ' 设计…';
    geometry19LoadEpoch++;
    disposeCurrentComponents();
    state.geometry19Simulation = event.detail?.simulation || null;
    state.simData = null;
    if (!await loadComponents(geometry19ComponentMetadata(urls), event.detail.require_complete_mesh === true)) return;
    state.exportName = `${modelLabel}_${String(event.detail.design_id || '').slice(0, 12)}`
        .replace(/[^a-zA-Z0-9_-]/g, '_');
    highlightConstraintComponents(event.detail.violated_components || []);
    document.getElementById('status-left').textContent =
        modelLabel + ' ' + String(event.detail.design_id || '').slice(0, 12) + ' 已加载';
    window.dispatchEvent(new CustomEvent('geometry19-model-loaded', {detail: {model_label: modelLabel, design_id: event.detail.design_id}}));
}

function highlightConstraintComponents(components) {
    const ids = new Set(components);
    const mark = mesh => {
        if (!mesh?.material) return;
        mesh.material.color?.lerp(new THREE.Color('#ef4444'), 0.7);
        mesh.material.emissive?.set('#4a0808');
        mesh.material.needsUpdate = true;
    };
    for (const [id, component] of Object.entries(state.components)) {
        if (ids.has(id)) mark(component.mesh);
    }
    if (ids.has('lattice')) {
        Object.values(state.latticeMeshes).forEach(mark);
    }
}

// --- Component List UI ---
function buildComponentList(components) {
    const list = document.getElementById('component-list');
    list.innerHTML = '';

    components.forEach(comp => {
        if (comp.id.includes('lattice')) return; // Lattice shown via button
        const item = document.createElement('div');
        item.className = 'comp-item';
        item.dataset.id = comp.id;
        item.innerHTML =
            '<div class="comp-swatch" style="background:' + comp.color + '"></div>' +
            '<span class="comp-label">' + zhComponent(comp.id, comp.label) + '</span>' +
            '<span class="comp-size">' + comp.size_kb.toFixed(0) + 'KB</span>' +
            '<svg class="comp-eye" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
            '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>' +
            '<circle cx="12" cy="12" r="3"/></svg>';

        item.addEventListener('click', () => toggleComponent(comp.id, item));
        list.appendChild(item);
    });
}

function toggleComponent(id, element) {
    const comp = state.components[id];
    if (!comp) return;
    comp.visible = !comp.visible;
    comp.mesh.visible = comp.visible;
    element.classList.toggle('hidden-comp', !comp.visible);
}

// --- Engine Data Panel ---
async function loadEngineData() {
    try {
        const resp = await fetch('/api/engine');
        const data = await resp.json();
        if (['v3_results', 'v3_new'].includes(document.getElementById('geometry19-model-mode').value)) return;
        renderEngineInfo(data);
    } catch(e) {
        if (['v3_results', 'v3_new'].includes(document.getElementById('geometry19-model-mode').value)) return;
        document.getElementById('engine-info').innerHTML =
            '<div class="loading-placeholder">无法加载发动机数据</div>';
    }
}

function renderEngineInfo(data) {
    const el = document.getElementById('engine-info');
    let html = '';

    if (data.metrics && !data.metrics.error) {
        const m = data.metrics;
        html += '<div class="info-section"><h4>几何</h4>';
        html += infoRow('长度', m.total_length_mm + ' mm');
        html += infoRow('最大直径', m.max_diameter_mm + ' mm');
        html += infoRow('总质量', m.total_mass_g + ' g', 'accent');
        html += '</div>';

        html += '<div class="info-section"><h4>规格</h4>';
        html += infoRow('压比', m.specs.pressure_ratio);
        html += infoRow('转速 (RPM)', formatNum(m.specs.rpm));
        html += infoRow('涡轮进口温度 (TIT)', m.specs.tit_k + ' K (' + m.specs.tit_c + ' \u00B0C)', 'warning');
        html += infoRow('主轴直径', m.specs.shaft_diameter_mm + ' mm');
        html += '</div>';

        html += '<div class="info-section"><h4>质量构成</h4>';
        const total = Object.values(m.mass_breakdown).reduce((a, b) => a + b, 0);
        const colors = ['#4FC3F7','#81C784','#FF8A65','#FFD54F','#E57373','#B0BEC5','#9575CD'];
        let ci = 0;
        html += '<div class="mass-bar">';
        for (const [k, v] of Object.entries(m.mass_breakdown)) {
            const pct = (v / total * 100).toFixed(1);
            html += '<div class="mass-bar-segment" style="width:' + pct + '%;background:' + colors[ci % colors.length] + '" title="' + zhComponent(k, k) + '：' + v + ' g (' + pct + '%)"></div>';
            ci++;
        }
        html += '</div>';
        for (const [k, v] of Object.entries(m.mass_breakdown)) {
            html += infoRow(zhComponent(k, capitalize(k)), v.toFixed(1) + ' g');
        }
        html += '</div>';
    }

    if (data.performance) {
        const p = data.performance;
        html += '<div class="info-section"><h4>AI 数据集</h4>';
        html += infoRow('设计总数', formatNum(p.total_designs));
        html += infoRow('有效', formatNum(p.valid_designs), 'success');
        html += infoRow('推力范围', p.thrust_range[0] + ' \u2013 ' + p.thrust_range[1] + ' N');
        html += infoRow('推重比范围', p.tw_range[0] + ' \u2013 ' + p.tw_range[1]);
        html += infoRow('效率', p.efficiency_range[0] + ' \u2013 ' + p.efficiency_range[1] + '%');
        html += infoRow('平均推力', p.mean_thrust + ' N', 'accent');
        html += infoRow('平均推重比', p.mean_tw, 'accent');
        html += '</div>';
    }

    if (data.metrics && data.metrics.positions) {
        html += '<div class="info-section"><h4>轴向位置</h4>';
        for (const [k, v] of Object.entries(data.metrics.positions)) {
            html += infoRow(zhComponent(k, capitalize(k)), 'z = ' + v + ' mm');
        }
        html += '</div>';
    }

    el.innerHTML = html;
}

function infoRow(label, value, cls) {
    return '<div class="info-row"><span class="info-label">' + label +
           '</span><span class="info-value' + (cls ? ' ' + cls : '') + '">' +
           value + '</span></div>';
}

// --- Toolbar Controls ---
function bindControls() {
window.addEventListener('geometry19-model-ready', showGeometry19Components);
window.addEventListener('geometry19-clear-model', () => {
    geometry19LoadEpoch++;
    disposeCurrentComponents();
    state.geometry19Simulation = null;
    state.simData = null;
    document.getElementById('component-list').innerHTML = '';
    document.getElementById('stat-verts').textContent = '0';
    document.getElementById('stat-faces').textContent = '0';
    document.getElementById('load-progress').textContent = '';
    document.getElementById('status-left').textContent = '等待 Geometry19 结果';
});
    document.getElementById('btn-reset-cam').addEventListener('click', resetCamera);
    document.getElementById('btn-wireframe').addEventListener('click', toggleWireframe);
    document.getElementById('btn-section').addEventListener('click', toggleSection);
    document.getElementById('btn-explode').addEventListener('click', toggleExplode);
    document.getElementById('btn-screenshot').addEventListener('click', takeScreenshot);
    document.getElementById('btn-export-geometry').addEventListener('click', exportGeometry);
    document.getElementById('btn-toggle-all').addEventListener('click', toggleAll);
    const flowOverlay = document.getElementById('inverse-flow-overlay');
    const flowButton = document.getElementById('btn-show-inverse-flow');
    const closeFlow = () => {
        flowOverlay.hidden = true;
        flowButton.focus();
    };
    flowButton.addEventListener('click', () => {
        flowOverlay.hidden = false;
        document.getElementById('btn-close-inverse-flow').focus();
    });
    document.getElementById('btn-close-inverse-flow').addEventListener('click', closeFlow);
    flowOverlay.addEventListener('click', event => {
        if (event.target === flowOverlay) closeFlow();
    });
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && !flowOverlay.hidden) closeFlow();
    });
    document.getElementById('btn-open-inverse-design').addEventListener('click', () => {
        closeFlow();
        openDashboard();
        const selector = document.getElementById('geometry19-model-mode');
        selector.value = 'v3_new';
        selector.dispatchEvent(new Event('change', {bubbles: true}));
    });

    // Simulation buttons
    document.getElementById('btn-sim-thermal').addEventListener('click', function() {
        if (state.simMode === 'thermal') { clearSimulation(); loadEngineData(); }
        else { activateThermalSim(); }
    });
    document.getElementById('btn-sim-flow').addEventListener('click', function() {
        if (state.simMode === 'flow') { clearSimulation(); loadEngineData(); }
        else { activateFlowSim(); }
    });
    document.getElementById('btn-sim-stress').addEventListener('click', function() {
        if (state.simMode === 'stress') { clearSimulation(); loadEngineData(); }
        else { activateStressSim(); }
    });
    document.getElementById('btn-sim-flame').addEventListener('click', function() {
        if (state.simMode === 'flame') { clearSimulation(); loadEngineData(); }
        else { activateFlameSim(); }
    });

    // Lattice toggle
    document.getElementById('btn-lattice').addEventListener('click', toggleLattice);

    // Dashboard toggle
    document.getElementById('btn-dashboard').addEventListener('click', function() {
        toggleDashboard();
    });

    document.getElementById('slider-opacity').addEventListener('input', function() {
        const v = parseFloat(this.value);
        document.getElementById('opacity-val').textContent = Math.round(v * 100) + '%';
        Object.values(state.components).forEach(c => {
            c.mesh.material.opacity = v;
            c.mesh.material.transparent = v < 1;
            c.mesh.material.needsUpdate = true;
        });
    });

    document.getElementById('slider-section').addEventListener('input', function() {
        const v = parseInt(this.value);
        document.getElementById('section-val').textContent = v + '%';
        const range = state.engineLength * 1.2;
        state.clippingPlane.constant = (v / 100) * range - range * 0.1 + state.engineCenter.x;
    });

    document.getElementById('slider-explode').addEventListener('input', function() {
        const v = parseInt(this.value);
        document.getElementById('explode-val').textContent = v + '%';
        state.explodeFactor = v / 100;
        applyExplode();
    });
}

function resetCamera() {
    state.controls.target.copy(state.engineCenter);
    state.camera.position.set(
        state.engineCenter.x + state.engineLength * 0.8,
        state.engineCenter.y + state.engineLength * 0.5,
        state.engineCenter.z + state.engineLength * 0.3
    );
}

function toggleWireframe() {
    state.wireframe = !state.wireframe;
    document.getElementById('btn-wireframe').classList.toggle('active', state.wireframe);
    Object.values(state.components).forEach(c => {
        c.mesh.material.wireframe = state.wireframe;
    });
}

function toggleSection() {
    state.sectionActive = !state.sectionActive;
    document.getElementById('btn-section').classList.toggle('active', state.sectionActive);
    document.getElementById('section-controls').style.display = state.sectionActive ? 'flex' : 'none';

    const planes = state.sectionActive ? [state.clippingPlane] : [];
    Object.values(state.components).forEach(c => {
        c.mesh.material.clippingPlanes = planes;
        c.mesh.material.needsUpdate = true;
    });
    // Apply clipping to lattice meshes too
    Object.values(state.latticeMeshes).forEach(m => {
        m.material.clippingPlanes = planes;
        m.material.needsUpdate = true;
    });
}

function toggleExplode() {
    state.explodeActive = !state.explodeActive;
    document.getElementById('btn-explode').classList.toggle('active', state.explodeActive);
    document.getElementById('explode-controls').style.display = state.explodeActive ? 'flex' : 'none';
    if (!state.explodeActive) {
        state.explodeFactor = 0;
        document.getElementById('slider-explode').value = 0;
        document.getElementById('explode-val').textContent = '0%';
        applyExplode();
    }
}

function applyExplode() {
    const order = ['inlet','compressor','combustor','turbine','nozzle','shaft','casing','transitions'];
    order.forEach((id, i) => {
        const comp = state.components[id];
        if (!comp) return;
        const offset = (i - order.length / 2) * state.explodeFactor * 40;
        comp.mesh.position.z = comp.originalPosition.z + offset;
    });
}

function toggleAll() {
    state.allVisible = !state.allVisible;
    document.getElementById('btn-toggle-all').textContent = state.allVisible ? '全部隐藏' : '全部显示';

    Object.keys(state.components).forEach(id => {
        state.components[id].visible = state.allVisible;
        state.components[id].mesh.visible = state.allVisible;
    });

    document.querySelectorAll('.comp-item').forEach(el => {
        el.classList.toggle('hidden-comp', !state.allVisible);
    });
}

function takeScreenshot() {
    state.renderer.render(state.scene, state.camera);
    const link = document.createElement('a');
    link.download = 'engine_view.png';
    link.href = state.renderer.domElement.toDataURL('image/png');
    link.click();
    document.getElementById('status-left').textContent = '截图已保存';
}

function exportGeometry() {
    const status = document.getElementById('status-left');
    const exportScene = new THREE.Group();
    for (const component of Object.values(state.components)) {
        if (!component.mesh.visible) continue;
        const mesh = component.mesh.clone(false);
        // The explode control moves meshes for display only; export the assembled design.
        mesh.position.copy(component.originalPosition);
        exportScene.add(mesh);
    }
    for (const mesh of Object.values(state.latticeMeshes)) {
        if (mesh.visible) exportScene.add(mesh.clone(false));
    }
    if (exportScene.children.length === 0) {
        status.textContent = '没有可导出的几何；请先显示组件';
        return;
    }
    try {
        exportScene.updateMatrixWorld(true);
        const binary = new STLExporter().parse(exportScene, {binary: true});
        const url = URL.createObjectURL(new Blob([binary.buffer], {type: 'model/stl'}));
        const link = document.createElement('a');
        link.href = url;
        link.download = `${state.exportName}.stl`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 60000);
        status.textContent = `已导出 ${exportScene.children.length} 个可见组件的 STL 几何（毫米）`;
    } catch (error) {
        console.error('STL export failed', error);
        status.textContent = '导出 STL 失败，请检查浏览器控制台';
    }
}

// --- Animation Loop ---
let frameCount = 0;
let lastTime = performance.now();

function animate() {
    requestAnimationFrame(animate);
    state.controls.update();

    // Animate flow particles
    if (state.simMode === 'flow' && state.flowParticles) {
        var positions = state.flowParticles.geometry.attributes.position;
        var velocities = state.flowParticles.geometry.attributes.velocity;
        var dt = state.clock.getDelta() * 15;
        for (var i = 0; i < positions.count; i++) {
            var z = positions.getZ(i);
            z += velocities.getZ(i) * dt * 0.05;
            if (z > state.engineLength + 30) z = -10 + Math.random() * 10;
            if (z < -20) z = state.engineLength + Math.random() * 10;
            positions.setZ(i, z);
        }
        positions.needsUpdate = true;
    }

    // Animate flame particles
    if (state.simMode === 'flame' && state.flameParticles && state.flameMeta) {
        var flamePos = state.flameParticles.geometry.attributes.position;
        var flameVel = state.flameParticles.geometry.attributes.velocity;
        var flameRegion = state.flameParticles.geometry.attributes.region;
        var dtf = state.clock.getDelta() * 25;
        var cb = state.flameMeta.combustor;
        var nb = state.flameMeta.nozzle;
        var plumeEnd = nb.zMax + state.flameMeta.plumeLength;

        for (var fi = 0; fi < flamePos.count; fi++) {
            var x = flamePos.getX(fi);
            var y = flamePos.getY(fi);
            var z = flamePos.getZ(fi);
            var vx = flameVel.getX(fi);
            var vy = flameVel.getY(fi);
            var vz = flameVel.getZ(fi);
            var reg = flameRegion.getX(fi);

            // Spiral turbulence for a fume-like motion
            var swirl = reg < 0.5 ? 0.0016 : 0.0009;
            vx += -y * swirl * dtf + (Math.random() - 0.5) * 0.02;
            vy +=  x * swirl * dtf + (Math.random() - 0.5) * 0.02;

            x += vx * dtf * 0.06;
            y += vy * dtf * 0.06;
            z += vz * dtf * 0.07;

            var r = Math.sqrt(x * x + y * y);
            var maxR = reg < 0.5 ? cb.radius : nb.radius + 6.0;
            var zStart = reg < 0.5 ? cb.zMin : nb.zMin;
            var zEnd = reg < 0.5 ? cb.zMax : plumeEnd;

            // Respawn when particle exits the flame envelope
            if (z > zEnd || z < zStart || r > maxR) {
                var theta = Math.random() * Math.PI * 2;
                var rr = Math.sqrt(Math.random()) * (reg < 0.5 ? cb.radius * 0.8 : nb.radius * 0.7);
                x = Math.cos(theta) * rr;
                y = Math.sin(theta) * rr;
                z = reg < 0.5
                    ? cb.zMin + Math.random() * (cb.zMax - cb.zMin) * 0.65
                    : nb.zMin + Math.random() * Math.max((nb.zMax - nb.zMin) * 0.25, 3.0);
                vx = (Math.random() - 0.5) * 0.9;
                vy = (Math.random() - 0.5) * 0.9;
                vz = reg < 0.5 ? 3.0 + Math.random() * 10.0 : 8.0 + Math.random() * 16.0;
            }

            flamePos.setXYZ(fi, x, y, z);
            flameVel.setXYZ(fi, vx, vy, vz);
        }
        flamePos.needsUpdate = true;
        flameVel.needsUpdate = true;

        // Subtle breathing of volumetric shells
        for (var fv = 0; fv < state.flameVolumes.length; fv++) {
            var shell = state.flameVolumes[fv];
            var pulse = 1.0 + 0.02 * Math.sin(performance.now() * 0.006 + fv * 1.2);
            shell.scale.set(pulse, pulse, 1.0);
            shell.material.opacity = 0.12 + 0.03 * Math.sin(performance.now() * 0.01 + fv);
        }
    }

    state.renderer.render(state.scene, state.camera);

    frameCount++;
    const now = performance.now();
    if (now - lastTime >= 1000) {
        document.getElementById('stat-fps').textContent = frameCount;
        frameCount = 0;
        lastTime = now;
    }
}

function onResize() {
    const vp = document.getElementById('viewport');
    const w = vp.clientWidth;
    const h = vp.clientHeight;
    state.camera.aspect = w / h;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(w, h);
}

// --- Helpers ---
function formatNum(n) { return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

function interpolateArray(xArr, yArr, x) {
    if (x <= xArr[0]) return yArr[0];
    if (x >= xArr[xArr.length - 1]) return yArr[yArr.length - 1];
    for (var i = 0; i < xArr.length - 1; i++) {
        if (x >= xArr[i] && x <= xArr[i + 1]) {
            var t = (x - xArr[i]) / (xArr[i + 1] - xArr[i]);
            return yArr[i] + t * (yArr[i + 1] - yArr[i]);
        }
    }
    return yArr[yArr.length - 1];
}

// ──── Simulation Engine ────
async function loadSimulation(mode) {
    if ((mode === 'thermal' || mode === 'stress') && state.geometry19Simulation) {
        return state.geometry19Simulation;
    }
    if (state.simData) return state.simData;
    document.getElementById('status-left').textContent = '正在运行模拟…';
    try {
        var resp = await fetch('/api/simulation');
        state.simData = await resp.json();
        document.getElementById('status-left').textContent = '模拟完成';
        return state.simData;
    } catch(e) {
        document.getElementById('status-left').textContent = '模拟失败';
        return null;
    }
}

function clearSimulation() {
    for (var id in state.components) {
        var comp = state.components[id];
        if (state.originalMaterials[id]) {
            comp.mesh.material = state.originalMaterials[id].clone();
            if (state.sectionActive) comp.mesh.material.clippingPlanes = [state.clippingPlane];
        }
        if (comp.mesh.geometry.attributes.color) {
            comp.mesh.geometry.deleteAttribute('color');
        }
    }
    if (state.flowParticles) {
        state.scene.remove(state.flowParticles);
        state.flowParticles = null;
    }
    if (state.flameParticles) {
        state.scene.remove(state.flameParticles);
        if (state.flameParticles.geometry) state.flameParticles.geometry.dispose();
        if (state.flameParticles.material) state.flameParticles.material.dispose();
        state.flameParticles = null;
    }
    state.flameVolumes.forEach(function(m) {
        state.scene.remove(m);
        if (m.geometry) m.geometry.dispose();
        if (m.material) m.material.dispose();
    });
    state.flameVolumes = [];
    state.flameMeta = null;
    state.flowArrows.forEach(function(a) { state.scene.remove(a); });
    state.flowArrows = [];
    document.getElementById('sim-legend').style.display = 'none';
    document.getElementById('sim-badge').style.display = 'none';
    ['btn-sim-thermal', 'btn-sim-flow', 'btn-sim-stress', 'btn-sim-flame'].forEach(function(btnId) {
        document.getElementById(btnId).classList.remove('active', 'active-thermal', 'active-flow', 'active-stress', 'active-flame');
    });
    state.simMode = null;
}

async function activateThermalSim() {
    var data = await loadSimulation('thermal');
    if (!data) return;
    clearSimulation();
    state.simMode = 'thermal';
    document.getElementById('btn-sim-thermal').classList.add('active-thermal');

    var thermal = data.thermal;
    var tMin = thermal.t_min;
    var tMax = thermal.t_max;

    for (var compId in state.components) {
        var comp = state.components[compId];
        var tempData = thermal.component_temps[compId];
        if (!tempData) continue;

        var geo = comp.mesh.geometry;
        var pos = geo.attributes.position;
        var colors = new Float32Array(pos.count * 3);
        var zOffset = tempData.z_offset || 0;

        for (var i = 0; i < pos.count; i++) {
            var z = pos.getZ(i);
            var localZ = z - zOffset;
            var temp = interpolateArray(tempData.z, tempData.wall_temp, localZ);
            var t = Math.max(0, Math.min(1, (temp - tMin) / (tMax - tMin)));
            var color = thermalColorMap(t);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }

        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        comp.mesh.material = new THREE.MeshPhysicalMaterial({
            vertexColors: true, metalness: 0.3, roughness: 0.5,
            side: THREE.DoubleSide,
            clippingPlanes: state.sectionActive ? [state.clippingPlane] : [],
        });
    }

    showLegend('温度 (K)', tMin, tMax, thermalColorMap);
    showBadge('热分析', '#ef4444');
    renderSimInfo(data, 'thermal');
}

async function activateFlowSim() {
    var data = await loadSimulation();
    if (!data) return;
    clearSimulation();
    state.simMode = 'flow';
    document.getElementById('btn-sim-flow').classList.add('active-flow');

    var flow = data.flow;

    // Make engine semi-transparent
    for (var id in state.components) {
        var comp = state.components[id];
        comp.mesh.material = new THREE.MeshPhysicalMaterial({
            color: comp.data.color, metalness: 0.5, roughness: 0.4,
            transparent: true, opacity: 0.18, side: THREE.DoubleSide,
            clippingPlanes: state.sectionActive ? [state.clippingPlane] : [],
        });
    }

    // Create streamline tubes (smooth curves through flow points)
    for (var s = 0; s < flow.streamlines.length; s++) {
        var pts = flow.streamlines[s].points;
        if (pts.length < 3) continue;

        // Build curve points
        var curvePoints = [];
        for (var p = 0; p < pts.length; p++) {
            curvePoints.push(new THREE.Vector3(pts[p].x, pts[p].y, pts[p].z));
        }
        var curve = new THREE.CatmullRomCurve3(curvePoints);

        // Create tube with velocity-based color
        var tubePts = 80;
        var tubeGeo = new THREE.TubeGeometry(curve, tubePts, 1.2, 6, false);
        var colors = new Float32Array(tubeGeo.attributes.position.count * 3);
        for (var vi = 0; vi < tubeGeo.attributes.position.count; vi++) {
            var z = tubeGeo.attributes.position.getZ(vi);
            // Map z to speed
            var zNorm = (z - (pts[0].z || 0)) / (state.engineLength + 1);
            zNorm = Math.max(0, Math.min(1, zNorm));
            // Find closest point for speed
            var idx = Math.min(Math.floor(zNorm * (pts.length - 1)), pts.length - 1);
            var speed = pts[idx].speed || 0;
            var t = Math.max(0, Math.min(1, (speed - flow.v_min) / (flow.v_max - flow.v_min + 0.1)));
            var c = flowColorMap(t);
            colors[vi * 3] = c.r;
            colors[vi * 3 + 1] = c.g;
            colors[vi * 3 + 2] = c.b;
        }
        tubeGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        var tubeMat = new THREE.MeshPhysicalMaterial({
            vertexColors: true, metalness: 0.6, roughness: 0.3,
            transparent: true, opacity: 0.85, side: THREE.DoubleSide,
        });
        var tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
        state.scene.add(tubeMesh);
        state.flowArrows.push(tubeMesh);
    }

    // Animated particles along streamlines
    var allPoints = [];
    var allColors = [];
    var allVelocities = [];

    for (var s = 0; s < flow.streamlines.length; s++) {
        var pts = flow.streamlines[s].points;
        for (var p = 0; p < pts.length; p++) {
            var pt = pts[p];
            // Multiple particles per point, spread radially
            for (var j = 0; j < 5; j++) {
                var angle = Math.random() * Math.PI * 2;
                var spread = 2 + Math.random() * 5;
                allPoints.push(
                    pt.x + Math.cos(angle) * spread,
                    pt.y + Math.sin(angle) * spread,
                    pt.z + (Math.random() - 0.5) * 3
                );
                var t = Math.max(0, Math.min(1, (pt.speed - flow.v_min) / (flow.v_max - flow.v_min + 0.1)));
                var c = flowColorMap(t);
                allColors.push(c.r, c.g, c.b);
                allVelocities.push(pt.vx * 0.3, pt.vy * 0.3, pt.vz);
            }
        }
    }

    var particleGeo = new THREE.BufferGeometry();
    particleGeo.setAttribute('position', new THREE.Float32BufferAttribute(allPoints, 3));
    particleGeo.setAttribute('color', new THREE.Float32BufferAttribute(allColors, 3));
    particleGeo.setAttribute('velocity', new THREE.Float32BufferAttribute(allVelocities, 3));

    var particleMat = new THREE.PointsMaterial({
        size: 1.8, vertexColors: true, transparent: true, opacity: 0.7,
        sizeAttenuation: true, blending: THREE.AdditiveBlending, depthWrite: false,
    });

    state.flowParticles = new THREE.Points(particleGeo, particleMat);
    state.scene.add(state.flowParticles);

    // Station labels (text sprites at component boundaries)
    var stations = data.cycle.stations;
    var stationColors = { 'S0': '#94a3b8', 'S1': '#4FC3F7', 'S2': '#81C784', 'S3': '#FF8A65', 'S4': '#FFD54F', 'S5': '#E57373' };
    for (var key in stations) {
        var st = stations[key];
        if (st.z_mm !== undefined) {
            var label = createTextSprite(
                st.name + '\n' + st.T_total_K + 'K | ' + st.P_total_kPa + 'kPa',
                stationColors[key] || '#ffffff'
            );
            label.position.set(0, 70, st.z_mm || 0);
            state.scene.add(label);
            state.flowArrows.push(label);
        }
    }

    // Flow direction arrows (larger, at key stations)
    var arrowPositions = [
        { z: 15, r: 0, label: '进气' },
        { z: 70, r: 0, label: '压缩' },
        { z: 130, r: 0, label: '燃烧' },
        { z: 180, r: 0, label: '膨胀' },
        { z: 215, r: 0, label: '排气' }
    ];
    for (var a = 0; a < arrowPositions.length; a++) {
        var ap = arrowPositions[a];
        var arrow = new THREE.ArrowHelper(
            new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, -55, ap.z),
            18, 0x3b82f6, 6, 4
        );
        state.scene.add(arrow);
        state.flowArrows.push(arrow);
    }

    showLegend('速度 (m/s)', flow.v_min, flow.v_max, flowColorMap);
    showBadge('气流分析', '#3b82f6');
    renderSimInfo(data, 'flow');
}

async function activateStressSim() {
    var data = await loadSimulation('stress');
    if (!data) return;
    clearSimulation();
    state.simMode = 'stress';
    document.getElementById('btn-sim-stress').classList.add('active-stress');

    var stress = data.stress;
    var sMin = stress.s_min;
    var sMax = stress.s_max;

    for (var compId in state.components) {
        var comp = state.components[compId];
        var stressData = stress.component_stress[compId];
        if (!stressData) continue;

        var geo = comp.mesh.geometry;
        var pos = geo.attributes.position;
        var colors = new Float32Array(pos.count * 3);
        var zOffset = stressData.z_offset || 0;

        for (var i = 0; i < pos.count; i++) {
            var z = pos.getZ(i);
            var localZ = z - zOffset;
            var sv = interpolateArray(stressData.z, stressData.total_MPa, localZ);
            var t = Math.max(0, Math.min(1, (sv - sMin) / (sMax - sMin + 0.1)));
            var color = stressColorMap(t);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }

        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        comp.mesh.material = new THREE.MeshPhysicalMaterial({
            vertexColors: true, metalness: 0.3, roughness: 0.5,
            side: THREE.DoubleSide,
            clippingPlanes: state.sectionActive ? [state.clippingPlane] : [],
        });
    }

    showLegend('应力 (MPa)', sMin, sMax, stressColorMap);
    showBadge('应力分析', '#f59e0b');
    renderSimInfo(data, 'stress');
}

function getComponentBounds(componentId) {
    var comp = state.components[componentId];
    if (!comp || !comp.mesh) return null;
    var box = new THREE.Box3().setFromObject(comp.mesh);
    return {
        zMin: box.min.z,
        zMax: box.max.z,
        radius: Math.max(box.max.x - box.min.x, box.max.y - box.min.y) * 0.5
    };
}

async function activateFlameSim() {
    var data = await loadSimulation();
    if (!data) return;
    clearSimulation();
    state.simMode = 'flame';
    document.getElementById('btn-sim-flame').classList.add('active-flame');

    // Fade solids to reveal volumetric flame core
    for (var id in state.components) {
        var comp = state.components[id];
        var op = (id === 'combustor' || id === 'nozzle') ? 0.12 : 0.22;
        comp.mesh.material = new THREE.MeshPhysicalMaterial({
            color: comp.data.color, metalness: 0.4, roughness: 0.45,
            transparent: true, opacity: op, side: THREE.DoubleSide,
            clippingPlanes: state.sectionActive ? [state.clippingPlane] : [],
        });
    }

    var cb = getComponentBounds('combustor');
    var nb = getComponentBounds('nozzle');
    if (!cb || !nb) {
        document.getElementById('status-left').textContent = '火焰模拟不可用：缺少燃烧室或喷管网格';
        return;
    }

    state.flameMeta = {
        combustor: { zMin: cb.zMin, zMax: cb.zMax, radius: Math.max(8, cb.radius * 0.78) },
        nozzle: { zMin: nb.zMin, zMax: nb.zMax, radius: Math.max(4, nb.radius * 0.68) },
        plumeLength: 48.0
    };

    var allPoints = [];
    var allColors = [];
    var allVelocities = [];
    var allRegions = [];

    function pushParticle(region) {
        var meta = region === 0 ? state.flameMeta.combustor : state.flameMeta.nozzle;
        var theta = Math.random() * Math.PI * 2;
        var r = Math.sqrt(Math.random()) * meta.radius;
        var x = Math.cos(theta) * r;
        var y = Math.sin(theta) * r;
        var z = region === 0
            ? meta.zMin + Math.random() * (meta.zMax - meta.zMin) * 0.75
            : meta.zMin + Math.random() * Math.max((meta.zMax - meta.zMin) * 0.35, 4.0);
        var t = region === 0 ? Math.random() * 0.75 + 0.2 : Math.random() * 0.55 + 0.15;
        var c = flameColorMap(Math.min(1, t));

        allPoints.push(x, y, z);
        allColors.push(c.r, c.g, c.b);
        allVelocities.push(
            (Math.random() - 0.5) * (region === 0 ? 0.8 : 1.1),
            (Math.random() - 0.5) * (region === 0 ? 0.8 : 1.1),
            region === 0 ? (3.0 + Math.random() * 10.0) : (8.0 + Math.random() * 16.0)
        );
        allRegions.push(region);
    }

    for (var i = 0; i < 2200; i++) pushParticle(0); // combustor flame body
    for (var j = 0; j < 1300; j++) pushParticle(1); // nozzle plume

    var flameGeo = new THREE.BufferGeometry();
    flameGeo.setAttribute('position', new THREE.Float32BufferAttribute(allPoints, 3));
    flameGeo.setAttribute('color', new THREE.Float32BufferAttribute(allColors, 3));
    flameGeo.setAttribute('velocity', new THREE.Float32BufferAttribute(allVelocities, 3));
    flameGeo.setAttribute('region', new THREE.Float32BufferAttribute(allRegions, 1));

    var flameMat = new THREE.PointsMaterial({
        size: 2.8, vertexColors: true, transparent: true, opacity: 0.72,
        sizeAttenuation: true, blending: THREE.AdditiveBlending, depthWrite: false,
    });

    state.flameParticles = new THREE.Points(flameGeo, flameMat);
    state.scene.add(state.flameParticles);

    // Volumetric shells to emulate fume-like combustion volume
    var coreShellGeo = new THREE.CylinderGeometry(
        state.flameMeta.combustor.radius * 0.95,
        state.flameMeta.combustor.radius * 0.62,
        state.flameMeta.combustor.zMax - state.flameMeta.combustor.zMin,
        24, 1, true
    );
    coreShellGeo.rotateX(Math.PI / 2);
    coreShellGeo.translate(0, 0, (state.flameMeta.combustor.zMin + state.flameMeta.combustor.zMax) * 0.5);
    var coreShell = new THREE.Mesh(coreShellGeo, new THREE.MeshBasicMaterial({
        color: 0xff5a00, transparent: true, opacity: 0.12, side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending, depthWrite: false
    }));
    state.scene.add(coreShell);
    state.flameVolumes.push(coreShell);

    var plumeGeo = new THREE.ConeGeometry(state.flameMeta.nozzle.radius * 0.72, state.flameMeta.plumeLength, 20, 1, true);
    plumeGeo.rotateX(Math.PI / 2);
    plumeGeo.translate(0, 0, state.flameMeta.nozzle.zMax + state.flameMeta.plumeLength * 0.5);
    var plume = new THREE.Mesh(plumeGeo, new THREE.MeshBasicMaterial({
        color: 0xff8a33, transparent: true, opacity: 0.1, side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending, depthWrite: false
    }));
    state.scene.add(plume);
    state.flameVolumes.push(plume);

    showLegend('火焰温度 (K)', 900, 2200, flameColorMap);
    showBadge('燃烧火焰', '#ff6a00');
    renderSimInfo(data, 'flame');
}

function createTextSprite(text, color) {
    var canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 128;
    var ctx = canvas.getContext('2d');
    ctx.font = 'bold 20px Inter, sans-serif';
    ctx.fillStyle = color || '#ffffff';
    ctx.textAlign = 'center';
    var lines = text.split('\n');
    for (var i = 0; i < lines.length; i++) {
        ctx.fillText(lines[i], 128, 40 + i * 28);
    }
    var texture = new THREE.CanvasTexture(canvas);
    var mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    var sprite = new THREE.Sprite(mat);
    sprite.scale.set(40, 20, 1);
    return sprite;
}

function showLegend(title, min, max, colorMapFn) {
    document.getElementById('sim-legend').style.display = 'block';
    document.getElementById('legend-title').textContent = title;
    document.getElementById('legend-max').textContent = Math.round(max);
    document.getElementById('legend-mid').textContent = Math.round((min + max) / 2);
    document.getElementById('legend-min').textContent = Math.round(min);
    var canvas = document.getElementById('legend-canvas');
    var ctx = canvas.getContext('2d');
    for (var y = 0; y < canvas.height; y++) {
        var t = 1 - y / canvas.height;
        var color = colorMapFn(t);
        ctx.fillStyle = 'rgb(' + Math.round(color.r * 255) + ',' + Math.round(color.g * 255) + ',' + Math.round(color.b * 255) + ')';
        ctx.fillRect(0, y, canvas.width, 1);
    }
}

function showBadge(text, color) {
    var badge = document.getElementById('sim-badge');
    badge.style.display = 'flex';
    document.getElementById('sim-badge-text').textContent = text;
    document.getElementById('sim-badge-icon').style.background = color;
    badge.style.color = color;
    badge.style.borderColor = color;
}

function renderSimInfo(data, mode) {
    var el = document.getElementById('engine-info');
    var html = '';
    var c = data.cycle;

    // Status header with overall health
    if (mode === 'thermal') {
        var allSafe = true;
        for (var comp in data.thermal.material_limits) {
            var temps = data.thermal.component_temps[comp];
            if (!temps) continue;
            var maxWall = Math.max.apply(null, temps.wall_temp);
            if (maxWall >= data.thermal.material_limits[comp]) allSafe = false;
        }
        html += '<div class="info-section" style="text-align:center;padding:12px;">';
        html += '<div style="font-size:28px;margin-bottom:4px;">' + (allSafe ? '✅' : '⚠️') + '</div>';
        html += '<div style="font-size:13px;font-weight:600;color:' + (allSafe ? 'var(--success)' : 'var(--warning)') + ';">';
        html += allSafe ? '所有温度均在限值内' : '检测到热风险';
        html += '</div></div>';
    }

    if (mode === 'stress') {
        var allSafe = true;
        for (var sc in data.stress.yield_margins) {
            if (data.stress.yield_margins[sc] <= 0) allSafe = false;
        }
        html += '<div class="info-section" style="text-align:center;padding:12px;">';
        html += '<div style="font-size:28px;margin-bottom:4px;">' + (allSafe ? '✅' : '🔴') + '</div>';
        html += '<div style="font-size:13px;font-weight:600;color:' + (allSafe ? 'var(--success)' : 'var(--danger)') + ';">';
        html += allSafe ? '所有应力均低于屈服强度' : '存在结构失效风险';
        html += '</div></div>';
    }

    if (mode === 'flow') {
        html += '<div class="info-section" style="text-align:center;padding:12px;">';
        html += '<div style="font-size:28px;margin-bottom:4px;">🌊</div>';
        html += '<div style="font-size:13px;font-weight:600;color:var(--accent);">气流分析</div>';
        html += '<div style="font-size:11px;color:var(--text-secondary);margin-top:4px;">流线以颜色表示发动机内部流速</div>';
        html += '</div>';
    }

    if (mode === 'flame') {
        html += '<div class="info-section" style="text-align:center;padding:12px;">';
        html += '<div style="font-size:28px;margin-bottom:4px;">🔥</div>';
        html += '<div style="font-size:13px;font-weight:600;color:#ff6a00;">燃烧火焰可视化</div>';
        html += '<div style="font-size:11px;color:var(--text-secondary);margin-top:4px;">粒子烟羽显示高温核心与喷管尾焰动态</div>';
        html += '</div>';
    }

    html += '<div class="info-section"><h4>循环性能</h4>';
    html += infoRow('推力', c.thrust_N + ' N (' + c.thrust_kgf + ' kgf)', 'accent');
    html += infoRow('排气速度', c.exhaust_velocity + ' m/s');
    html += infoRow('排气温度', Math.round(c.exhaust_temp_K) + ' K', 'warning');
    html += infoRow('燃油流量', c.fuel_flow_g_hr + ' g/hr');
    html += infoRow('单位推力油耗', c.tsfc_g_kNs + ' g/kN\u00B7s');
    html += infoRow('热效率', c.thermal_efficiency_pct + '%', 'success');
    html += infoRow('压气机功率', c.compressor_power_kW + ' kW');
    html += infoRow('涡轮功率', c.turbine_power_kW + ' kW');
    html += '</div>';

    html += '<div class="info-section"><h4>截面分析</h4>';
    var stationLabels = {
        'S0': '🌬️ 环境', 'S1': '📥 进气道出口',
        'S2': '🔄 压气机出口', 'S3': '🔥 燃烧室出口',
        'S4': '⚙️ 涡轮出口', 'S5': '💨 喷管出口'
    };
    for (var key in c.stations) {
        var st = c.stations[key];
        var label = stationLabels[key] || st.name;
        html += infoRow(label, st.T_total_K + ' K / ' + st.P_total_kPa + ' kPa');
    }
    html += '</div>';

    if (mode === 'thermal') {
        html += '<div class="info-section"><h4>🌡️ 热限值</h4>';
        html += '<div style="font-size:10px;color:var(--text-muted);margin-bottom:8px;">最高壁温与材料限值对比。绿色为安全，黄色为警示。</div>';
        for (var comp in data.thermal.material_limits) {
            var temps = data.thermal.component_temps[comp];
            if (!temps) continue;
            var maxWall = Math.round(Math.max.apply(null, temps.wall_temp));
            var limit = data.thermal.material_limits[comp];
            var margin = limit - maxWall;
            var pct = Math.round(maxWall / limit * 100);
            var status = maxWall < limit * 0.85 ? 'success' : (maxWall < limit ? 'warning' : 'danger');
            html += '<div class="info-row">';
            html += '<span class="info-label">' + zhComponent(comp, capitalize(comp)) + '</span>';
            html += '<span class="info-value ' + status + '">' + maxWall + ' / ' + limit + ' K</span>';
            html += '</div>';
            // Progress bar showing how close to limit
            html += '<div style="height:4px;background:var(--bg-hover);border-radius:2px;margin:-4px 0 6px 0;">';
            html += '<div style="height:100%;width:' + Math.min(pct, 100) + '%;border-radius:2px;background:';
            html += status === 'success' ? 'var(--success)' : (status === 'warning' ? 'var(--warning)' : 'var(--danger)');
            html += ';"></div></div>';
        }
        html += '</div>';
    }

    if (mode === 'flow') {
        html += '<div class="info-section"><h4>🌊 流动阶段</h4>';
        html += '<div style="font-size:10px;color:var(--text-muted);margin-bottom:8px;">各组件压比与温度变化</div>';
        for (var fc in data.flow.component_flow) {
            var fd = data.flow.component_flow[fc];
            html += '<div class="info-row">';
            html += '<span class="info-label">' + zhComponent(fc, capitalize(fc)) + '</span>';
            html += '<span class="info-value">压比=' + fd.pressure_ratio + '</span>';
            html += '</div>';
            html += '<div style="font-size:10px;color:var(--text-secondary);margin:-2px 0 6px 12px;">';
            html += fd.inlet_temp_K + 'K → ' + fd.outlet_temp_K + 'K (Δ' + (fd.outlet_temp_K - fd.inlet_temp_K > 0 ? '+' : '') + (fd.outlet_temp_K - fd.inlet_temp_K) + 'K)';
            html += '</div>';
        }
        html += '</div>';

        html += '<div class="info-section"><h4>📊 流动图例</h4>';
        html += '<div style="font-size:10px;color:var(--text-muted);margin-bottom:6px;">';
        html += '• <span style="color:#0a1a4a;">深蓝</span> = 低速流动（进气）<br>';
        html += '• <span style="color:#1a9fff;">青色</span> = 中速流动<br>';
        html += '• <span style="color:#ffffff;">白色</span> = 高速流动（排气）<br>';
        html += '• 管线表示流线路径<br>';
        html += '• 粒子显示流动方向';
        html += '</div></div>';
    }

    if (mode === 'flame') {
        var s3 = null, s5 = null;
        if (c.stations) {
            s3 = c.stations['4_combustor_exit'] || c.stations['S3'] || null;
            s5 = c.stations['6_nozzle_exit'] || c.stations['S5'] || null;
        }
        html += '<div class="info-section"><h4>🔥 火焰核心</h4>';
        html += infoRow('燃烧室核心', s3 ? s3.T_total_K + ' K' : '无数据', 'warning');
        html += infoRow('喷管出口', s5 ? s5.T_total_K + ' K' : '无数据');
        html += infoRow('燃油流量', c.fuel_flow_g_hr + ' g/hr');
        html += infoRow('显示模式', '烟羽粒子效果', 'accent');
        html += '</div>';

        html += '<div class="info-section"><h4>📊 火焰图例</h4>';
        html += '<div style="font-size:10px;color:var(--text-muted);">';
        html += '• <span style="color:#ff3b00;">红色/橙色</span> = 高温反应区<br>';
        html += '• <span style="color:#ffd166;">黄色/白色</span> = 燃烧峰值<br>';
        html += '• 核心壳层 = 燃烧室体积<br>';
        html += '• 锥形羽流 = 排气尾焰';
        html += '</div></div>';
    }

    if (mode === 'stress') {
        html += '<div class="info-section"><h4>🛡️ 屈服裕度</h4>';
        html += '<div style="font-size:10px;color:var(--text-muted);margin-bottom:8px;">峰值应力与材料屈服强度对比；裕度越高越安全。</div>';
        for (var sc in data.stress.yield_margins) {
            var sd = data.stress.component_stress[sc];
            var maxS = Math.round(Math.max.apply(null, sd.total_MPa));
            var margin = Math.round(data.stress.yield_margins[sc]);
            var yieldMPa = sd.yield_MPa;
            var status = margin > 30 ? 'success' : (margin > 10 ? 'warning' : 'danger');
            html += '<div class="info-row">';
            html += '<span class="info-label">' + zhComponent(sc, capitalize(sc)) + '</span>';
            html += '<span class="info-value ' + status + '">' + maxS + ' / ' + yieldMPa + ' MPa</span>';
            html += '</div>';
            html += '<div style="display:flex;align-items:center;margin:-2px 0 6px 0;gap:6px;">';
            html += '<div style="flex:1;height:4px;background:var(--bg-hover);border-radius:2px;">';
            var usePct = Math.min(Math.round(maxS / yieldMPa * 100), 100);
            html += '<div style="height:100%;width:' + usePct + '%;border-radius:2px;background:';
            html += status === 'success' ? 'var(--success)' : (status === 'warning' ? 'var(--warning)' : 'var(--danger)');
            html += ';"></div></div>';
            html += '<span style="font-size:9px;color:var(--text-muted);">' + margin + '% 裕度</span>';
            html += '</div>';
            html += '<div style="font-size:9px;color:var(--text-muted);margin:-2px 0 6px 12px;">材料：' + sd.material + '</div>';
        }
        html += '</div>';

        html += '<div class="info-section"><h4>📊 应力图例</h4>';
        html += '<div style="font-size:10px;color:var(--text-muted);">';
        html += '• <span style="color:#1acc1a;">绿色</span> = 低应力（安全）<br>';
        html += '• <span style="color:#cccc00;">黄色</span> = 中等应力<br>';
        html += '• <span style="color:#cc3300;">红色</span> = 高应力（接近屈服）<br>';
        html += '• 来源：离心力 + 压力 + 热应力';
        html += '</div></div>';
    }

    if (c.warnings && c.warnings.length > 0) {
        html += '<div class="info-section"><h4>⚠️ 警告</h4>';
        c.warnings.forEach(function(w) { html += '<div class="info-row" style="color:var(--warning);font-size:11px;">\u26A0 ' + zhMessage(w) + '</div>'; });
        html += '</div>';
    }

    el.innerHTML = html;
}

// ──── TPMS Lattice ────
async function toggleLattice() {
    state.latticeVisible = !state.latticeVisible;
    document.getElementById('btn-lattice').classList.toggle('active', state.latticeVisible);

    if (state.latticeVisible) {
        if (Object.keys(state.latticeMeshes).length === 0) {
            await loadLatticeMeshes();
        } else {
            for (var id in state.latticeMeshes) {
                state.latticeMeshes[id].visible = true;
            }
        }
        // Make solid components semi-transparent to reveal lattice
        for (var cid in state.components) {
            if (cid === 'combustor' || cid === 'nozzle') {
                state.components[cid].mesh.material.transparent = true;
                state.components[cid].mesh.material.opacity = 0.15;
                state.components[cid].mesh.material.needsUpdate = true;
            }
        }
        showBadge('TPMS 点阵', '#FF6D00');
        renderLatticeInfo();
    } else {
        for (var id in state.latticeMeshes) {
            state.latticeMeshes[id].visible = false;
        }
        // Restore opacity
        for (var cid in state.components) {
            if (cid === 'combustor' || cid === 'nozzle') {
                var op = parseFloat(document.getElementById('slider-opacity').value);
                state.components[cid].mesh.material.transparent = op < 1;
                state.components[cid].mesh.material.opacity = op;
                state.components[cid].mesh.material.needsUpdate = true;
            }
        }
        document.getElementById('sim-badge').style.display = 'none';
        loadEngineData();
    }
}

async function loadLatticeMeshes(variation) {
    var varKey = variation || state.currentVariation || 'v1_gyroid_standard';
    state.currentVariation = varKey;

    document.getElementById('status-left').textContent = '正在加载 TPMS 点阵…';
    var loader = new STLLoader();

    // Determine file paths based on variation
    var basePath = varKey === 'default' ? '/stl/' : '/stl/variation/' + varKey + '/';
    var latticeFiles = [
        { id: 'combustor_lattice', file: basePath + 'combustor_lattice.stl', color: '#FF6D00', label: 'Combustor' },
        { id: 'nozzle_lattice',    file: basePath + 'nozzle_lattice.stl',    color: '#FF3D00', label: 'Nozzle' },
    ];

    for (var i = 0; i < latticeFiles.length; i++) {
        var lf = latticeFiles[i];
        try {
            var geometry = await new Promise(function(resolve, reject) {
                loader.load(lf.file, resolve, undefined, reject);
            });
            geometry.computeVertexNormals();

            var material = new THREE.MeshPhysicalMaterial({
                color: new THREE.Color(lf.color),
                metalness: 0.85,
                roughness: 0.2,
                clearcoat: 0.3,
                side: THREE.DoubleSide,
                clippingPlanes: state.sectionActive ? [state.clippingPlane] : [],
            });

            var mesh = new THREE.Mesh(geometry, material);
            mesh.castShadow = true;
            state.scene.add(mesh);
            state.latticeMeshes[lf.id] = mesh;

            var vCount = geometry.attributes.position.count;
            document.getElementById('status-left').textContent = '已加载' + zhComponent(lf.id, lf.label) + '（' + formatNum(Math.round(vCount)) + ' 个顶点）';
        } catch(e) {
            console.warn('Lattice load failed: ' + lf.id, e);
        }
    }
    document.getElementById('status-left').textContent = 'TPMS 点阵已加载';
}

async function switchVariation(varKey) {
    // Remove existing lattice meshes
    for (var id in state.latticeMeshes) {
        state.scene.remove(state.latticeMeshes[id]);
        if (state.latticeMeshes[id].geometry) state.latticeMeshes[id].geometry.dispose();
        if (state.latticeMeshes[id].material) state.latticeMeshes[id].material.dispose();
    }
    state.latticeMeshes = {};
    state.currentVariation = varKey;

    // Load new variation
    await loadLatticeMeshes(varKey);

    // Update active button
    document.querySelectorAll('.var-btn').forEach(function(btn) {
        btn.classList.toggle('active', btn.dataset.var === varKey);
    });

    // Re-render info panel
    renderLatticeInfo();
}

function renderLatticeInfo() {
    var el = document.getElementById('engine-info');
    var html = '';
    var varKey = state.currentVariation || 'v1_gyroid_standard';

    var variations = {
        'v1_gyroid_standard': { label: 'V1 螺旋曲面', desc: '兼顾传热与强度', type: '螺旋曲面', eq: 'sin(x)cos(y)+sin(y)cos(z)+sin(z)cos(x)=t', cell: '14', color: '#FF6D00' },
        'v2_gyroid_dense':    { label: 'V2 高密度',  desc: '换热能力更强，质量更大',    type: '高密度螺旋曲面', eq: 'sin(x)cos(y)+sin(y)cos(z)+sin(z)cos(x)=t', cell: '10', color: '#FF8F00' },
        'v3_schwarz_p':       { label: 'V3 Schwarz 曲面', desc: '直通式流道',   type: 'Schwarz-P',     eq: 'cos(x)+cos(y)+cos(z)=t',                  cell: '14', color: '#00E5FF' },
        'v4_diamond':         { label: 'V4 钻石曲面', desc: '最高比刚度',      type: '钻石曲面',       eq: 'sin·sin·sin + sin·cos·cos + ...',          cell: '14', color: '#76FF03' },
    };

    var cur = variations[varKey] || variations['v1_gyroid_standard'];

    // Variation selector buttons
    html += '<div class="info-section"><h4>🔬 TPMS 设计变体</h4>';
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin:6px 0;">';
    for (var vk in variations) {
        var v = variations[vk];
        var isActive = vk === varKey;
        var style = isActive
            ? 'background:' + v.color + '22;border:1px solid ' + v.color + ';color:' + v.color
            : 'background:var(--surface);border:1px solid var(--border);color:var(--text-secondary)';
        html += '<button class="var-btn' + (isActive ? ' active' : '') + '" data-var="' + vk + '" '
             + 'onclick="switchVariation(\'' + vk + '\')" '
             + 'style="' + style + ';padding:6px 4px;border-radius:6px;cursor:pointer;font-size:10px;font-weight:600;'
             + 'transition:all 0.2s;text-align:center;">'
             + v.label + '</button>';
    }
    html += '</div>';
    html += '<div style="color:var(--text-tertiary);font-size:10px;margin-top:4px;">' + cur.desc + '</div>';
    html += '</div>';

    // Current variant details
    html += '<div class="info-section"><h4>' + cur.type + '</h4>';
    html += infoRow('方程', cur.eq);
    html += infoRow('单元尺寸', cur.cell + ' mm');
    html += infoRow('壁面类型', '实体壳层（厚壁）', 'accent');
    html += infoRow('平滑', '拉普拉斯平滑（2 次迭代）');
    html += '</div>';

    html += '<div class="info-section"><h4>优势</h4>';
    html += '<div class="info-row" style="color:var(--success);">✓ 最多减重 60%</div>';
    html += '<div class="info-row" style="color:var(--success);">✓ 换热表面积扩大 3 倍</div>';
    html += '<div class="info-row" style="color:var(--success);">✓ 支持金属三维打印自支撑</div>';
    html += '<div class="info-row" style="color:var(--success);">✓ 位于发动机壁内</div>';
    html += '<div class="info-row" style="color:var(--success);">✓ 无大于 45° 的悬垂（适配 DMLS）</div>';
    html += '</div>';

    html += '<div class="info-section"><h4>制造</h4>';
    html += infoRow('工艺', 'DMLS / SLM 金属三维打印');
    html += infoRow('材料', 'Inconel 718（燃烧室）');
    html += infoRow('最小特征', '0.4 mm 壁厚');
    html += infoRow('打印方向', '竖直（Z 轴为发动机轴线）');
    html += '</div>';

    el.innerHTML = html;
}

// Expose to global scope for inline onclick handlers
window.switchVariation = switchVariation;

// Start
init();
