// Presentation-only translations. API keys, enum values and engineering units stay unchanged.
const components = {
    inlet: '进气道', compressor: '压气机', combustor: '燃烧室', turbine: '涡轮',
    nozzle: '喷管', shaft: '主轴', casing: '机匣', transitions: '过渡段',
    combustor_lattice: '燃烧室点阵', nozzle_lattice: '喷管点阵', misc: '其他',
};

const variables = {
    compressor_impeller_exit_diameter_m: '压气机叶轮出口直径',
    compressor_inducer_tip_diameter_m: '压气机诱导轮叶尖直径',
    compressor_inducer_hub_tip_ratio: '压气机诱导轮轮毂叶尖比',
    compressor_impeller_exit_width_m: '压气机叶轮出口宽度',
    compressor_exit_blade_angle_deg: '压气机出口叶片角',
    compressor_blade_count: '压气机叶片数',
    combustor_outer_diameter_m: '燃烧室外径',
    combustor_inner_diameter_m: '燃烧室内径',
    combustor_length_m: '燃烧室长度',
    combustor_liner_thickness_m: '燃烧室衬套厚度',
    combustor_injector_count: '燃烧室喷嘴数',
    turbine_tip_diameter_m: '涡轮叶尖直径',
    turbine_hub_tip_ratio: '涡轮轮毂叶尖比',
    turbine_blade_count: '涡轮叶片数',
    turbine_inlet_total_temperature_K: '涡轮进口总温',
    nozzle_throat_diameter_m: '喷管喉部直径',
    shaft_speed_rpm: '主轴转速',
    lattice_cell_size_m: '点阵单元尺寸',
    lattice_relative_density: '点阵相对密度',
    diffuser_area_ratio: '扩压器面积比',
    combustor_primary_air_fraction: '燃烧室一次空气比例',
    compressor_tip_clearance_ratio: '压气机叶尖间隙比',
};

const constraints = {
    combustor_injector_capacity: '燃烧室喷嘴容量',
    combustor_injector_pitch_min: '燃烧室喷嘴间距下限',
    combustor_lattice_skin_thickness_min: '燃烧室点阵表皮厚度下限',
    combustor_liner_hoop_stress: '燃烧室衬套环向应力',
    combustor_map_coordinate_domain: '燃烧室特性图坐标范围',
    combustor_map_lean_stability: '燃烧室贫油稳定性',
    combustor_map_rich_stability: '燃烧室富油稳定性',
    combustor_metal_temperature: '燃烧室金属温度',
    combustor_pressure_loss_fraction_max: '燃烧室压损比例上限',
    combustor_primary_residence_time_min: '燃烧室一次区停留时间下限',
    combustor_to_turbine_outer_diameter: '燃烧室至涡轮外径匹配',
    compressor_efficiency_min: '压气机效率下限',
    compressor_map_choke_boundary: '压气机特性图阻塞边界',
    compressor_map_coordinate_domain: '压气机特性图坐标范围',
    compressor_map_surge_boundary: '压气机特性图喘振边界',
    compressor_metal_temperature: '压气机金属温度',
    compressor_rotating_stress: '压气机旋转应力',
    compressor_tip_mach_max: '压气机叶尖马赫数上限',
    compressor_tip_speed_limit: '压气机叶尖速度限值',
    diffuser_to_combustor_area: '扩压器至燃烧室面积匹配',
    inlet_map_coordinate_domain: '进气道特性图坐标范围',
    inlet_map_maximum_capture_area: '进气道捕获面积上限',
    inlet_map_minimum_capture_area: '进气道捕获面积下限',
    inlet_total_enthalpy_closure: '进气道总焓闭合',
    lattice_cell_size_min: '点阵单元尺寸下限',
    lattice_cells_across_region_min: '区域横向点阵单元数下限',
    lattice_relative_density_max: '点阵相对密度上限',
    lattice_relative_density_min: '点阵相对密度下限',
    nozzle_exit_enthalpy_closure: '喷管出口焓闭合',
    nozzle_lattice_skin_thickness_min: '喷管点阵表皮厚度下限',
    nozzle_map_coordinate_domain: '喷管特性图坐标范围',
    nozzle_map_maximum_reynolds: '喷管特性图雷诺数上限',
    nozzle_map_minimum_reynolds: '喷管特性图雷诺数下限',
    nozzle_total_pressure_usable: '喷管可用总压',
    positive_thrust: '正推力', positive_tsfc: '正单位推力油耗',
    shaft_clearance: '主轴间隙',
    turbine_efficiency_min: '涡轮效率下限',
    turbine_inlet_temperature_max: '涡轮进口温度上限',
    turbine_inlet_temperature_min: '涡轮进口温度下限',
    turbine_map_coordinate_domain: '涡轮特性图坐标范围',
    turbine_map_maximum_capacity: '涡轮特性图容量上限',
    turbine_map_minimum_capacity: '涡轮特性图容量下限',
    turbine_mean_radius_mach_max: '涡轮平均半径马赫数上限',
    turbine_metal_temperature: '涡轮金属温度',
    turbine_phi_max: '涡轮流量系数上限', turbine_phi_min: '涡轮流量系数下限',
    turbine_psi_max: '涡轮负荷系数上限',
    turbine_rotating_stress: '涡轮旋转应力',
    turbine_solidity_max: '涡轮稠度上限', turbine_solidity_min: '涡轮稠度下限',
    turbine_tip_speed_limit: '涡轮叶尖速度限值',
    turbine_to_nozzle: '涡轮至喷管匹配',
    whole_engine_flow_match: '整机流量匹配',
};

const names = {
    'Air Inlet': '进气道', Compressor: '压气机', 'Combustion Chamber': '燃烧室',
    'Turbine Stage': '涡轮级', 'Exhaust Nozzle': '排气喷管', 'Main Shaft': '主轴',
    'Outer Casing': '外机匣', Transitions: '过渡段',
    'Combustor Lattice': '燃烧室点阵', 'Nozzle Lattice': '喷管点阵',
    Ambient: '环境', Inlet: '进气道', Combustor: '燃烧室', Turbine: '涡轮', Nozzle: '喷管',
    'Diatomite-Silica': '硅藻土-二氧化硅', 'Prismatic Chitin': '棱柱几丁质',
    'Nacre-Layered': '珍珠层结构', 'Ceramic Aerogel': '陶瓷气凝胶',
    'YSZ Standard': '标准 YSZ', 'YSZ EB-PVD': 'YSZ 电子束物理气相沉积',
    'Gadolinium Zirconate': '锆酸钆',
    'YSZ (7% Yttria-Stabilized Zirconia)': 'YSZ（7% 氧化钇稳定氧化锆）',
    'YSZ EB-PVD (Columnar)': 'YSZ 电子束物理气相沉积（柱状）',
    'Gadolinium Zirconate (Gd2Zr2O7)': '锆酸钆（Gd2Zr2O7）',
    'Diatomite-Silica Composite': '硅藻土-二氧化硅复合材料',
    'Prismatic Chitin-Ceramic Hybrid': '棱柱几丁质-陶瓷复合材料',
    'Nacre-Inspired Layered Ceramic': '仿珍珠层陶瓷',
    'Ceramic Aerogel Coating': '陶瓷气凝胶涂层',
};

const states = {
    queued: '排队中', running: '运行中', complete: '已完成', failed: '失败',
    interrupted: '已中断', pending: '待处理', ready: '就绪',
    delivered: '已交付', accepted: '已接受', rejected: '已拒绝',
    NO_HIT: '未命中目标', NO_CANDIDATES: '无候选设计',
    routable: '可路由', abstained_no_target_support: '目标缺少支持，已弃权',
    not_completed: '未完成', development_diagnostic: '开发版诊断',
};

const tokens = {
    compressor: '压气机', combustor: '燃烧室', turbine: '涡轮', nozzle: '喷管',
    shaft: '主轴', lattice: '点阵', diffuser: '扩压器',
    primary: '一次', air: '空气', tip: '叶尖', clearance: '间隙', blade: '叶片',
    flow: '流量', mass: '质量', thrust: '推力', pressure: '压力', temperature: '温度',
    diameter: '直径', ratio: '比', area: '面积', residual: '残差', efficiency: '效率',
    speed: '转速', density: '密度', relative: '相对', length: '长度', outer: '外',
    inner: '内', injector: '喷嘴', count: '数量', thickness: '厚度', total: '总',
    exit: '出口', inlet: '入口', geometry: '几何', mesh: '网格', feasible: '可行',
    limit: '限制', constraint: '约束', minimum: '下限', maximum: '上限',
};

export function zhComponent(id, fallback = '') {
    return components[id] || names[fallback] || fallback || id;
}

export function zhName(value) {
    return names[value] || value;
}

export function zhState(value) {
    return states[value] || value;
}

export function zhVariable(value) {
    return variables[value] || zhIdentifier(value);
}

export function zhIdentifier(value) {
    if (variables[value]) return variables[value];
    if (constraints[value]) return constraints[value];
    if (value === 'geometry19_wp19_8_bootstrap_complete') return 'Geometry19 初始闭环评估已完成';
    const parts = String(value).split('.');
    if (parts.length > 1) {
        const group = {engineering_soft: '工程软约束', numerical_hard: '数值硬约束', physics_domain: '物理适用域',
            structural: '结构', thermodynamic: '热力', thermal: '热', assembly: '装配', manufacturing: '制造'};
        return parts.map(part => group[part] || constraints[part] || zhIdentifier(part)).join(' · ');
    }
    const words = String(value).split('_');
    if (words.every(word => tokens[word])) return words.map(word => tokens[word]).join('');
    return String(value);
}

export function zhUnit(value) {
    return {dimensionless: '无量纲', count: '个', deg: '°', rpm: '转/分'}[value] || value;
}

export function zhMessage(value) {
    const source = String(value || '');
    const known = {
        'Negative fuel flow — TIT too low or efficiency too low': '燃油流量为负：涡轮进口温度或效率过低',
        'Invalid turbine exit conditions': '涡轮出口工况无效',
        'Nozzle is choked (sonic exit)': '喷管达到阻塞状态（出口声速）',
        'Training started from UI samples': '已开始使用界面样本训练',
        'No training started': '尚未开始训练',
        'No CFD calibration started': '尚未开始 CFD 标定',
    };
    if (known[source]) return known[source];
    const patterns = [
        [/^TIT=(\d+)K exceeds typical micro turbojet limits$/, '涡轮进口温度 $1 K 超出典型微型涡喷发动机限值'],
        [/^Exhaust temp (\d+)K is very high$/, '排气温度 $1 K 过高'],
        [/^De Haller ([\d.]+) < 0\.72 — high stall risk!$/, 'De Haller 数 $1 小于 0.72：失速风险高'],
        [/^De Haller ([\d.]+) — moderate stall margin$/, 'De Haller 数 $1：失速裕度中等'],
        [/^D-factor ([\d.]+) > 0\.6 — boundary layer separation likely$/, '扩压因子 $1 大于 0.6：可能发生边界层分离'],
        [/^D-factor ([\d.]+) — near design limit$/, '扩压因子 $1：接近设计限值'],
        [/^Reaction ([\d.]+) — impulse-heavy, low efficiency$/, '反力度 $1：冲动成分较高，效率偏低'],
        [/^Reaction ([\d.]+) — rotor does most work, check stall$/, '反力度 $1：转子承担大部分做功，请检查失速风险'],
        [/^Zweifel ([\d.]+) < 0\.6 — under-loaded, add blades or increase turning$/, 'Zweifel 系数 $1 小于 0.6：负荷偏低，可增加叶片或转角'],
        [/^Zweifel ([\d.]+) > 1\.1 — over-loaded, boundary layer separation risk$/, 'Zweifel 系数 $1 大于 1.1：负荷过高，存在边界层分离风险'],
        [/^Zweifel ([\d.]+) — good blade loading range$/, 'Zweifel 系数 $1：叶片负荷处于合适范围'],
        [/^Reaction ([\d.]+) — nearly impulse design$/, '反力度 $1：接近冲动式设计'],
        [/^Reaction ([\d.]+) — high reaction, check rotor inlet conditions$/, '反力度 $1：反动度较高，请检查转子进口工况'],
    ];
    for (const [pattern, replacement] of patterns) {
        if (pattern.test(source)) return source.replace(pattern, replacement);
    }
    return source;
}
