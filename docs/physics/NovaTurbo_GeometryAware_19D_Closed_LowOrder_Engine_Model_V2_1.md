# NovaTurbo Geometry-aware 19D 闭合低阶整机模型 V2.1

> **定位**：用于 NovaTurbo 从 active-10D 扩展到真正 Geometry-aware 19D 的前向物理评价器。  
> **用途**：只在参数空间采样，不依赖 CFD；为 DOE 数据生成、前向代理模型和条件反向生成器提供统一物理真值。  
> **逆设计条件固定为**：
>
> \[
> \boxed{c^*=[F^*,TSFC^*,M^*]}
> \]
>
> **V2.1 结论**：数学接口可进入编码与单点验证；在 calibration pack 的数值、适用范围和来源全部冻结，并通过守恒测试与小规模 DOE 前，**不得用于正式大规模 19D 数据生成**。

---

# 1. 总体接口

## 1.1 前向物理模型

\[
\boxed{
\mathcal P:(x_{19},e,\theta)
\mapsto
(F,TSFC,M,\mathbf g,status)
}
\]

其中：

- \(x_{19}\)：19D 几何与设计点运行参数；
- \(e\)：环境条件；
- \(\theta\)：固定模型/标定参数；
- \(\mathbf g\)：约束残差，统一约定 \(g_j\le0\) 为满足；
- `status`：收敛、失败类型和 failure code。

## 1.2 逆设计模型

\[
\boxed{
x_{19}=G(F^*,TSFC^*,M^*,z)
}
\]

候选必须回代：

\[
[\hat F,\widehat{TSFC},\hat M,\mathbf g]
=
\mathcal P(x_{19})
\]

并计算：

\[
e_F=\frac{|\hat F-F^*|}{F^*}
\]

\[
e_T=
\frac{|\widehat{TSFC}-TSFC^*|}{TSFC^*}
\]

\[
e_M=
\frac{|\hat M-M^*|}{M^*}
\]

最终交付必须同时满足三个目标误差阈值和：

\[
\boxed{\mathbf g\le0}
\]

---

# 2. 冻结后的 19D

\[
\boxed{
\begin{aligned}
x_{19}=[
&D_{c2},D_{c1t},\lambda_{c1},b_{c2},\beta_{c2},Z_c,\\
&D_{bo},D_{bi},L_b,t_b,Z_{\mathrm{inj}},\\
&D_t,\lambda_t,Z_t,T_{t4},\\
&D_*,N,a_{\mathrm{lat}},\bar\rho_{\mathrm{lat}}
]
\end{aligned}}
\]

| # | 参数 | 含义 |
|---:|---|---|
| 1 | \(D_{c2}\) | 压气机叶轮出口直径 |
| 2 | \(D_{c1t}\) | 压气机进口叶尖直径 |
| 3 | \(\lambda_{c1}\) | 压气机进口 hub-tip ratio |
| 4 | \(b_{c2}\) | 压气机出口叶宽 |
| 5 | \(\beta_{c2}\) | 压气机出口叶片金属角 |
| 6 | \(Z_c\) | 压气机叶片数 |
| 7 | \(D_{bo}\) | 燃烧室外径 |
| 8 | \(D_{bi}\) | 燃烧室内径 |
| 9 | \(L_b\) | 燃烧室长度 |
| 10 | \(t_b\) | liner 壁厚 |
| 11 | \(Z_{\rm inj}\) | 喷油嘴数 |
| 12 | \(D_t\) | 涡轮叶尖直径 |
| 13 | \(\lambda_t\) | 涡轮 hub-tip ratio |
| 14 | \(Z_t\) | 涡轮叶片数 |
| 15 | \(T_{t4}\) | TIT，设计点运行参数 |
| 16 | \(D_*\) | 收敛喷管最小面积直径 |
| 17 | \(N\) | 单轴转速，rpm |
| 18 | \(a_{\rm lat}\) | 点阵胞元尺寸 |
| 19 | \(\bar\rho_{\rm lat}\) | 点阵相对密度 |

由模型派生，不再独立采样：

\[
\boxed{
\dot m_a,\pi_c,\eta_c,\eta_t,
\dot m_f,AFR,P_{t5},T_{t5}
}
\]

---

# 3. 环境与统一参考状态

环境：

\[
e=[T_0,P_0,M_0]
\]

其中进气总压恢复系数 \(\sigma_i\) 不属于逐样本环境输入，而是由
calibration pack 冻结的模型参数 `inlet_total_pressure_recovery`。因此同一
calibration pack 内所有设计点共享其定义；若需要研究进气畸变或可变恢复，
必须通过新的环境/不确定性接口显式升版，不能在运行时静默覆盖。

入口总温：

\[
T_{t0}
=
T_0\left(1+\frac{\gamma_c-1}{2}M_0^2\right)
\]

入口总压：

\[
P_{t0}
=
P_0
\left(1+\frac{\gamma_c-1}{2}M_0^2\right)^{
\gamma_c/(\gamma_c-1)}
\]

进气总压恢复：

\[
\boxed{
T_{t2}=T_{t0},\qquad
P_{t2}=\sigma_iP_{t0}
}
\]

飞行速度：

\[
V_0=M_0\sqrt{\gamma_cR_cT_0}
\]

台架状态可令：

\[
M_0=0
\]

---

# 4. 热物性与燃烧焓参考

V2.1 可以继续采用常比热低阶模型，但必须冻结共同焓参考状态：

\[
\boxed{T_{\rm ref,h}}
\]

空气：

\[
h_a(T)
=
h_{a,\rm ref}
+
\int_{T_{\rm ref,h}}^T c_{p,a}(T)\,dT
\]

燃气：

\[
h_g(T)
=
h_{g,\rm ref}
+
\int_{T_{\rm ref,h}}^T c_{p,g}(T)\,dT
\]

燃料：

\[
h_f(T_f)
\]

必须采用同一参考体系。

若采用常 \(c_p\)：

\[
h_a(T)=c_{p,c}(T-T_{\rm ref,h})
\]

\[
h_g(T)=c_{p,h}(T-T_{\rm ref,h})
\]

并在 calibration/config 中冻结：

\[
R_c,R_g,c_{p,c},c_{p,h},
\gamma_c,\gamma_h,LHV
\]

---

# 5. Geometry Derivation Layer

这里的 Geometry 是解析几何派生，不是 STL/CAD。

\[
\boxed{
x_{\rm geom}
\rightarrow
A,V,R,h,U,\sigma,\ldots
}
\]

## 5.1 压气机

\[
D_{c1h}
=
\lambda_{c1}D_{c1t}
\]

\[
A_{c1}
=
\frac{\pi}{4}
(D_{c1t}^2-D_{c1h}^2)
\]

\[
r_{c2}=\frac{D_{c2}}2
\]

\[
A_{c2,g}
=
2\pi r_{c2}b_{c2}
\]

角速度：

\[
\boxed{
\omega=\frac{2\pi N}{60}
}
\]

\[
U_{c2}
=
\omega r_{c2}
=
\frac{\pi D_{c2}N}{60}
\]

定义叶片厚度 closure：

\[
t_{c2}=k_{tc}b_{c2}
\]

出口 blockage：

\[
B_{c2}
=
1-
\frac{
Z_ct_{c2}
}{
2\pi r_{c2}\sin\beta_{c2}
}
\]

\[
\boxed{
A_{c2}=B_{c2}A_{c2,g}
}
\]

必须：

\[
0<B_{c2}<1
\]

派生扩压器不增加19D自由度：

\[
\boxed{
A_{d,out}=\Pi_{A,d}A_{c2},
\qquad
L_d=k_{Ld}D_{c2}
}
\]

## 5.2 燃烧室

\[
A_b
=
\frac{\pi}{4}
(D_{bo}^2-D_{bi}^2)
\]

\[
V_b=A_bL_b
\]

\[
D_{h,b}=D_{bo}-D_{bi}
\]

\[
D_{bm}=\frac{D_{bo}+D_{bi}}2
\]

\[
s_{\rm inj}
=
\frac{\pi D_{bm}}{Z_{\rm inj}}
\]

## 5.3 涡轮

\[
R_t=\frac{D_t}{2}
\]

\[
R_h=\lambda_tR_t
\]

\[
R_m=\frac{R_t+R_h}{2}
\]

\[
h_t=R_t-R_h
\]

\[
A_t=
\pi(R_t^2-R_h^2)
\]

\[
U_m=\omega R_m
\]

\[
U_{t,\rm tip}=\omega R_t
\]

固定展弦比 closure：

\[
c_t=\frac{h_t}{AR_t}
\]

\[
s_t=\frac{2\pi R_m}{Z_t}
\]

\[
\boxed{
\sigma_t=\frac{c_t}{s_t}
=
\frac{Z_tc_t}{2\pi R_m}
}
\]

## 5.4 喷管

纯收敛喷管：

\[
\boxed{
A_*=A_e=
\frac{\pi D_*^2}{4}
}
\]

---

# 6. 压气机低阶模型

输入：

\[
[
\dot m_a,T_{t2},P_{t2},
D_{c2},D_{c1t},\lambda_{c1},
b_{c2},\beta_{c2},Z_c,N
]
\]

输出：

\[
[
T_{t3},P_{t3},\pi_c,\eta_c,
\Delta h_c,P_c,
\phi_{c,\rm ref},\psi_c,Ma_{U,c},Re_c
]
\]

## 6.1 模型范围

\[
\boxed{
\eta_c=
\text{叶轮 + 扩压器 + 混合等未单独建模损失的整台压气机总—总效率}
}
\]

因此 V2.1 不单独建 diffuser loss；其影响由 calibration pack 吸收。

## 6.2 参考流量系数

采用入口总状态参考密度：

\[
\rho_{c,\rm ref}
=
\frac{P_{t2}}{R_cT_{t2}}
\]

定义：

\[
\boxed{
C_{m2,\rm ref}
=
\frac{\dot m_a}
{\rho_{c,\rm ref}A_{c2}}
}
\]

以及：

\[
\boxed{
\phi_{c,\rm ref}
=
\frac{C_{m2,\rm ref}}{U_{c2}}
}
\]

这是**低阶等效流量系数**，不能直接解释成真实叶轮出口静态速度。

## 6.3 滑移因子

采用 Stanitz 型 baseline：

\[
\boxed{
\sigma_s
=
1-\frac{0.63\pi}{Z_c}
}
\]

要求：

\[
0<\sigma_s<1
\]

## 6.4 速度三角形与 Euler 功

入口无预旋：

\[
C_{\theta1}\simeq0
\]

\[
C_{\theta2}
=
\sigma_s
\left(
U_{c2}
-
C_{m2,\rm ref}\cot\beta_{c2}
\right)
\]

Euler 比功：

\[
\boxed{
\Delta h_c
=
U_{c2}C_{\theta2}
}
\]

负荷系数：

\[
\boxed{
\psi_c
=
\frac{\Delta h_c}{U_{c2}^2}
}
\]

必须：

\[
\Delta h_c>0
\]

## 6.5 Mach 与 Reynolds

\[
Ma_{U,c}
=
\frac{U_{c2}}
{\sqrt{\gamma_cR_cT_{t2}}}
\]

黏度使用 Sutherland 关系：

\[
\mu(T)
=
\mu_{\rm ref}
\left(\frac{T}{T_{\rm ref}}\right)^{3/2}
\frac{T_{\rm ref}+S}{T+S}
\]

\[
Re_c
=
\frac{
\rho_{c,\rm ref}U_{c2}D_{c2}
}{
\mu(T_{t2})
}
\]

## 6.6 压气机效率 closure

定义：

\[
\delta_{\phi c}
=
\frac{\phi_{c,\rm ref}-\phi_{c,d}}{\phi_{c,d}}
\]

\[
\delta_{\psi c}
=
\frac{\psi_c-\psi_{c,d}}{\psi_{c,d}}
\]

\[
\delta_{Mc}
=
\max
\left[
0,
\frac{Ma_{U,c}-Ma_{U,c,\lim}}
{Ma_{U,c,\lim}}
\right]
\]

\[
\delta_{Rec}
=
\max
\left[
0,
\frac{Re_{c,d}}{Re_c}-1
\right]
\]

\[
\boxed{
\eta_c
=
\eta_{c,d}
-
a_{\phi c}\delta_{\phi c}^2
-
a_{\psi c}\delta_{\psi c}^2
-
a_{Mc}\delta_{Mc}^2
-
a_{Rec}\delta_{Rec}^2
}
\]

这是 NovaTurbo 的可标定 closure，不是通用教科书公式。

若：

\[
\eta_c\notin(0,1)
\]

返回：

```text
COMPRESSOR_EFFICIENCY_INVALID
```

不得 clipping。

## 6.7 出口总状态

\[
T_{t3}
=
T_{t2}
+
\frac{\Delta h_c}{c_{p,c}}
\]

\[
T_{t3s}
=
T_{t2}
+
\eta_c
\frac{\Delta h_c}{c_{p,c}}
\]

\[
\boxed{
\pi_c
=
\left(
\frac{T_{t3s}}{T_{t2}}
\right)^{
\gamma_c/(\gamma_c-1)
}
}
\]

\[
P_{t3}=\pi_cP_{t2}
\]

压气机功率：

\[
\boxed{
P_c=\dot m_a\Delta h_c
}
\]

---

# 7. 燃烧室低阶模型

输入：

\[
[
\dot m_a,T_{t3},P_{t3},T_{t4},
D_{bo},D_{bi},L_b,t_b,Z_{\rm inj}
]
\]

输出：

\[
[
P_{t4},\Delta P_{tb},
f,\dot m_f,\dot m_g,AFR,
\tau_{b,p},\varphi,\dot m_{f,\rm inj}
]
\]

## 7.1 参考速度

\[
\rho_{3,\rm ref}
=
\frac{P_{t3}}{R_cT_{t3}}
\]

\[
C_{\rm ref}
=
\frac{\dot m_a}
{\rho_{3,\rm ref}A_b}
\]

## 7.2 总压损失

\[
K_b
=
K_{b0}
+
K_{bL}\frac{L_b}{D_{h,b}}
+
K_{bA}
\left(
\frac{A_{d,out}}{A_b}-1
\right)^2
\]

\[
\Delta P_{tb}
=
K_b
\frac12\rho_{3,\rm ref}C_{\rm ref}^2
\]

\[
\delta_b=
\frac{\Delta P_{tb}}{P_{t3}}
\]

\[
\boxed{
P_{t4}=P_{t3}-\Delta P_{tb}
}
\]

## 7.3 燃烧能量平衡

通用：

\[
\dot m_ah_a(T_{t3})
+
\dot m_f[
\eta_bLHV+h_f(T_f)]
=
(\dot m_a+\dot m_f)h_g(T_{t4})
\]

因此：

\[
\boxed{
f=
\frac{
h_g(T_{t4})-h_a(T_{t3})
}{
\eta_bLHV+h_f(T_f)-h_g(T_{t4})
}
}
\]

常 \(c_p\) 且共同参考零点时：

\[
\boxed{
f
=
\frac{
c_{p,h}(T_{t4}-T_{\rm ref,h})
-
c_{p,c}(T_{t3}-T_{\rm ref,h})
}{
\eta_bLHV
-
c_{p,h}(T_{t4}-T_{\rm ref,h})
}
}
\]

然后：

\[
\dot m_f=f\dot m_a
\]

\[
AFR=\frac1f
\]

\[
\dot m_g=(1+f)\dot m_a
\]

若无正燃油解，返回：

```text
COMBUSTOR_ENERGY_INVALID
```

## 7.4 当量比

当前实现使用全部压气机空气流量，因此这是整机燃烧室的全局当量比：

\[
\boxed{
\varphi_{\rm global}=\frac{f}{f_{\rm st}}
}
\]

它不是仅以 primary-zone 空气计算的局部主燃区当量比，不能直接与文献中的
\(\varphi_{\rm primary}\approx0.8\sim1.0\) 数值对比。

正式约束：

\[
\varphi_{\min}
\le\varphi_{\rm global}
\le\varphi_{\max}
\]

## 7.5 Primary-zone 停留时间

\[
\boxed{
V_{b,p}
=
\chi_{V,p}V_b
}
\]

\[
\dot m_{a,p}
=
\chi_{\dot m,p}\dot m_a
\]

\[
\dot m_{g,p}
=
\dot m_{a,p}+\dot m_f
\]

\[
\bar T_{b,p}
=
\frac{T_{t3}+T_{t4}}2
\]

\[
\bar P_{b,p}
=
\frac{P_{t3}+P_{t4}}2
\]

\[
\bar\rho_{b,p}
=
\frac{\bar P_{b,p}}
{R_g\bar T_{b,p}}
\]

\[
\boxed{
\tau_{b,p}
=
\frac{
\bar\rho_{b,p}V_{b,p}
}{
\dot m_{g,p}
}
}
\]

必须：

\[
\tau_{b,p}\ge\tau_{b,p,\min}
\]

## 7.6 喷油嘴容量

\[
\dot m_{f,\rm inj}
=
\frac{\dot m_f}{Z_{\rm inj}}
\]

若喷嘴参数冻结：

\[
\dot m_{f,\rm inj,max}
=
C_{d,\rm inj}
A_{\rm inj}
\sqrt{2\rho_f\Delta P_f}
\]

要求：

\[
\dot m_{f,\rm inj}
\le
\dot m_{f,\rm inj,max}
\]

以及：

\[
s_{\rm inj}\ge s_{\rm inj,min}
\]

---

# 8. 涡轮与轴功模型

输入：

\[
[
\dot m_g,T_{t4},P_{t4},
P_c,P_{\rm aux},
D_t,\lambda_t,Z_t,N
]
\]

输出：

\[
[
T_{t5},P_{t5},\eta_t,
\Delta h_t,
\phi_t,\psi_t,\sigma_t,
Ma_{U,t},Re_t,\Gamma_t
]
\]

## 8.1 边界说明

V2.1 没有真实 turbine map。

因此其准确名称是：

\[
\boxed{
\text{具有正式涡轮工作域/吞流容量筛选的低阶整机模型}
}
\]

而不是完整 map-based component matching model。

## 8.2 轴功平衡

\[
\eta_m\dot m_g\Delta h_t
=
P_c+P_{\rm aux}
\]

因此：

\[
\boxed{
\Delta h_t
=
\frac{P_c+P_{\rm aux}}
{\eta_m\dot m_g}
}
\]

\[
\boxed{
T_{t5}
=
T_{t4}
-
\frac{\Delta h_t}{c_{p,h}}
}
\]

## 8.3 流量与负荷系数

\[
\rho_{t,\rm ref}
=
\frac{P_{t4}}{R_gT_{t4}}
\]

\[
C_{ax,t}
=
\frac{\dot m_g}
{\rho_{t,\rm ref}A_t}
\]

\[
\boxed{
\phi_t=\frac{C_{ax,t}}{U_m}
}
\]

\[
\boxed{
\psi_t=
\frac{\Delta h_t}{U_m^2}
}
\]

## 8.4 Mach、Re 与吞流容量代理

\[
Ma_{U,t}
=
\frac{U_m}
{\sqrt{\gamma_hR_gT_{t4}}}
\]

\[
Re_t
=
\frac{
\rho_{t,\rm ref}C_{ax,t}c_t
}{
\mu_h(T_{t4})
}
\]

定义：

\[
\boxed{
\Gamma_t
=
\frac{
\dot m_g\sqrt{T_{t4}/T_{\rm ref}}
}{
(P_{t4}/P_{\rm ref})A_t
}
}
\]

正式可行性条件：

\[
\boxed{
\Gamma_{t,\min}
\le
\Gamma_t
\le
\Gamma_{t,\max}
}
\]

未来升级为 map-based 模型时，可替换为：

\[
\Gamma_t
=
\mathcal M_{W,t}
\left(
N/\sqrt{T_{t4}},
P_{t4}/P_{t5},
geometry
\right)
\]

并引入额外 residual。

## 8.5 涡轮效率 closure

\[
\delta_{\phi t}
=
\frac{\phi_t-\phi_{t,d}}{\phi_{t,d}}
\]

\[
\delta_{\psi t}
=
\frac{\psi_t-\psi_{t,d}}{\psi_{t,d}}
\]

\[
\delta_{\sigma t}
=
\frac{\sigma_t-\sigma_{t,d}}{\sigma_{t,d}}
\]

\[
\delta_{Mt}
=
\max
\left[
0,
\frac{Ma_{U,t}-Ma_{U,t,\lim}}
{Ma_{U,t,\lim}}
\right]
\]

\[
\delta_{Ret}
=
\max
\left[
0,
\frac{Re_{t,d}}{Re_t}-1
\right]
\]

\[
\boxed{
\eta_t
=
\eta_{t,d}
-
a_{\phi t}\delta_{\phi t}^2
-
a_{\psi t}\delta_{\psi t}^2
-
a_{\sigma t}\delta_{\sigma t}^2
-
a_{Mt}\delta_{Mt}^2
-
a_{Ret}\delta_{Ret}^2
}
\]

## 8.6 涡轮出口压力

\[
T_{t5s}
=
T_{t4}
-
\frac{T_{t4}-T_{t5}}
{\eta_t}
\]

\[
\boxed{
P_{t5}
=
P_{t4}
\left(
\frac{T_{t5s}}{T_{t4}}
\right)^{
\gamma_h/(\gamma_h-1)
}
}
\]

---

# 9. 喷管模型 V2.1

V2.0 中 choked 状态同时使用：

\[
M_e=1
\]

和：

\[
V_e=
\sqrt{
2\eta_nc_p(T_t-T_{e,s})
}
\]

在 \(\eta_n<1\) 时会导致实际速度小于声速，状态不一致。

V2.1 改为：

\[
\boxed{
\text{喷管不可逆损失用总压恢复 }\sigma_n
}
\]

不再使用单独的速度效率 \(\eta_n\)。

## 9.1 喷管入口总状态

\[
\boxed{
P_{tn}
=
\sigma_nP_{t5}
}
\]

\[
\boxed{
T_{tn}=T_{t5}
}
\]

其中：

\[
0<\sigma_n\le1
\]

若同时保留 \(C_{d,n}\)，只允许解释为有效流通面积/流量系数，不得和 \(\sigma_n\) 重复表示总压损失。

## 9.2 质量流量函数

\[
\boxed{
\mathcal Q(M,\gamma)
=
\sqrt{\frac{\gamma}{R_g}}
M
\left(
1+\frac{\gamma-1}{2}M^2
\right)^{
-\frac{\gamma+1}{2(\gamma-1)}
}
}
\]

\[
\boxed{
\dot m_n
=
C_{d,n}A_*
\frac{P_{tn}}{\sqrt{T_{tn}}}
\mathcal Q(M_e,\gamma_h)
}
\]

## 9.3 堵塞

临界压力：

\[
P_{\rm crit}
=
P_{tn}
\left(
\frac{2}{\gamma_h+1}
\right)^{
\gamma_h/(\gamma_h-1)
}
\]

若：

\[
P_0\le P_{\rm crit}
\]

则：

\[
\boxed{
M_e=1
}
\]

\[
\boxed{
T_e=
T_{tn}\frac{2}{\gamma_h+1}
}
\]

\[
\boxed{
P_e=P_{\rm crit}
}
\]

\[
\boxed{
V_e=
\sqrt{\gamma_hR_gT_e}
}
\]

\[
\boxed{
\dot m_n
=
C_{d,n}A_*
\frac{P_{tn}}{\sqrt{T_{tn}}}
\mathcal Q(1,\gamma_h)
}
\]

这时：

\[
M_e,\ T_e,\ V_e
\]

完全自洽。

## 9.4 未堵塞

若：

\[
P_0>P_{\rm crit}
\]

则：

\[
P_e=P_0
\]

\[
\boxed{
M_e
=
\sqrt{
\frac{2}{\gamma_h-1}
\left[
\left(
\frac{P_{tn}}{P_0}
\right)^{
(\gamma_h-1)/\gamma_h
}
-1
\right]
}
}
\]

\[
\boxed{
T_e
=
\frac{
T_{tn}
}{
1+\frac{\gamma_h-1}{2}M_e^2
}
}
\]

\[
\boxed{
V_e
=
M_e\sqrt{\gamma_hR_gT_e}
}
\]

## 9.5 完整推力

\[
\boxed{
F
=
\dot m_gV_e
-
\dot m_aV_0
+
(P_e-P_0)A_e
}
\]

纯收敛喷管：

\[
A_e=A_*
\]

---

# 10. 整机工作点非线性求解

## 10.1 主未知量

在给定 \(x_{19}\) 后，V2.1 主工作点未知量为：

\[
\boxed{\dot m_a}
\]

因为：

- \(\pi_c,\eta_c\) 由压气机模块求；
- \(f,\dot m_f\) 由燃烧能量方程求；
- \(T_{t5}\) 由轴功平衡求；
- \(\eta_t,P_{t5}\) 由涡轮 closure 求；
- 喷管给出可通过流量。

## 10.2 流量连续残差

\[
\dot m_g=(1+f)\dot m_a
\]

喷管容量：

\[
\dot m_n
=
\mathcal N(P_{tn},T_{tn},A_*)
\]

必须：

\[
\boxed{
R_{\rm flow}
=
\dot m_g-\dot m_n
=
0
}
\]

无量纲残差：

\[
\boxed{
r_{\rm flow}
=
\frac{
\dot m_g-\dot m_n
}{
\max(\dot m_g,\epsilon_{\dot m})
}
}
\]

---

# 11. Root Solver V2.1

## 11.1 搜索区间

利用压气机进口面积构造保守 choked capacity：

\[
\dot m_{\rm in,max}
=
C_{d,i}
A_{c1,\rm eff}
\frac{P_{t2}}{\sqrt{T_{t2}}}
\sqrt{\frac{\gamma_c}{R_c}}
\left(
\frac{2}{\gamma_c+1}
\right)^{
\frac{\gamma_c+1}
{2(\gamma_c-1)}
}
\]

构造：

\[
[\dot m_{\min},\dot m_{\max}]
\]

## 11.2 有效域分割

扫描同时使用：

- 线性点；
- 对数点。

每个 trial 返回：

```text
VALID_STATE + r_flow
```

或：

```text
HARD_FAILURE
```

**只能在两个相邻、且均为 VALID_STATE 的点之间建立 bracket。**

禁止：

```text
valid(+)
hard failure
valid(-)
```

被错误视为根区间。

## 11.3 符号变化根

若：

\[
r_i r_{i+1}<0
\]

使用 Brent：

\[
\boxed{
\dot m_a^*
=
Brent(r_{\rm flow},\dot m_L,\dot m_R)
}
\]

## 11.4 切触根和窄根

对连续有效域上的：

\[
|r_{\rm flow}|
\]

局部极小值做补充检查。

若：

\[
|r_{\rm flow}|
<
r_{\rm refine}
\]

则局部自适应加密。

用于捕捉：

- 很窄的根区间；
- 偶重根/切触零点；
- 固定扫描点漏检。

## 11.5 多根

最终：

- 0 根：`NO_FLOW_MATCH`
- 1 根：进入完整约束判断
- >1 根：`MULTIPLE_OPERATING_POINTS`

V2.1 不自动选择多根分支。

---

# 12. 性能

求解成功：

\[
F=
\dot m_gV_e-\dot m_aV_0+(P_e-P_0)A_*
\]

\[
\boxed{
TSFC=\frac{\dot m_f}{F}
}
\]

要求：

\[
F>0,\qquad TSFC>0
\]

---

# 13. 质量模型 V2.1

\[
\boxed{
M=
M_c+M_b+M_t+M_n+M_s+
M_{\rm inj}+M_{\rm bear/accessory}
}
\]

质量完全由设计和材料计算，不使用 \(M^*\)。

## 13.1 压气机

定义进口平均半径：

\[
r_{1m}
=
\frac{D_{c1t}+D_{c1h}}{4}
\]

轮毂/背盘：

\[
t_{c,\rm hub}=k_{c,hb}b_{c2}
\]

\[
V_{c,\rm hub}
=
\kappa_{c,h}
\pi
(r_2^2-r_{1m}^2)
t_{c,\rm hub}
\]

平均叶片长度：

\[
\ell_{c,b}
=
k_{c,\ell}
\sqrt{
(r_2-r_{1m})^2+
\left(
\frac{D_{c1t}-D_{c1h}}{2}
\right)^2
}
\]

平均叶高：

\[
h_{c,b}
=
k_{c,h}
\left[
b_{c2}
+
\frac{D_{c1t}-D_{c1h}}{2}
\right]
\]

叶片厚度：

\[
t_{c,b}=k_{c,t}b_{c2}
\]

\[
V_{c,\rm blade}
=
Z_c\ell_{c,b}h_{c,b}t_{c,b}
\]

casing/diffuser：

\[
L_c=k_{Lc}D_{c2}
\]

\[
V_{c,\rm casing}
=
k_{c,\rm casing}
\pi D_{c2}L_ct_{c,\rm case}
\]

因此：

\[
\boxed{
M_c
=
\rho_{\rm mat,c}
(
V_{c,\rm hub}
+
V_{c,\rm blade}
+
V_{c,\rm casing}
)
}
\]

这样 \(D_{c2},D_{c1t},\lambda_{c1},b_{c2},Z_c\) 都进入质量。

## 13.2 燃烧室

\[
V_{b,\rm cyl}
=
\pi(D_{bo}+D_{bi})L_bt_b
\]

\[
V_{b,\rm shell}
=
k_{bd}V_{b,\rm cyl}
\]

若点阵体积分数：

\[
\chi_{b,\rm lat}
\]

则：

\[
\rho_{b,\rm eff}
=
\rho_{\rm mat,b}
[
(1-\chi_{b,\rm lat})
+
\chi_{b,\rm lat}\bar\rho_{\rm lat}
]
\]

\[
M_b
=
\rho_{b,\rm eff}V_{b,\rm shell}
\]

\[
M_{\rm inj}
=
Z_{\rm inj}m_{\rm inj,unit}
\]

## 13.3 涡轮

V2.1 增加轮盘主体。

\[
t_{\rm disk}
=
k_{t,disk}D_t
\]

\[
\boxed{
V_{t,\rm disk}
=
\kappa_{\rm disk}
\pi R_h^2t_{\rm disk}
}
\]

叶片：

\[
t_{t,b}=k_{t,b}c_t
\]

\[
V_{t,\rm blade}
=
Z_tc_th_tt_{t,b}
\]

NGV：

\[
V_{t,\rm NGV}
=
k_{t,\rm NGV}V_{t,\rm blade}
\]

casing：

\[
L_t=k_{Lt}h_t
\]

\[
V_{t,\rm casing}
=
k_{t,\rm case}
\pi D_tL_tt_{t,\rm case}
\]

最终：

\[
\boxed{
M_t
=
\rho_{\rm mat,t}
(
V_{t,\rm disk}
+
V_{t,\rm blade}
+
V_{t,\rm NGV}
+
V_{t,\rm casing}
)
}
\]

因此当：

\[
\lambda_t\rightarrow1
\]

时，涡轮质量不会因 annulus area 趋零而错误趋零。

## 13.4 喷管

冻结：

\[
D_{n,in}=k_{Din}D_*
\]

\[
L_n=k_{Ln}D_*
\]

\[
r_1=\frac{D_{n,in}}2,\qquad
r_2=\frac{D_*}2
\]

\[
s_n=
\sqrt{L_n^2+(r_1-r_2)^2}
\]

\[
A_{n,\rm shell}
=
\pi(r_1+r_2)s_n
\]

\[
t_n=k_{tn}D_*
\]

\[
V_{n,\rm shell}
=
A_{n,\rm shell}t_n
\]

冻结点阵比例：

\[
\chi_{n,\rm lat}
\]

\[
\rho_{n,\rm eff}
=
\rho_{\rm mat,n}
[
(1-\chi_{n,\rm lat})
+
\chi_{n,\rm lat}\bar\rho_{\rm lat}
]
\]

\[
M_n=
\rho_{n,\rm eff}V_{n,\rm shell}
\]

## 13.5 轴和附件

\[
r_s=k_{rs}R_h
\]

\[
L_s=k_{Ls}(L_c+L_b+L_t)
\]

\[
M_s=
\rho_{\rm mat,s}\pi r_s^2L_s
\]

\[
M_{\rm bear/accessory}
=
M_{\rm fixed}
+
k_{\rm ba}(M_c+M_t+M_s)
\]

---

# 14. 点阵有效性质

点阵质量：

\[
\rho^*
=
\rho_{\rm mat,solid}\bar\rho_{\rm lat}
\]

但：

\[
a_{\rm lat}
\]

不人为进入质量。

低阶结构 scaling：

\[
\frac{E^*}{E_s}
=
C_E\bar\rho_{\rm lat}^{n_E}
\]

\[
\frac{\sigma_y^*}{\sigma_{y,s}}
=
C_\sigma\bar\rho_{\rm lat}^{n_\sigma}
\]

燃烧室只有比例 \(\chi_{b,\rm lat}\) 的 eligible shell 被点阵替代，
因此 WP19-8 的 liner 低阶强度 proxy 使用混合闭合：

\[
\boxed{
\sigma_{y,b,\rm eff}
=
\sigma_{y,b}
\left[
(1-\chi_{b,\rm lat})
+
\chi_{b,\rm lat}
C_\sigma\bar\rho_{\rm lat}^{n_\sigma}
\right]
}
\]

这只是均匀化筛选关系，不替代实体 skin、局部连接和屈曲分析。

calibration pack 必须冻结：

- 点阵拓扑；
- 哪些部件点阵化；
- 燃烧室点阵体积分数；
- 喷管点阵体积分数；
- 是否保留实体 skin；
- flange/joint 是否允许点阵化；
- 最小实体蒙皮厚度。

---

# 15. 正式可行性约束

统一：

\[
\boxed{g_j\le0}
\]

## 15.1 压气机

\[
0<\lambda_{c1}<1
\]

\[
D_{c1h}<D_{c1t}<D_{c2}
\]

\[
0<B_{c2}<1
\]

\[
\phi_{c,\min}
\le
\phi_{c,\rm ref}
\le
\phi_{c,\max}
\]

\[
\psi_{c,\min}
\le
\psi_c
\le
\psi_{c,\max}
\]

\[
Ma_{U,c}\le Ma_{U,c,\max}
\]

\[
\eta_c\ge\eta_{c,\min}
\]

在没有 compressor map 前，这些称 calibration-domain constraints，不称真实 surge/choke margin。

## 15.2 燃烧室

\[
D_{bo}>D_{bi}+2t_b
\]

\[
\delta_b\le\delta_{b,\max}
\]

\[
\tau_{b,p}\ge\tau_{b,p,\min}
\]

\[
\varphi_{\min}\le\varphi\le\varphi_{\max}
\]

\[
\dot m_{f,\rm inj}
\le
\dot m_{f,\rm inj,max}
\]

## 15.3 涡轮

\[
\phi_{t,\min}
\le\phi_t\le
\phi_{t,\max}
\]

\[
\psi_t\le\psi_{t,\max}
\]

\[
\sigma_{t,\min}
\le\sigma_t\le
\sigma_{t,\max}
\]

\[
Ma_{U,t}\le Ma_{U,t,\max}
\]

\[
\eta_t\ge\eta_{t,\min}
\]

以及正式吞流能力：

\[
\boxed{
\Gamma_{t,\min}
\le
\Gamma_t
\le
\Gamma_{t,\max}
}
\]

这不是仅报告的诊断量，而是正式 feasibility condition。

---

# 16. 组件装配一致性

## 16.1 压气机—燃烧室

定义扩压器出口到燃烧室入口的目标面积比：

\[
\Pi_{A,db}
\]

\[
\boxed{
g_{\rm d-b,area}
=
\left|
\frac{A_{d,out}}{A_{b,in}}
-
\Pi_{A,db}
\right|
-
\epsilon_{A,db}
}
\]

## 16.2 燃烧室—涡轮

\[
\boxed{
g_{\rm b-t,outer}
=
\frac{|D_{bo}-D_t|}
{D_t}
-
\epsilon_{D,bt}
}
\]

## 16.3 轴间隙

\[
D_s=2r_s
\]

\[
\boxed{
g_{\rm shaft-clearance}
=
D_s+2\delta_s-D_{bi}
}
\]

## 16.4 涡轮—喷管

\[
D_{n,in}=k_{Din}D_*
\]

\[
\boxed{
g_{\rm t-n}
=
\frac{|D_{n,in}-D_t|}
{D_t}
-
\epsilon_{D,tn}
}
\]

这些约束防止“部件单独可算、整机无法装配”。

---

# 17. 结构、热和制造约束

## 17.1 转子应力代理

\[
\sigma_{\rm rot,c}
=
k_{\sigma c}\rho_{\rm mat,c}U_{c2}^2
\]

\[
\sigma_{\rm rot,t}
=
k_{\sigma t}\rho_{\rm mat,t}U_{t,\rm tip}^2
\]

安全约束：

\[
SF_c
\frac{\sigma_{\rm rot,c}}
{\sigma_{y,c}(T_c)}
\le1
\]

\[
SF_t
\frac{\sigma_{\rm rot,t}}
{\sigma_{y,t}(T_{m,t})}
\le1
\]

WP19-8 bootstrap 实现中还没有经验证的高温屈服曲线，因此暂时冻结：

\[
T_c=T_{t3},\qquad
\sigma_{y,c}(T_c)\approx\sigma_{y,c,\rm room},\qquad
\sigma_{y,t}(T_{m,t})\approx\sigma_{y,t,\rm room}.
\]

该近似只用于低阶筛选，并必须与 17.3 的材料最高温度约束同时使用。
在 calibration freeze 前，应以试验或材料手册中的高温许用曲线替换，且不得
将当前 proxy 描述为已验证的温度相关材料模型。

仅作为低阶筛选，不等于 FEA。

## 17.2 燃烧室 liner

\[
\Delta P_{\rm liner}
=
k_{P,b}\Delta P_{tb}
\]

\[
\sigma_{\theta,b}
=
\frac{
\Delta P_{\rm liner}D_{bm}
}{
2t_b
}
\]

要求：

\[
SF_b
\frac{\sigma_{\theta,b}}
{\sigma_{y,b}(T_{m,b})}
\le1
\]

## 17.3 金属温度

涡轮：

\[
T_{m,t}
=
T_{t3}
+
\chi_{T,t}(T_{t4}-T_{t3})
\]

\[
\boxed{
g_{T,t}
=
T_{m,t}-T_{m,t,\max}
}
\]

燃烧室：

\[
T_{m,b}
=
T_{t3}
+
\chi_{T,b}(T_{t4}-T_{t3})
\]

V2.1 补充：

\[
\boxed{
g_{T,b}
=
T_{m,b}-T_{m,b,\max}
}
\]

## 17.4 TIT

\[
\boxed{
g_{TIT,\min}
=
T_{t4,\min}-T_{t4}
}
\]

\[
\boxed{
g_{TIT,\max}
=
T_{t4}-T_{t4,\max}
}
\]

## 17.5 点阵制造

\[
\bar\rho_{\min}
\le
\bar\rho_{\rm lat}
\le
\bar\rho_{\max}
\]

\[
a_{\rm lat}\ge a_{\rm lat,min}
\]

定义：

\[
N_{\rm cell}
=
\frac{L_{\rm lat,min}}
{a_{\rm lat}}
\]

对当前仅在燃烧室和喷管使用点阵的冻结区域，派生：

\[
\boxed{
L_{\rm lat,min}
=
\min(D_{bo}-D_{bi},L_n)
}
\]

必须：

\[
N_{\rm cell}\ge N_{\rm cell,min}
\]

若 `lattice_skin_retained=true`，还必须满足：

\[
t_b\ge t_{\rm skin,min},\qquad
t_n\ge t_{\rm skin,min}.
\]

---

# 18. 流量和喷管约束

\[
g_{\rm flow}
=
|r_{\rm flow}|
-
\epsilon_{\rm flow}
\]

喷管必须具有可用总压：

\[
\boxed{
P_{tn}>P_0
}
\]

正推力：

\[
F>0
\]

---

# 19. Hard Failure 与 Soft Infeasible

## 19.1 Hard failure

没有完整可计算物理状态：

```text
GEOMETRY_INVALID
COMPRESSOR_BLOCKAGE_INVALID
COMPRESSOR_SLIP_INVALID
COMPRESSOR_NONPOSITIVE_WORK
COMPRESSOR_EFFICIENCY_INVALID
COMBUSTOR_PRESSURE_INVALID
COMBUSTOR_ENERGY_INVALID
TURBINE_POWER_INSUFFICIENT
TURBINE_EFFICIENCY_INVALID
TURBINE_PRESSURE_INVALID
NOZZLE_STATE_INVALID
NO_FLOW_MATCH
MULTIPLE_OPERATING_POINTS
NONFINITE_STATE
ROOT_SOLVER_FAILED
```

## 19.2 Soft infeasible

工作点数学上收敛，但：

\[
\exists j,\quad g_j>0
\]

则：

```text
converged = true
feasible = false
```

不可通过 clipping 把它修复成可行样本。

---

# 20. EngineResult

建议统一结构：

```yaml
EngineResult:
  design:
    x19: [...]

  environment:
    T0_K:
    P0_Pa:
    Mach:

  operating_point:
    mdot_air_kg_s:
    mdot_fuel_kg_s:
    fuel_air_ratio:
    afr:

  stations:
    t2: {Tt_K, Pt_Pa}
    t3: {Tt_K, Pt_Pa}
    t4: {Tt_K, Pt_Pa}
    t5: {Tt_K, Pt_Pa}
    nozzle_total: {Ttn_K, Ptn_Pa}
    exit: {T_K, P_Pa, Mach, V_m_s}

  compressor:
    pressure_ratio:
    total_to_total_efficiency:
    phi_ref:
    psi:
    slip_factor:
    shaft_power_W:

  combustor:
    pressure_loss_fraction:
    primary_residence_time_s:
    equivalence_ratio:

  turbine:
    total_to_total_efficiency:
    phi:
    psi:
    solidity:
    swallowing_parameter:

  nozzle:
    choked:
    mdot_capacity_kg_s:
    total_pressure_recovery:

  performance:
    thrust_N:
    tsfc_kg_N_s:
    mass_kg:

  mass_breakdown_kg:
    compressor:
    combustor:
    turbine:
    nozzle:
    shaft:
    injectors:
    bearing_accessory:

  constraints:
    values: {...}
    thermodynamic: {...}
    assembly: {...}
    structural: {...}
    thermal: {...}
    manufacturing: {...}
    feasible:

  formal_diagnostics:
    yield_strength_temperature_model:

  qualification:
    feasibility_scope:
    feasibility_complete:
    calibration_status:
    formal_dataset_generation_ready:

  solver:
    converged:
    flow_residual:
    iterations:
    failure_code:

  provenance:
    physics_model_version:
    calibration_pack_hash:
```

---

# 21. 数据生成

\[
x_{19}^{(i)}
\sim
DOE
\]

然后：

\[
\boxed{
x_{19}^{(i)}
\rightarrow
Geometry
\rightarrow
Component\ Models
\rightarrow
Engine\ Match
\rightarrow
[F_i,TSFC_i,M_i,\mathbf g_i,status_i]
}
\]

所有样本都保存：

```text
sample_id
x19
converged
feasible
failure_code
failure_class
flow_residual
physics_model_version
calibration_pack_hash
```

只有：

```text
converged == true
```

的样本具有完整：

\[
F,TSFC,M,\mathbf g
\]

---

# 22. 代理模型必须分层

Hard-failure 样本没有完整 \(F,TSFC,M\)，因此不能和收敛样本一起训练普通性能回归。

## 22.1 分类层

\[
\boxed{
x_{19}
\rightarrow
p(
converged,
failure\ class,
feasible
)
}
\]

## 22.2 性能层

仅收敛样本：

\[
\boxed{
x_{19}\mid converged
\rightarrow
[F,TSFC,M,\mathbf g]
}
\]

## 22.3 逆设计筛选

\[
[F^*,TSFC^*,M^*]
\rightarrow
G
\rightarrow
x_{19}
\]

先：

\[
S_{\rm class}
\]

再可选：

\[
S_{\rm perf}
\]

最后必须：

\[
\boxed{
\mathcal P(x_{19})
}
\]

真实回代。

---

# 23. Calibration Pack

\[
\theta=
[
\theta_c,\theta_b,\theta_t,\theta_n,
\theta_m,\theta_s,\theta_{\rm assembly}
]
\]

所有参数必须记录：

```text
name
value
unit
valid_range: [valid_min, valid_max]
source_type
source_reference
physical_meaning
calibration_target
notes (optional)
```

其中 `valid_range` 是闭区间，数值参数必须满足

\[
\texttt{valid\_min}\leq \texttt{value}\leq \texttt{valid\_max}.
\]

`notes` 仅在需要补充额外限定或风险说明时提供；省略表示
`source_reference`、`physical_meaning` 与 `calibration_target` 已完整表达
当前参数资格。

`source_type`：

```text
TEXTBOOK_RELATION
NASA_NACA_CORRELATION
CURRENT_NOVATURBO_BASELINE
EXPERIMENT
HIGH_FIDELITY
PUBLISHED_LITERATURE
MATERIAL_STANDARD
PROJECT_BOOTSTRAP_ASSUMPTION
```

## 23.1 压气机必须冻结

- \(\eta_{c,d}\)
- \(\phi_{c,d}\)
- \(\psi_{c,d}\)
- Mach/Re 参考值和限制
- closure 系数
- blockage/thickness 系数
- 适用几何和转速域
- “整台压气机总—总效率”定义

## 23.2 燃烧室必须冻结

- \(K_{b0},K_{bL},K_{bA}\)
- \(\chi_{V,p}\)
- \(\chi_{\dot m,p}\)
- \(\tau_{b,p,\min}\)
- 当量比范围
- 喷油嘴参数
- 共同焓参考状态

## 23.3 涡轮必须冻结

- efficiency closure
- \(\phi,\psi,\sigma\) bounds
- \(\Gamma_{t,\min},\Gamma_{t,\max}\)
- aspect ratio
- Mach/Re limits

## 23.4 喷管必须冻结

- \(\sigma_n\)
- \(C_{d,n}\)

并明确：

\[
\sigma_n
\]

表示总压损失，

\[
C_{d,n}
\]

只表示有效流通面积/流量偏差，防止 double counting。

## 23.5 质量、结构和装配必须冻结

- 压气机轮毂/叶片/casing 系数
- 涡轮 disk/blade/NGV/casing 系数
- 材料密度与温度许用
- 点阵拓扑/区域/比例
- 装配目标比和容差
- 安全系数

生成：

```text
geometry19_physics_v2_1_calibration.yaml
geometry19_physics_v2_1_calibration_sources.md
```

以及：

```text
calibration_pack_hash
```

只要存在：

```text
null
TBD
...
```

或无来源数值，则：

```text
formal_dataset_generation_ready = false
```

---

# 24. 推荐代码结构

```text
src/physics/
├── geometry_state.py
├── inlet.py
├── compressor_meanline.py
├── combustor_loworder.py
├── turbine_meanline.py
├── nozzle.py
├── engine_match.py
├── mass_model_v2.py
├── constraints_v2.py
├── materials_v2.py
└── model_config.py
```

正式入口：

```python
result = solve_design_point(
    x19,
    environment,
    model_config,
)
```

旧 `brayton.py` 保留为 active-10D regression baseline。

---

# 25. 完整实施流程

## WP19-0 基线冻结

- 冻结当前 active-10D evaluator；
- 冻结当前逆设计正式链；
- 保存 regression cases。

## WP19-1 Schema

冻结新 19D、单位、bounds、integer rules。

## WP19-2 GeometryState

实现所有解析几何和装配派生量。

## WP19-3 Compressor

实现：

\[
geometry+N+\dot m_a
\rightarrow
\pi_c,\eta_c,P_c
\]

## WP19-4 Combustor

实现：

\[
T_{t3},P_{t3},T_{t4},geometry
\rightarrow
P_{t4},f,\tau_{b,p}
\]

## WP19-5 Turbine/Shaft

实现：

\[
P_c,\dot m_g,T_{t4},geometry,N
\rightarrow
T_{t5},P_{t5},\eta_t,\Gamma_t
\]

## WP19-6 Nozzle

实现 V2.1 一致状态：

\[
P_{tn},T_{tn},M_e,T_e,P_e,V_e,\dot m_n
\]

## WP19-7 EngineMatch

实现：

- mixed linear/log scan；
- valid-domain segmentation；
- Brent；
- tangency check；
- multiple-root detection。

## WP19-8 Mass/Assembly/Structure

实现 V2.1 质量和所有正式约束。

## WP19-9 单点验证

逐站检查：

\[
T_t,P_t,\dot m,\pi,\eta,\phi,\psi
\]

以及：

- 燃烧能量残差；
- 轴功残差；
- 流量残差；
- nozzle state consistency。

## WP19-10 小规模 DOE

先：

\[
10^3\sim10^4
\]

级样本。

检查：

- root success；
- failure distribution；
- multiple roots；
- 每维 sensitivity；
- dead-dimension；
- assembly feasibility；
- performance distribution。

## WP19-11 Calibration Freeze

文献筛选修正必须首先区分系数定义、最优效率区和硬可行域。不得把不同
归一化方式的流量系数，或把爆破速度裕度与屈服应力安全系数直接互换。
当前决策记录见：

```text
docs/physics/WP19_11_LITERATURE_SCREENING_DECISIONS_V1.md
```

填满并冻结所有数值和来源。

## WP19-12 正式大规模 DOE

只有 WP19-9、10、11 全部通过后启动。

## WP19-13 Surrogate / Inverse Generator

最后才训练新的 19D 网络。

---

# 26. 必做 Gate

## G19-A 喷管一致性

堵塞时必须数值验证：

\[
M_e=1
\]

\[
V_e=\sqrt{\gamma R T_e}
\]

不能再出现理想 Mach 和实际速度不一致。

## G19-B 燃烧能量

\[
\left|
\dot m_ah_a
+
\dot m_f(\eta_bLHV+h_f)
-
\dot m_gh_g
\right|
<\epsilon_E
\]

## G19-C 轴功

\[
|R_{\rm shaft}|<\epsilon_P
\]

## G19-D 流量

\[
|r_{\rm flow}|<\epsilon_{\rm flow}
\]

## G19-E 涡轮容量

所有最终 feasible 样本必须满足：

\[
\Gamma_{t,\min}
\le
\Gamma_t
\le
\Gamma_{t,\max}
\]

## G19-F 装配

所有 assembly residual：

\[
g_{\rm assembly}\le0
\]

## G19-G 数据标签

Hard failure 不得被送入普通性能 regression loss。

## G19-H Calibration

正式 DOE 前必须：

```text
all calibration fields non-null
all sources recorded
all valid ranges recorded
calibration hash frozen
```

---

# 27. 标准方程与项目 closure 的边界

| 模型关系 | 类型 |
|---|---|
| 总温/总压与 Mach | 标准气体动力学 |
| Euler turbomachinery equation | 标准教材 |
| 压气机/涡轮等熵效率 | 标准教材 |
| 燃烧能量守恒 | 标准教材 |
| 轴功平衡 | 标准整机匹配 |
| 喷管 choked mass flow | 标准可压缩流 |
| 完整推力 | 标准推进方程 |
| Stanitz slip | 经典工程 correlation |
| \(\eta_c(\phi,\psi,Ma,Re)\) | NovaTurbo 可标定 closure |
| \(K_b\) 几何压损式 | NovaTurbo 可标定 closure |
| \(\eta_t(\phi,\psi,\sigma,Ma,Re)\) | NovaTurbo 可标定 closure |
| \(\Gamma_t\) 工作域边界 | NovaTurbo V2.1 容量筛选 |
| 组件质量体积 closure | NovaTurbo 可标定模型 |
| 金属温度 proxy | NovaTurbo 筛选近似 |
| Gibson–Ashby scaling | 教材型 cellular-solid scaling，系数需按 topology 冻结 |

---

# 28. 教科书与参考来源

## 28.1 Gas Turbine Theory

H. Cohen, G. F. C. Rogers, H. I. H. Saravanamuttoo, P. Straznicky, A. Nix, **Gas Turbine Theory**, 7th ed., Pearson.

用于：

- Brayton / turbojet cycle；
- 压气机和涡轮效率；
- 燃烧器；
- 轴功匹配；
- design/off-design 基本框架。

NovaTurbo 简化：

- 常比热；
- 单轴；
- 无 cooling bleed；
- 无真实 compressor/turbine map；
- 只做稳态 design-point。

## 28.2 Mattingly

J. D. Mattingly, **Elements of Gas Turbine Propulsion**, McGraw-Hill.

用于：

- station bookkeeping；
- turbojet design-point analysis；
- thrust；
- TSFC；
- 部件效率与整机性能关系。

核心：

\[
F=
\dot m_eV_e-\dot m_0V_0+(P_e-P_0)A_e
\]

## 28.3 Dixon & Hall

S. L. Dixon, C. Hall, **Fluid Mechanics and Thermodynamics of Turbomachinery**, Elsevier.

用于：

- Euler equation；
- velocity triangle；
- blade speed；
- flow coefficient；
- loading coefficient；
- turbomachinery dimensionless design language。

\[
\Delta h_0
=
U_2C_{\theta2}-U_1C_{\theta1}
\]

\[
\phi=\frac{C_m}{U}
\]

\[
\psi=\frac{\Delta h_0}{U^2}
\]

## 28.4 Aungier

R. H. Aungier, **Centrifugal Compressors: A Strategy for Aerodynamic Design and Analysis**, ASME Press.

作为未来 compressor closure 从 bootstrap response model 升级到更完整 mean-line/loss model 的参考。

## 28.5 Lefebvre & Ballal

A. H. Lefebvre, D. R. Ballal, **Gas Turbine Combustion: Alternative Fuels and Emissions**, CRC Press.

用于：

- combustor pressure loss；
- reference velocity；
- fuel-air ratio；
- combustor loading；
- primary zone；
- injector；
- combustion stability/residence。

## 28.6 Gibson & Ashby

L. J. Gibson, M. F. Ashby, **Cellular Solids: Structure and Properties**, Cambridge University Press.

用于：

\[
E^*/E_s=C_E\bar\rho^n
\]

\[
\sigma^*/\sigma_s
=
C_\sigma\bar\rho^m
\]

---

# 29. NASA/NACA 交叉验证方向

实现测试时建议继续使用 NASA/NACA 公开关系作为 equation-level sanity check：

- compressor thermodynamics；
- burner fuel-flow / energy balance；
- compressor–turbine matching；
- choked mass flow；
- general thrust equation；
- Stanitz centrifugal-compressor slip/mean-line work；
- turbine flow/loading coefficient studies。

这些来源用于核对标准方程，不用于证明 NovaTurbo 自定义效率 closure 是“通用真值”。

---

# 30. 最终模型定义

V2.1 的前向链正式定义为：

\[
\boxed{
\begin{aligned}
x_{19}
&\rightarrow
GeometryState\\
&\rightarrow
Compressor(\dot m_a)\\
&\rightarrow
Combustor\\
&\rightarrow
Shaft/Turbine\\
&\rightarrow
Nozzle\\
&\rightarrow
R_{\rm flow}(\dot m_a)=0\\
&\rightarrow
Mass\\
&\rightarrow
Assembly/Structure/Thermal/Manufacturing\\
&\rightarrow
[F,TSFC,M,\mathbf g,status]
\end{aligned}
}
\]

逆向：

\[
\boxed{
[F^*,TSFC^*,M^*]
\rightarrow
G
\rightarrow
x_{19}
\rightarrow
\mathcal P
\rightarrow
\text{accept/reject}
}
\]

---

# 31. V2.1 当前发布评级

| 项目 | V2.1 状态 |
|---|---|
| 19D 接口 | 通过 |
| 三个 inverse conditions | 通过 |
| 压气机数学闭合 | 条件通过 |
| 燃烧室能量/primary-zone | 条件通过 |
| 轴功平衡 | 通过 |
| 涡轮低阶工作域 | 条件通过 |
| 喷管状态一致性 | **已修正** |
| 单变量整机流量匹配 | 条件通过 |
| Root bracket 鲁棒性 | **已增强定义** |
| 质量模型 | **已补充 disk/blade/assembly 依赖** |
| 组件装配 | **已补充** |
| TIT / combustor metal temperature | **已补充** |
| 分类 + 回归数据层 | **已补充** |
| calibration pack schema | 通过 |
| calibration pack 数值冻结 | **未完成** |
| 正式大规模数据生成 | **未开放** |

---

# 32. 最终结论

V2.1 已解决 V2.0 最明确的喷管状态矛盾，并把其余评审意见纳入完整模型规范。

现在系统可以准确称为：

\[
\boxed{
\text{Geometry-aware, closed, low-order design-point turbojet model}
}
\]

但仍不能称为：

- 完整 compressor/turbine map matching model；
- CFD/试验真值；
- 精确结构/热分析器。

其当前最重要的最后一道门槛不是再增加方程，而是：

\[
\boxed{
\text{冻结 calibration pack}
+
\text{单点守恒验证}
+
\text{小规模 DOE 验证}
}
\]

只有这三项通过，NovaTurbo 才应启动正式 19D 大规模参数采样、surrogate 训练和 inverse-generator 训练。
