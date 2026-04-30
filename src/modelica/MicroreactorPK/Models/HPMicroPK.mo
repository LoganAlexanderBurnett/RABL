within MicroreactorPK.Models;
model HPMicroPK
  // Point-kinetics + 4-node reactor thermal network + active-metal steam-generator node
  // + 12 drums (same angle)
  //
  // State summary:
  //   n(t)      : normalized reactor power (n=1 -> P = P_r)
  //   c[i](t)   : delayed neutron precursor concentrations (6 groups)
  //   Tf(t)     : fuel temperature
  //   Tm(t)     : moderator / graphite / structure temperature
  //   Thp(t)    : heat pipe effective temperature node
  //   TN2(t)    : nitrogen loop average temperature node
  //   Tsg(t)    : active steam-generator heat-transfer metal temperature node
  //
  // Inputs:
  //   drumAngleDeg(t) applied to all drums equally
  //
  // Outputs:
  //   rho(t)             : reactivity in Δk/k
  //   rho_dollars(t)     : reactivity in $
  //   P(t)               : thermal power in W
  //   P_MW(t)            : thermal power in MW
  //   T_steam_out(t)     : calculated outlet temperature at fixed outlet pressure
  //   x_steam_out(t)     : calculated outlet steam quality / dryness fraction
  import SI = Modelica.Units.SI;

  // ---------- KINETIC PARAMETERS ----------

  // Prompt neutron generation time [s].
  // Controls how fast power responds to reactivity changes in point kinetics.
  parameter Real Lambda = 1.95734e-4 "Prompt neutron generation time [s]";

  // Number of delayed neutron precursor groups (classic 6-group approximation).
  parameter Integer nGroups = 6;

  // Delayed group decay constants λ_i [1/s].
  // Each group i has characteristic decay time 1/λ_i.
  parameter Real lambdas[nGroups] = {
    0.01334,
    0.03274,
    0.1208,
    0.3028,
    0.8495,
    2.853}
    "Delayed group decay constants [1/s]";

  // Delayed neutron fractions β_i (dimensionless).
  // Fraction of neutrons born delayed in each group.
  parameter Real betas[nGroups] = {
    2.90591e-4,
    1.12869e-3,
    1.16402e-3,
    2.69454e-3,
    9.20043e-4,
    4.41185e-4}
    "Delayed group fractions";

  // Total delayed fraction β = Σβ_i.
  parameter Real beta = sum(betas) "Total delayed neutron fraction";

  // Steady-state precursor concentrations at n=1 and ρ=0:
  //   0 = (β_i/Λ)*n - λ_i*C_i  =>  C_i = β_i/(Λ*λ_i)
  parameter Real C_ss[nGroups] = { betas[i] / (Lambda * lambdas[i]) for i in 1:nGroups };


  // ---------- THERMAL POWER SCALE ----------

  // Rated thermal power [W]. This is the scaling for normalized power n.
  parameter SI.Power P_r = 6.0e6 "Rated thermal power [W]";

  // Fraction of thermal power deposited directly in the fuel node.
  // The remainder (1-heat_f) goes into the moderator/structure node in this lumped model.
  parameter Real heat_f = 0.90 "Fraction of power deposited in fuel";


  // ---------- THERMAL DESIGN TEMPERATURES ----------
  // These define the steady-state point around which:
  // - reactivity feedback is computed (Tf - Tf0, Tm - Tm0)
  // - UA values (G_*) are computed using design temperature differences

  parameter SI.Temperature Tf0   = 1173.15 "Fuel nominal temperature [K]";
  parameter SI.Temperature Tm0   = 1150.15 "Moderator/graphite nominal temperature [K]";
  parameter SI.Temperature Thp0  = 1073.15 "Heat pipe nominal temperature [K]";

  // N2 inlet/outlet design temperatures. TN0 is the average used as the N2 node reference.
  parameter SI.Temperature TN_in = 683.15   "N2 inlet design temperature [K]";
  parameter SI.Temperature TN_out= 1073.15  "N2 outlet design temperature [K]";
  parameter SI.Temperature TN0   = 0.5*(TN_in + TN_out) "N2 average design temperature [K]";


  // ---------- STEAM GENERATOR / SECONDARY SIDE PARAMETERS ----------
  // The new steam-generator node represents active heat-transfer tube metal only.
  // It does not include drum metal, headers, casing, supports, or contained fluid inventory.

  // Fixed secondary-side feedwater mass flow rate.
  parameter SI.MassFlowRate m_dot_fw = 2.95
    "Fixed feedwater mass flow rate [kg/s]";

  // Fixed steam outlet/header pressure.
  // This model uses the pressure only to define the saturation temperature and property constants.
  parameter SI.Pressure p_steam_out = 1.5e6
    "Fixed steam outlet/header pressure [Pa]";

  // Feedwater inlet temperature.
  // For this simplified model, feedwater is treated as approximately saturated liquid
  // at the fixed steam pressure.
  parameter SI.Temperature T_fw_in = 198.3 + 273.15
    "Feedwater inlet temperature [K]";

  // Saturation temperature at p_steam_out.
  // At 1.5 MPa, Tsat is approximately 198.3 C.
  parameter SI.Temperature T_sat_steam = 198.3 + 273.15
    "Saturation temperature at fixed steam pressure [K]";

  // Nominal/design outlet steam temperature.
  parameter SI.Temperature T_steam_out_nom = 232.0 + 273.15
    "Nominal outlet steam temperature [K]";

  // Initial active SG metal temperature.
  // This is an effective active-metal temperature used to initialize the lumped node.
  parameter SI.Temperature Tsg0 = T_steam_out_nom
    "Nominal active steam-generator metal temperature [K]";

  // Latent heat required to fully evaporate saturated liquid to saturated vapor near 1.5 MPa.
  parameter SI.SpecificEnthalpy h_fg = 1946.45e3
    "Latent heat from saturated liquid to saturated vapor near 1.5 MPa [J/kg]";

  // Superheat enthalpy rise from approximately 198 C saturated steam to 232 C steam.
  parameter SI.SpecificEnthalpy h_sh_nom = 89.42e3
    "Nominal superheat enthalpy rise from saturation to 232 C [J/kg]";

  // Total nominal enthalpy rise from feedwater to nominal outlet steam.
  parameter SI.SpecificEnthalpy delta_h_nom = h_fg + h_sh_nom
    "Nominal feedwater-to-steam enthalpy rise [J/kg]";

  // Effective superheated-steam heat capacity over the nominal superheat interval.
  // h_sh_nom = cp_sh_eff * (T_steam_out_nom - T_sat_steam)
  parameter SI.SpecificHeatCapacity cp_sh_eff =
    h_sh_nom / (T_steam_out_nom - T_sat_steam)
    "Effective superheated-steam cp [J/(kg K)]";

  // Active heat-transfer metal thermal capacitance.
  // Baseline from Sulzer/KVK active-tube metal area scaling.
  parameter SI.HeatCapacity C_sg = 2.55e5
    "Active steam-generator heat-transfer metal capacitance [J/K]";


  // ---------- LUMPED MASSES AND HEAT CAPACITIES ----------
  // These define the thermal inertia: M*cp [J/K] of each lump.

  // Fuel lump
  parameter SI.Mass M_f  = 4355.259;
  parameter SI.SpecificHeatCapacity cp_f = 748.72;

  // Moderator/graphite/structure lump
  parameter SI.Mass M_g  = 673.921;
  parameter SI.SpecificHeatCapacity cp_g = 1850.0;

  // Heat pipe lump
  parameter SI.Mass M_hp = 119.678;
  parameter SI.SpecificHeatCapacity cp_hp = 771.3;

  // Nitrogen loop lump:
  // We treat N2 as a well-mixed average node with mass set by:
  //   M_N2 = m_dot_N * tau_N
  // where tau_N is an assumed residence time / transport lag.
  parameter SI.SpecificHeatCapacity cp_N2 = 1182.0;
  parameter SI.Time tau_N = 5.0;

  // Compute the N2 mass flow required to remove P_r given N2 ΔT design:
  //   P_r ≈ m_dot * cp * (TN_out - TN_in)
  parameter SI.MassFlowRate m_dot_N = P_r / (cp_N2*(TN_out - TN_in));

  // Effective N2 mass in the average control volume.
  parameter SI.Mass M_N2 = m_dot_N * tau_N;


  // ---------- CONDUCTION / HEAT-TRANSFER UA's (G = UA) FROM DESIGN TEMPERATURES ----------
  // Reverse-engineering effective thermal conductances between lumps
  // from the design temperature drops at rated power:
  //   Q = G * ΔT  =>  G = Q / ΔT
  // abs() is used so G is positive regardless of temperature ordering.

  parameter SI.ThermalConductance G_f_g  = heat_f * P_r / abs(Tf0  - Tm0);
  parameter SI.ThermalConductance G_g_hp = P_r / abs(Tm0  - Thp0);
  parameter SI.ThermalConductance G_hp_N2= P_r / abs(Thp0 - TN0);

  // Nitrogen-to-active-SG-metal conductance.
  parameter SI.ThermalConductance G_N2_sg = P_r / abs(TN0 - Tsg0)
    "Conductance from N2 node to active SG metal [W/K]";

  // Active-SG-metal-to-water/steam conductance.
  // Calibrated so that at nominal conditions:
  //   Q = P_r = G_sg_fw*(Tsg0 - T_sat_steam)
  parameter SI.ThermalConductance G_sg_fw = P_r / abs(Tsg0 - T_sat_steam)
    "Conductance from active SG metal to secondary water/steam stream [W/K]";


  // ---------- THERMAL TIME CONSTANTS ----------
  // These convert a conductance G between two lumps into a time constant
  // for the form (T_other - T_this)/tau. In general:
  //   dT_this/dt includes (T_other - T_this) * (G / (M_this*cp_this))
  // so tau_this_other = (M_this*cp_this)/G.

  // Fuel <-> moderator link uses G_f_g
  parameter SI.Time tau_f_g   = M_f  * cp_f  / G_f_g "Fuel response to moderator through G_f_g [s]";
  parameter SI.Time tau_g_f   = M_g  * cp_g  / G_f_g "Moderator response to fuel through G_f_g [s]";

  // Moderator <-> heat pipe link uses G_g_hp
  parameter SI.Time tau_g_hp  = M_g  * cp_g  / G_g_hp;
  parameter SI.Time tau_hp_g  = M_hp * cp_hp / G_g_hp;

  // Heat pipe <-> N2 link uses G_hp_N2
  parameter SI.Time tau_hp_N2 = M_hp * cp_hp / G_hp_N2;
  parameter SI.Time tau_N2_hp = M_N2 * cp_N2 / G_hp_N2;

  // Diagnostics for the new active SG node.
  parameter SI.Time tau_N2_sg = M_N2 * cp_N2 / G_N2_sg
    "N2 response time through N2-to-SG conductance [s]";
  parameter SI.Time tau_sg_N2 = C_sg / G_N2_sg
    "Active SG metal response time through N2-side conductance only [s]";
  parameter SI.Time tau_sg_fw = C_sg / G_sg_fw
    "Active SG metal response time through secondary-side conductance only [s]";
  parameter SI.Time tau_sg_total = C_sg / (G_N2_sg + G_sg_fw)
    "Approximate active SG metal small-signal time constant [s]";

  // Precompute 1/(M_f*cp_f) because it is used repeatedly in dTf/dt.
  parameter Real inv_Mf_cp_f = 1.0 / (M_f * cp_f);


  // ---------- REACTIVITY FEEDBACK COEFFICIENTS ----------
  // These are temperature feedback coefficients in Δk/k per K.
  // Unit conversion: 1 pcm = 1e-5 Δk/k.
  parameter Real alpha_f = -4.59e-5 "Fuel feedback [Δk/k per K]";
  parameter Real alpha_m = -2.21e-5 "Moderator feedback [Δk/k per K]";


  // ---------- CONTROL DRUM WORTH ----------
  parameter Integer n_drums = 12;

  // Total worth at full insertion/rotation.
  // Given in pcm, converted to Δk/k with *1e-5 below.
  parameter Real rho_max_total_pcm = -13174;

  // Distribute total worth evenly among drums.
  parameter Real rho_max_per_drum = (rho_max_total_pcm*1e-5)/n_drums;

  // Steady-state angle used as the zero-reactivity reference point.
  parameter Real u0 = 45.0 "Steady-state drum angle [deg]";

  // Drum worth curve (cosine) for a single drum:
  //   rho_drum(angle) = rho_max_per_drum * (1 - cos(angle))/2
  parameter Real rho_ss_single =
    rho_max_per_drum * (1.0 - Modelica.Math.cos(Modelica.Constants.pi/180*u0)) / 2.0;

  // Total steady-state drum reactivity, used to shift rho so that
  // at (Tf0, Tm0, u0) the net reactivity is approximately zero.
  parameter Real rho_ss_total = n_drums * rho_ss_single;


  // ---------- INPUT ----------
  // External signal: the angle applied to all drums [deg].
  // RunOneProfile connects this from DrumProfileFromFile.
  input Real drumAngleDeg "Angle applied to all drums [deg]";


  // ---------- STATES ----------
  // Normalized power (n=1 => P=P_r). fixed=true ensures the solver uses these as
  // initial conditions rather than trying to solve them from other equations.
  Real n(start=1.0, fixed=true) "Normalized power";

  // Delayed neutron precursors (6 groups), initialized to steady state for n=1 and ρ=0.
  Real c[nGroups](start=C_ss, each fixed=true) "Delayed precursors";

  // Thermal node temperatures [K], initialized at design temperatures.
  SI.Temperature Tf(start=Tf0, fixed=true) "Fuel temperature [K]";
  SI.Temperature Tm(start=Tm0, fixed=true) "Moderator/graphite temperature [K]";
  SI.Temperature Thp(start=Thp0, fixed=true) "Heat pipe effective temperature [K]";
  SI.Temperature TN2(start=TN0, fixed=true) "Nitrogen loop average temperature [K]";

  // Active steam-generator heat-transfer metal node.
  SI.Temperature Tsg(start=Tsg0, fixed=true)
    "Active SG heat-transfer metal temperature [K]";


  // ---------- OUTPUTS ----------
  Real rho "Reactivity [Δk/k]";
  Real rho_drums "Drum reactivity contribution [Δk/k]";
  Real rho_fuel "Fuel temperature reactivity contribution [Δk/k]";
  Real rho_moderator "Moderator temperature reactivity contribution [Δk/k]";
  Real rho_dollars "Reactivity [$]";
  Real rho_drums_dollars "Drum reactivity contribution [$]";
  Real rho_fuel_dollars "Fuel reactivity contribution [$]";
  Real rho_moderator_dollars "Moderator reactivity contribution [$]";
  SI.Power P "Thermal power [W]";
  Real P_MW "Thermal power [MW]";

  // Steam-generator outlet quantities retained as public model variables.
  // These are the only new SG quantities intended for extraction by RunOneProfile.
  SI.Temperature T_steam_out
    "Calculated outlet temperature at fixed pressure [K]";

  Real x_steam_out(min=0.0, max=1.0)
    "Outlet steam quality/dryness fraction [-]";

protected
  // Internal SG heat-transfer variables used to compute Tsg and outlet state.
  // These are intentionally not exposed by RunOneProfile as LSTM targets.
  SI.Power Q_N2_to_sg
    "Internal heat transferred from nitrogen node to active SG metal [W]";

  SI.Power Q_sg_to_fw
    "Internal heat transferred from active SG metal to feedwater/steam stream [W]";

  SI.SpecificEnthalpy q_sg_per_kg
    "Internal specific heat addition to fixed feedwater stream [J/kg]";


equation
  // -------------------------
  // Power scaling
  // -------------------------
  // Convert normalized power to physical power.
  P = n * P_r;
  P_MW = P / 1e6;

  // -------------------------
  // Reactivity model
  // -------------------------
  // rho = drum contribution relative to steady-state
  //     + fuel temperature feedback
  //     + moderator temperature feedback.
  rho_drums =
    n_drums * rho_max_per_drum *
    (1.0 - Modelica.Math.cos(Modelica.Constants.pi/180*drumAngleDeg)) / 2.0
    - rho_ss_total;

  rho_fuel = alpha_f*(Tf - Tf0);
  rho_moderator = alpha_m*(Tm - Tm0);
  rho = rho_drums + rho_fuel + rho_moderator;

  // Convert reactivity to dollars ($) by dividing by β.
  // 1$ = β in Δk/k units.
  rho_dollars = rho / beta;
  rho_drums_dollars = rho_drums / beta;
  rho_fuel_dollars = rho_fuel / beta;
  rho_moderator_dollars = rho_moderator / beta;

  // -------------------------
  // Point kinetics equations
  // -------------------------
  // dn/dt = ((rho - beta)/Lambda)*n + Σ(lambda_i*c_i)
  // dc_i/dt = (beta_i/Lambda)*n - lambda_i*c_i
  der(n) =
    ((rho - beta) / Lambda) * n
    + sum(lambdas[i] * c[i] for i in 1:nGroups);

  for i in 1:nGroups loop
    der(c[i]) = (betas[i] / Lambda) * n - lambdas[i] * c[i];
  end for;

  // -------------------------
  // Thermal network ODEs
  // -------------------------
  // Each lump gets:
  //   reactor power deposition + heat exchange with neighboring nodes.

  // Fuel:
  //   - Receives heat_f * P
  //   - Exchanges heat with moderator node via (Tm - Tf)/tau_f_g
  der(Tf) =
    heat_f*P*inv_Mf_cp_f
    + (Tm - Tf)/tau_f_g;

  // Moderator/graphite:
  //   - Receives (1-heat_f) * P
  //   - Exchanges with fuel and heat pipe
  der(Tm) =
    (1.0 - heat_f)*P/(M_g*cp_g)
    + (Tf - Tm)/tau_g_f
    + (Thp - Tm)/tau_g_hp;

  // Heat pipe:
  //   - Exchanges with moderator and N2
  der(Thp) =
    (Tm - Thp)/tau_hp_g
    + (TN2 - Thp)/tau_hp_N2;

  // -------------------------
  // Steam-generator thermal node
  // -------------------------

  // Heat transferred from nitrogen loop to active SG metal.
  Q_N2_to_sg = G_N2_sg*(TN2 - Tsg);

  // Heat transferred from active SG metal to the secondary water/steam stream.
  // The nonnegative clamp prevents reverse "steam production" if Tsg falls below
  // saturation temperature. This is a simplified operational assumption intended
  // for transients near nominal power.
  Q_sg_to_fw =
    noEvent(if G_sg_fw*(Tsg - T_sat_steam) > 0.0 then
      G_sg_fw*(Tsg - T_sat_steam)
    else
      0.0);

  // Active SG metal energy balance:
  //   C_sg*dTsg/dt = heat from N2 - heat to water/steam.
  der(Tsg) =
    (Q_N2_to_sg - Q_sg_to_fw)/C_sg;

  // Nitrogen loop average node:
  //   - Exchanges with heat pipe
  //   - Loses heat to active SG metal
  der(TN2) =
    (Thp - TN2)/tau_N2_hp
    - Q_N2_to_sg/(M_N2*cp_N2);

  // -------------------------
  // Variable outlet steam condition at fixed pressure
  // -------------------------

  // Specific heat addition to the fixed feedwater stream.
  q_sg_per_kg = Q_sg_to_fw / m_dot_fw;

  // Outlet quality.
  // If q_sg_per_kg < h_fg, the outlet is a wet saturated mixture.
  // If q_sg_per_kg >= h_fg, the outlet is dry saturated or superheated.
  x_steam_out =
    noEvent(if q_sg_per_kg <= 0.0 then
      0.0
    elseif q_sg_per_kg < h_fg then
      q_sg_per_kg/h_fg
    else
      1.0);

  // Outlet temperature.
  // Below dryout, temperature remains at saturation temperature.
  // Above dryout, added heat becomes superheat.
  T_steam_out =
    noEvent(if q_sg_per_kg < h_fg then
      T_sat_steam
    else
      T_sat_steam + (q_sg_per_kg - h_fg)/cp_sh_eff);


end HPMicroPK;
