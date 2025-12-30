within ;
package MicroreactorPK
  /*
    PACKAGE OVERVIEW
    ----------------
    This package contains a small “driver + plant” simulation:

      1) DrumProfileFromFile  (block)
         - Reads a time-varying control drum angle (deg) from a MAT-file table.
         - Produces angleDeg(t) using linear interpolation.

      2) HPMicroPK (model)
         - A lumped point-kinetics reactor model with:
             * 6 delayed neutron precursor groups
             * 5-node thermal network (fuel, moderator/graphite, heat-pipe, N2, steam generator)
             * drum reactivity + temperature feedback

      3) RunOneProfile (model)
         - Wires (1) -> (2) so the drum angle profile drives the reactor.
         - Sets an experiment stop time and tolerance.

    Notes:
    - Modelica is declarative: you write equations, not a step-by-step program.
    - der(x) means dx/dt (time derivative). The solver integrates these ODEs.
  */

  import SI = Modelica.Units.SI;  // Convenient alias: SI.Temperature, SI.Power, etc.

  // ------------------------------------------------------------
  // Reads an Nx2 table [time, angle_deg] from a .mat file variable
  // named `profile` (or whatever tableName is set to).
  //
  // Behavior:
  // - Linear interpolation between table points
  // - Holds the last value after the final time (HoldLastPoint)
  // - (By default) if time is before the first row, it also holds the first point
  //
  // ------------------------------------------------------------
  block DrumProfileFromFile

    // Absolute path to the .mat file containing the drum profile data.
    parameter String fileName =
      "C:/Users/Logan/Desktop/mit_coursework/eVinci/drum_profile_matern52_ell5_velGP_draw0.mat";

    // Name of the variable (table) inside the MAT-file.
    // The variable should be an Nx2 matrix: [time, angleDeg]
    parameter String tableName = "profile";

    // Output signal: the interpolated drum angle in degrees.
    output Real angleDeg "Drum angle [deg]";

  protected
    /*
      CombiTimeTable is a standard Modelica block that outputs columns of a table
      as a function of time.

      Key settings here:
      - tableOnFile=true: table is read from an external file (MAT-file).
      - columns={2}: we output column 2, which is the angle.
                   Column 1 is implicitly the time column.
      - smoothness=LinearSegments: piecewise linear interpolation.
      - extrapolation=HoldLastPoint: after the last time, hold the last value.
    */
    Modelica.Blocks.Sources.CombiTimeTable tab(
      tableOnFile=true,
      fileName=fileName,
      tableName=tableName,
      columns={2}, // column 2 = angle_deg, column 1 = time
      smoothness=Modelica.Blocks.Types.Smoothness.LinearSegments,
      extrapolation=Modelica.Blocks.Types.Extrapolation.HoldLastPoint);

  equation
    /*
      Safety/physical constraint:
      - The control drum is only meaningful in [0, 180] degrees in this model.
      - Clip the interpolated value to keep the rest of the model sane.
        (This avoids weird file issues like negative angles or >180 degrees.)
    */
    angleDeg = max(0.0, min(180.0, tab.y[1]));
  end DrumProfileFromFile;
  // ------------------------------------------------------------
  // Point-kinetics + 5-node thermal network + 12 drums (same angle)
  //
  // State summary:
  //   n(t)      : normalized reactor power (n=1 -> P = P_r)
  //   c[i](t)   : normalized delayed precursor concentrations (6 groups)
  //   Tf(t)     : fuel temperature
  //   Tm(t)     : moderator / graphite / structure temperature
  //   Thp(t)    : heat pipe effective temperature node
  //   TN2(t)    : nitrogen loop average temperature node
  //   Tsg(t)    : steam generator metal / primary-side lump temperature node
  //
  // Inputs:
  //   drumAngleDeg(t) applied to all drums equally
  //
  // Outputs:
  //   rho(t)        : reactivity in Δk/k
  //   rho_dollars(t): reactivity in $
  //   P(t)          : thermal power in W
  //   P_MW(t)       : thermal power in MW
  // ------------------------------------------------------------
  model HPMicroPK

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


    // ---------- THERMAL POWER SCALE ----------

    // Rated thermal power [W]. This is the scaling for normalized power n.
    parameter SI.Power P_r = 6.0e6 "Rated thermal power [W]";

    // Fraction of thermal power deposited directly in the fuel node.
    // (The remainder (1-heat_f) goes into the moderator/structure node in this lumped model.)
    parameter Real heat_f = 0.90 "Fraction of power deposited in fuel";


    // ---------- THERMAL DESIGN TEMPERATURES ----------
    // These define the steady-state point around which:
    // - reactivity feedback is computed (Tf - Tf0, Tm - Tm0)
    // - UA values (G_*) are computed using ΔT design splits

    parameter SI.Temperature Tf0   = 1173.15; // fuel nominal [K]
    parameter SI.Temperature Tm0   = 1150.15; // moderator/graphite nominal [K]
    parameter SI.Temperature Thp0  = 1073.15; // heat pipe nominal [K]

    // N2 inlet/outlet design temps; TN0 is the average used as the N2 node reference
    parameter SI.Temperature TN_in = 683.15;   // N2 inlet [K]
    parameter SI.Temperature TN_out= 1073.15;  // N2 outlet [K]
    parameter SI.Temperature TN0   = 0.5*(TN_in + TN_out); // N2 average [K]

    // Steam generator node design temperature and feedwater inlet temperature
    parameter SI.Temperature Tsg0  = 232.0 + 273.15;   // SG node nominal [K]
    parameter SI.Temperature T_fw_in = 198.3 + 273.15; // feedwater inlet [K]
    parameter SI.Temperature T_steam_out = 232.0 + 273.15; // Steam outlet [K]


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
    // We treat N2 as a well-mixed "average" node with mass set by:
    //   M_N2 = m_dot_N * tau_N
    // where tau_N is an assumed residence time / transport lag.
    parameter SI.SpecificHeatCapacity cp_N2 = 1182.0;
    parameter SI.Time tau_N = 5.0;

    // Compute the N2 mass flow required to remove P_r given N2 ΔT design
    //   P_r ≈ m_dot * cp * (TN_out - TN_in)
    parameter SI.MassFlowRate m_dot_N = P_r / (cp_N2*(TN_out - TN_in));

    // Effective N2 mass in the “average” control volume
    parameter SI.Mass M_N2 = m_dot_N * tau_N;

    // Steam generator lump cp (metal / primary-side effective)
    parameter SI.SpecificHeatCapacity cp_sg = 500.0;

    // Steam-side “effective cp”:
    // Instead of explicitly modeling two-phase and enthalpy, approximate the
    // steam-side heat removal as:
    //   Q = m_dot_s * cp_eff * (Tsg - T_fw_in)
    // where cp_eff is chosen so that at design point the enthalpy rise is delta_h.
    parameter Real delta_h = 2035.87e3 "J/kg"; // effective enthalpy gain for 198C FW -> 232C steam [J/kg]
    //     parameter SI.MassFlowRate m_dot_s = P_r / delta_h; // steam mass flow to carry P_r
    //     parameter SI.TemperatureDifference deltaT_sg_fw = (Tsg0 - T_fw_in);
    //     parameter SI.SpecificHeatCapacity cp_eff = delta_h / deltaT_sg_fw;




    // ---------- CONDUCTION / HEAT-TRANSFER UA's (G = UA) FROM DESIGN TEMPS ----------
    // Reverse-engineering effective thermal conductances between lumps
    // from the design temperature drops at rated power:
    //
    //   Q = G * ΔT  =>  G = Q / ΔT
    //
    // abs() is used so G is positive regardless of temperature ordering.

    parameter SI.ThermalConductance G_f_g  = P_r / abs(Tf0  - Tm0);
    parameter SI.ThermalConductance G_g_hp = P_r / abs(Tm0  - Thp0);
    parameter SI.ThermalConductance G_hp_N2= P_r / abs(Thp0 - TN0);
    parameter SI.ThermalConductance G_N2_sg= P_r / abs(TN0  - Tsg0);

    // SG mass from a desired SG time constant tau_sg:
    // For a first-order lump: tau ≈ (M*cp)/G  =>  M ≈ tau*G/cp
    parameter SI.Time tau_sg = 30.0;
    parameter SI.Mass M_sg = tau_sg * G_N2_sg / cp_sg;


    // ---------- THERMAL TIME CONSTANTS ----------
    // These convert a conductance G between two lumps into a “time constant”
    // for the form (T_other - T_this)/tau. In general:
    //   dT_this/dt includes (T_other - T_this) * (G / (M_this*cp_this))
    // so tau_this_other = (M_this*cp_this)/G

    // Fuel <-> moderator link uses G_f_g
    parameter SI.Time tau_f_g   = M_f  * cp_f  / G_f_g; // fuel responding to moderator
    parameter SI.Time tau_g_f   = M_g  * cp_g  / G_f_g; // moderator responding to fuel

    // Moderator <-> heat pipe link uses G_g_hp
    parameter SI.Time tau_g_hp  = M_g  * cp_g  / G_g_hp;
    parameter SI.Time tau_hp_g  = M_hp * cp_hp / G_g_hp;

    // Heat pipe <-> N2 link uses G_hp_N2
    parameter SI.Time tau_hp_N2 = M_hp * cp_hp / G_hp_N2;
    parameter SI.Time tau_N2_hp = M_N2 * cp_N2 / G_hp_N2;

    // N2 <-> SG link uses G_N2_sg
    parameter SI.Time tau_N2_sg = M_N2 * cp_N2 / G_N2_sg;
    parameter SI.Time tau_sg_N2 = M_sg * cp_sg / G_N2_sg;

    // Steam-side convective removal from SG node:
    //   Q_out ≈ m_dot_s * cp_eff * (Tsg - T_fw_in)
    // Represented as a first-order sink:
    //   dTsg/dt includes -(Tsg - T_fw_in)/tau_conv
    // with tau_conv = (M_sg*cp_sg)/(m_dot_s*cp_eff)
    //     parameter SI.Time tau_conv  = M_sg * cp_sg / (m_dot_s * cp_eff);

    // Precompute 1/(M_f*cp_f) because it’s used repeatedly in dTf/dt
    parameter Real inv_Mf_cp_f = 1.0 / (M_f * cp_f);


    // ---------- REACTIVITY FEEDBACK COEFFICIENTS ----------
    // These are temperature feedback coefficients in Δk/k per K.
    // unit conversion: 1 pcm = 1e-5 Δk/k
    // So -4.59 pcm/K -> -4.59e-5 Δk/k per K
    parameter Real alpha_f = -4.59e-5 "Fuel feedback [Δk/k per K]";
    parameter Real alpha_m = -2.21e-5 "Moderator feedback [Δk/k per K]";


    // ---------- CONTROL DRUM WORTH ----------
    parameter Integer n_drums = 12;

    // Total worth at full insertion/rotation
    // Given in pcm, converted to Δk/k with *1e-5 below.
    parameter Real rho_max_total_pcm = -13174;

    // Distribute total worth evenly among drums (still in Δk/k after conversion).
    // Note: this assumes all drums have identical worth.
    parameter Real rho_max_per_drum = (rho_max_total_pcm*1e-5)/n_drums;

    // “Steady-state” angle used as the zero-reactivity reference point.
    parameter Real u0 = 45.0 "Steady-state drum angle [deg]";

    // Drum worth curve (cosine) for a single drum:
    //   rho_drum(angle) = rho_max_per_drum * (1 - cos(angle))/2
    //
    // Evaluate it at u0 to get the steady-state reactivity contribution of one drum.
    parameter Real rho_ss_single =
      rho_max_per_drum * (1.0 - Modelica.Math.cos(Modelica.Constants.pi/180*u0)) / 2.0;

    // Total steady-state drum reactivity (all drums), used to shift rho so that
    // at (Tf0, Tm0, u0) the net reactivity is ~0.
    parameter Real rho_ss_total = n_drums * rho_ss_single;


    // ---------- INPUT ----------
    // External signal: the angle applied to all drums [deg].
    // RunOneProfile connects this from DrumProfileFromFile.
    input Real drumAngleDeg "Angle applied to all drums [deg]";


    // ---------- STATES ----------
    // Normalized power (n=1 => P=P_r). fixed=true ensures the solver uses these as
    // initial conditions rather than trying to solve them from other equations.
    Real n(start=1.0, fixed=true) "Normalized power";

    // Delayed neutron precursor states, normalized so that at steady power n=1,
    // the steady precursor values are c[i]=1.
    Real c[nGroups](each start=1.0, each fixed=true) "Normalized precursors";

    // Thermal node temperatures [K], initialized at design temps.
    SI.Temperature Tf(start=Tf0, fixed=true);
    SI.Temperature Tm(start=Tm0, fixed=true);
    SI.Temperature Thp(start=Thp0, fixed=true);
    SI.Temperature TN2(start=TN0, fixed=true);
    // SI.Temperature Tsg(start=Tsg0, fixed=true);


    // ---------- OUTPUTS ----------
    Real rho "Reactivity [Δk/k]";
    Real rho_dollars "Reactivity [$]"; // rho/beta
    SI.Power P "Thermal power [W]";
    Real P_MW "Thermal power [MW]";

    SI.Power Q_to_steam "Heat available to generate 232C steam [W]";
    SI.MassFlowRate m_dot_steam "Steam production rate [kg/s]";

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
    // rho = (drum contribution relative to steady-state) + (fuel temp feedback) + (moderator temp feedback)
    //
    // Drum cosine worth:
    //   rho_drums(angle) = n_drums * rho_max_per_drum * (1 - cos(angle))/2
    //
    // Subtract rho_ss_total so that at u0 and nominal temps, drum contribution is zeroed.
    //
    // Temperature feedback:
    //   alpha_f*(Tf - Tf0) + alpha_m*(Tm - Tm0)
    // Negative alphas mean hotter temps reduce reactivity (stabilizing feedback).
    rho =
      ( n_drums * rho_max_per_drum *
        (1.0 - Modelica.Math.cos(Modelica.Constants.pi/180*drumAngleDeg)) / 2.0
        - rho_ss_total)
      + alpha_f*(Tf - Tf0)
      + alpha_m*(Tm - Tm0);

    // Convert reactivity to dollars ($) by dividing by β.
    // 1$ = β in Δk/k units.
    rho_dollars = rho / beta;

    // -------------------------
    // Point kinetics equations
    // -------------------------
    // Normalized form (n and c are normalized so that steady n=1 implies steady c=1)
    //
    // dn/dt = [ (rho - beta)*n + Σ(beta_i*c_i) ] / Lambda
    //
    // dc_i/dt = λ_i * (n - c_i)
    //
    // Interpretation:
    // - Power rises quickly with positive reactivity
    // - Precursors lag behind changes in n with their decay constants
    der(n) =
      ( ((rho - beta)*n) + sum(betas[i]*c[i] for i in 1:nGroups)) / Lambda;

    for i in 1:nGroups loop
      der(c[i]) = lambdas[i] * (n - c[i]);
    end for;

    // -------------------------
    // Thermal network ODEs
    // -------------------------
    // Each lump gets:
    //   (source from reactor power deposition) + (heat exchange with neighbors)
    //
    // Signs:
    // - (T_other - T_this)/tau is positive when other is hotter (heats this node)
    // - negative when this node is hotter (cools this node)

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

    // Heat transferred to steam boundary at T_steam_out
    Q_to_steam = G_N2_sg*(TN2 - T_steam_out);

    // Steam rate, heat available / enthalpy needed per kg
    m_dot_steam = Q_to_steam / delta_h;


    // Nitrogen loop average node:
    //   - Exchanges with heat pipe and steam generator node
    der(TN2) =
      (Thp - TN2)/tau_N2_hp
      - Q_to_steam/(M_N2*cp_N2);

    // Steam generator lump:
    //   - Receives heat from N2
    //   - Rejects heat to feedwater as a first-order sink to T_fw_in
    // der(Tsg) = (TN2 - Tsg)/tau_sg_N2 - (Tsg - T_fw_in)/tau_conv;

  end HPMicroPK;
  // ------------------------------------------------------------
  // Single profile runner: file -> angle -> reactor
  //
  // “top-level” model to simulate:
  //   DrumProfileFromFile reads the time profile
  //   HPMicroPK uses that profile as its drum input
  // ------------------------------------------------------------
  model RunOneProfile

    // Pass-through parameters to pick the file/table at the top level
    parameter String profileFile =
      "C:/Users/Logan/Desktop/mit_coursework/eVinci/drum_profile_matern52_ell5_velGP_draw0.mat";
    parameter String tableName = "profile";

    // Instantiate the profile reader
    DrumProfileFromFile prof(fileName=profileFile, tableName=tableName);

    // Instantiate the reactor model
    HPMicroPK reactor;

  equation
    // Wire the profile output into the reactor input
    reactor.drumAngleDeg = prof.angleDeg;

    // Simulation settings
    // - StopTime: end of simulation in seconds
    // - Tolerance: solver error tolerance (smaller = more accurate, slower)
    annotation(experiment(StopTime=200.0, Tolerance=1e-8));
  end RunOneProfile;
end MicroreactorPK;
