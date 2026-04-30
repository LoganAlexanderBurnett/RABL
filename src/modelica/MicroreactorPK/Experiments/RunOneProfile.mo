within MicroreactorPK.Experiments;
model RunOneProfile
  import SI = Modelica.Units.SI;
  import MicroreactorPK.Blocks.DrumProfileFromFile;
  import MicroreactorPK.Models.HPMicroPK;

  // Set these from Python/Dymola for each run.
  // `profileFile` may be either:
  // 1) a full profile that starts at t=0, or
  // 2) a suffix profile that starts at a nonzero absolute branch time
  //    when running restart-based branch simulations.
  parameter String profileFile = ""
    annotation(Evaluate=false);
  parameter String tableName   = "profile"
    "Matrix variable name in MAT-file"
    annotation(Evaluate=false);
  parameter Integer angleColumn(min=2) = 2
    annotation(Evaluate=false);
  parameter Integer velColumn(min=2) = 3
    annotation(Evaluate=false);
  parameter Integer accColumn(min=2) = 4
    annotation(Evaluate=false);

  // ------------------------------------------------------------
  // Steam-generator / secondary-side pass-through parameters
  // ------------------------------------------------------------
  // These are passed to HPMicroPK. Defaults match the active-SG-node model.

  parameter SI.MassFlowRate m_dot_fw = 2.95
    "Fixed feedwater mass flow rate [kg/s]"
    annotation(Evaluate=false);

  parameter SI.Pressure p_steam_out = 1.5e6
    "Fixed steam outlet/header pressure [Pa]"
    annotation(Evaluate=false);

  parameter SI.Temperature T_fw_in = 198.3 + 273.15
    "Feedwater inlet temperature [K]"
    annotation(Evaluate=false);

  parameter SI.Temperature T_sat_steam = 198.3 + 273.15
    "Saturation temperature at fixed steam pressure [K]"
    annotation(Evaluate=false);

  parameter SI.Temperature T_steam_out_nom = 232.0 + 273.15
    "Nominal outlet steam temperature [K]"
    annotation(Evaluate=false);

  parameter SI.SpecificEnthalpy h_fg = 1946.45e3
    "Latent heat from saturated liquid to saturated vapor near 1.5 MPa [J/kg]"
    annotation(Evaluate=false);

  parameter SI.SpecificEnthalpy h_sh_nom = 89.42e3
    "Nominal superheat enthalpy rise from saturation to 232 C [J/kg]"
    annotation(Evaluate=false);

  parameter SI.HeatCapacity C_sg = 2.55e5
    "Active SG heat-transfer metal capacitance [J/K]"
    annotation(Evaluate=false);

  // Instantiate the profile reader.
  DrumProfileFromFile prof(
    fileName=profileFile,
    tableName=tableName,
    angleColumn=angleColumn,
    velColumn=velColumn,
    accColumn=accColumn);

  // Instantiate the reactor model and bind the drum input to the profile.
  HPMicroPK reactor(
    drumAngleDeg=prof.angleDeg,
    m_dot_fw=m_dot_fw,
    p_steam_out=p_steam_out,
    T_fw_in=T_fw_in,
    T_sat_steam=T_sat_steam,
    T_steam_out_nom=T_steam_out_nom,
    h_fg=h_fg,
    h_sh_nom=h_sh_nom,
    C_sg=C_sg);

  // -------------------------
  // Convenience outputs
  // -------------------------
  output SI.Time t "Time [s]";

  output SI.Temperature TN2 "Reactor N2 node temperature [K]";
  output SI.Temperature Tm  "Moderator/graphite temperature [K]";
  output SI.Temperature Thp "Heat-pipe node temperature [K]";
  output SI.Temperature Tf  "Fuel temperature [K]";

  // New SG variables exposed for LSTM training.
  output SI.Temperature Tsg
    "Active SG heat-transfer metal temperature [K]";
  output SI.Temperature T_steam_out
    "Calculated outlet temperature at fixed steam pressure [K]";
  output Real x_steam_out
    "Outlet steam quality / dryness fraction [-]";

  output SI.TemperatureSlope dTN2 "d(TN2)/dt [K/s]";
  output SI.TemperatureSlope dTm  "d(Tm)/dt [K/s]";
  output SI.TemperatureSlope dThp "d(Thp)/dt [K/s]";
  output SI.TemperatureSlope dTf  "d(Tf)/dt [K/s]";

  output Real c[6]   "Delayed neutron precursor states c[1..6]";

  output Real n   "Normalized power";
  output Real dn  "d(n)/dt [1/s]";

  output Real P_MW "Thermal power [MW]";
  output Real drumAngleDeg "Applied drum angle [deg]";
  output Real drumVelDeg_s "Drum velocity [deg/s]";
  output Real drumAccDeg_s2 "Drum acceleration [deg/s2]";
  output Real rho_dollars "Reactivity [$]";
  output Real rho_drums_dollars "Drum reactivity contribution [$]";
  output Real rho_fuel_dollars "Fuel reactivity contribution [$]";
  output Real rho_moderator_dollars "Moderator reactivity contribution [$]";

equation
  // Time.
  t = time;

  // Alias outputs for easy extraction / column naming.
  drumAngleDeg = reactor.drumAngleDeg;
  drumVelDeg_s  = prof.velDeg_s;
  drumAccDeg_s2 = prof.accDeg_s2;

  TN2 = reactor.TN2;
  Tm  = reactor.Tm;
  Thp = reactor.Thp;
  Tf  = reactor.Tf;

  Tsg = reactor.Tsg;
  T_steam_out = reactor.T_steam_out;
  x_steam_out = reactor.x_steam_out;

  dTN2 = der(reactor.TN2);
  dTm  = der(reactor.Tm);
  dThp = der(reactor.Thp);
  dTf  = der(reactor.Tf);

  c  = reactor.c;

  n  = reactor.n;
  dn = der(reactor.n);

  P_MW = reactor.P_MW;
  rho_dollars = reactor.rho_dollars;
  rho_drums_dollars = reactor.rho_drums_dollars;
  rho_fuel_dollars = reactor.rho_fuel_dollars;
  rho_moderator_dollars = reactor.rho_moderator_dollars;

  // Simulation settings. StopTime can still be overridden from Python/Dymola.
  annotation(experiment(StopTime=200.0, Tolerance=1e-8));
end RunOneProfile;
