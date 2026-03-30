within MicroreactorPK.Experiments;
model RunOneProfile
  import SI = Modelica.Units.SI;
  import MicroreactorPK.Blocks.DrumProfileFromFile;
  import MicroreactorPK.Models.HPMicroPK;

  // Set these from Python for each run.
  // `profileFile` may be either:
  // 1) a full profile that starts at t=0, or
  // 2) a suffix profile that starts at a nonzero absolute branch time
  //    when running restart-based branch simulations.
  parameter String profileFile = ""
    annotation(Evaluate=false);
  parameter String tableName   = "profile"
    annotation(Evaluate=false)
    "Matrix variable name in MAT-file";
  parameter Integer angleColumn(min=2) = 2
    annotation(Evaluate=false);
  parameter Integer velColumn(min=2) = 3
    annotation(Evaluate=false);
  parameter Integer accColumn(min=2) = 4
    annotation(Evaluate=false);

  // Instantiate the profile reader
  DrumProfileFromFile prof(
    fileName=profileFile,
    tableName=tableName,
    angleColumn=angleColumn,
    velColumn=velColumn,
    accColumn=accColumn);


  // Instantiate the reactor model and bind the drum input to the profile
  HPMicroPK reactor(drumAngleDeg = prof.angleDeg);

  // -------------------------
  // Convenience outputs
  // -------------------------
  output SI.Time t "Time [s]";

  output SI.Temperature TN2 "Reactor N2 node temperature [K]";
  output SI.Temperature Tm  "Moderator/graphite temperature [K]";
  output SI.Temperature Thp "Heat-pipe node temperature [K]";
  output SI.Temperature Tf  "Fuel temperature [K]";

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

  output SI.MassFlowRate m_dot_steam "Steam production rate [kg/s]";
  output SI.Power Q_to_steam "Heat available to steam [W]";

equation
  // Time
  t = time;

  // Alias outputs for easy extraction / column naming
  drumAngleDeg = reactor.drumAngleDeg;
  drumVelDeg_s  = prof.velDeg_s;
  drumAccDeg_s2 = prof.accDeg_s2;


  TN2 = reactor.TN2;
  Tm  = reactor.Tm;
  Thp = reactor.Thp;
  Tf  = reactor.Tf;

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

  m_dot_steam = reactor.m_dot_steam;
  Q_to_steam  = reactor.Q_to_steam;

  // Simulation settings (can override stopTime in simulateModel from Python)
  annotation(experiment(StopTime=200.0, Tolerance=1e-8));
end RunOneProfile;
