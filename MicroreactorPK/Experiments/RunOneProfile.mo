within MicroreactorPK.Experiments;
model RunOneProfile
  import SI = Modelica.Units.SI;
  import MicroreactorPK.Blocks.DrumProfileFromFile;
  import MicroreactorPK.Models.HPMicroPK;

  // Set these from Python for each run
  parameter String profileFile = "" "Path to MAT-file containing the profile matrix";
  parameter String tableName   = "profile" "Matrix variable name in MAT-file";
  parameter Integer angleColumn(min=2) = 2 "Angle column in table (time is column 1)";

  // Instantiate the profile reader
  DrumProfileFromFile prof(
    fileName=profileFile,
    tableName=tableName,
    angleColumn=angleColumn);

  // Instantiate the reactor model
  HPMicroPK reactor;

  // -------------------------
  // Convenience outputs
  // -------------------------
  output SI.Time t "Time [s]";

  output SI.Temperature TN2 "Reactor N2 node temperature [K]";
  output SI.Temperature Tm  "Moderator/graphite temperature [K]";
  output SI.Temperature Thp "Heat-pipe node temperature [K]";
  output SI.Temperature Tf  "Fuel temperature [K]";

  output SI.TemperatureRate dTN2 "d(TN2)/dt [K/s]";
  output SI.TemperatureRate dTm  "d(Tm)/dt [K/s]";
  output SI.TemperatureRate dThp "d(Thp)/dt [K/s]";
  output SI.TemperatureRate dTf  "d(Tf)/dt [K/s]";

  output Real c[6]   "Delayed neutron precursor states c[1..6]";
  output Real dc[6]  "d(c[i])/dt [1/s]";

  output Real n   "Normalized power";
  output Real dn  "d(n)/dt [1/s]";

  output Real P_MW "Thermal power [MW]";
  output Real drumAngleDeg "Applied drum angle [deg]";
  output Real rho "Reactivity [Δk/k]";
  output Real rho_dollars "Reactivity [$]";

  output SI.MassFlowRate m_dot_steam "Steam production rate [kg/s]";
  output SI.Power Q_to_steam "Heat available to steam [W]";

equation
  // Wire the profile output into the reactor input
  reactor.drumAngleDeg = prof.angleDeg;

  // Time
  t = time;

  // Alias “nice” outputs for easy extraction / column naming
  drumAngleDeg = reactor.drumAngleDeg;

  TN2 = reactor.TN2;
  Tm  = reactor.Tm;
  Thp = reactor.Thp;
  Tf  = reactor.Tf;

  dTN2 = der(reactor.TN2);
  dTm  = der(reactor.Tm);
  dThp = der(reactor.Thp);
  dTf  = der(reactor.Tf);

  c  = reactor.c;
  for i in 1:6 loop
    dc[i] = der(reactor.c[i]);
  end for;

  n  = reactor.n;
  dn = der(reactor.n);

  P_MW = reactor.P_MW;
  rho = reactor.rho;
  rho_dollars = reactor.rho_dollars;

  m_dot_steam = reactor.m_dot_steam;
  Q_to_steam  = reactor.Q_to_steam;

  // Simulation settings (you can override stopTime in simulateModel from Python anyway)
  annotation(experiment(StopTime=200.0, Tolerance=1e-8));
end RunOneProfile;
