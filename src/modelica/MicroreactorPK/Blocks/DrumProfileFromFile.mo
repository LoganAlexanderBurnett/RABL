within MicroreactorPK.Blocks;
block DrumProfileFromFile
  /*
    Reads an Nx4 table [time, angle_deg, vel, acc] from a .mat file variable

    IMPORTANT: time in column 1 must be absolute simulation time.
    For restart runs, suffix tables should start at the branch time (not at 0).
    named `profile`

    Behavior:
    - Linear interpolation between table points
    - Holds the last value after the final time (HoldLastPoint)
    - (By default) if time is before the first row, it also holds the first point
  */

  import SI = Modelica.Units.SI;

  // Path to the MAT-file containing the table (Matlab v4).
  // Leave empty by default and set from Python.
  parameter String fileName = ""
    "Path to .mat file containing profile table"
    annotation(Evaluate=false);

  // Name of the matrix variable inside the MAT-file (e.g., "profile").
  parameter String tableName = "profile"
    annotation(Evaluate=false);

  // Which column contains the drum angle. Column 1 is time.
  parameter Integer angleColumn(min=2) = 2
    "Column index for angle"
    annotation(Evaluate=false);
  parameter Integer velColumn(min=2) = 3
    annotation(Evaluate=false);
  parameter Integer accColumn(min=2) = 4
    annotation(Evaluate=false);

  // Output signal: interpolated drum angle in degrees.
  output Real angleDeg "Drum angle [deg]";
  output Real velDeg_s "Drum velocity [deg/s]";
  output Real accDeg_s2 "Drum acceleration [deg/s2]";

protected
  Modelica.Blocks.Sources.CombiTimeTable tab(
    tableOnFile=true,
    fileName=fileName,
    tableName=tableName,
    columns={angleColumn, velColumn, accColumn},
    smoothness=Modelica.Blocks.Types.Smoothness.LinearSegments,
    extrapolation=Modelica.Blocks.Types.Extrapolation.HoldLastPoint);

equation
  // No angle clipping; assume the profile is already valid
  angleDeg  = tab.y[1];
  velDeg_s  = tab.y[2];
  accDeg_s2 = tab.y[3];
end DrumProfileFromFile;
