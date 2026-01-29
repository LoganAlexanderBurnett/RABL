within MicroreactorPK.Blocks;
block DrumProfileFromFile
  /*
    Reads an Nx5 table [time, angle_deg, vel, acc, f_divert] from a .mat file variable
    named `profile`

    Behavior:
    - Linear interpolation between table points
    - Holds the last value after the final time (HoldLastPoint)
    - (By default) if time is before the first row, it also holds the first point
  */

  import SI = Modelica.Units.SI;

  // Path to the MAT-file containing the table (Matlab v4).
  // Leave empty by default and set from Python.
  parameter String fileName = "" "Path to .mat file containing profile table";

  // Name of the matrix variable inside the MAT-file (e.g., "profile").
  parameter String tableName = "profile";

  // Which column contains the drum angle. Column 1 is time.
  parameter Integer angleColumn(min=2) = 2 "Column index for angle";
  parameter Integer velColumn(min=2) = 3;
  parameter Integer accColumn(min=2) = 4;
  parameter Integer divertColumn(min=2) = 5 "Column index for diversion fraction (time is column 1)";

  // Output signal: interpolated drum angle in degrees.
  output Real angleDeg "Drum angle [deg]";
  output Real velDeg_s "Drum velocity [deg/s]";
  output Real accDeg_s2 "Drum acceleration [deg/s2]";
  output Real f_divert "Diversion fraction (0..1)";

protected
  Modelica.Blocks.Sources.CombiTimeTable tab(
    tableOnFile=true,
    fileName=fileName,
    tableName=tableName,
    columns={angleColumn, velColumn, accColumn, divertColumn},
    smoothness=Modelica.Blocks.Types.Smoothness.LinearSegments,
    extrapolation=Modelica.Blocks.Types.Extrapolation.HoldLastPoint);

equation
  // No angle clipping; assume the profile is already valid
  angleDeg  = tab.y[1];
  velDeg_s  = tab.y[2];
  accDeg_s2 = tab.y[3];
  f_divert  = tab.y[4];
  assert(f_divert >= 0 and f_divert <= 1,
         "DrumProfileFromFile: f_divert must be in [0,1]. Check the MAT profile column 5.");
end DrumProfileFromFile;
