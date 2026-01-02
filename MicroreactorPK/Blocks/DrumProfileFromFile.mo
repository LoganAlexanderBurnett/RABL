within MicroreactorPK.Blocks;
block DrumProfileFromFile
  /*
    Reads an Nx2 table [time, angle_deg] from a .mat file variable
    named `profile` (or whatever tableName is set to).

    Behavior:
    - Linear interpolation between table points
    - Holds the last value after the final time (HoldLastPoint)
    - (By default) if time is before the first row, it also holds the first point
  */

  import SI = Modelica.Units.SI; // convenient alias if needed by users

  // Path to the MAT-file containing the table (Matlab v4 recommended).
  // Leave empty by default and set from Python.
  parameter String fileName = "" "Path to .mat file containing profile table";

  // Name of the matrix variable inside the MAT-file (e.g., "profile").
  parameter String tableName = "profile";

  // Which column contains the drum angle. Column 1 is time.
  // For a [time, angle] table this should be 2.
  parameter Integer angleColumn(min=2) = 2 "Column index for angle (time is column 1)";

  // Output signal: interpolated drum angle in degrees.
  output Real angleDeg "Drum angle [deg]";

protected
  Modelica.Blocks.Sources.CombiTimeTable tab(
    tableOnFile=true,
    fileName=fileName,
    tableName=tableName,
    columns={angleColumn},
    smoothness=Modelica.Blocks.Types.Smoothness.LinearSegments,
    extrapolation=Modelica.Blocks.Types.Extrapolation.HoldLastPoint);

equation
  // No angle clipping; assume the profile is already valid
  angleDeg = tab.y[1];
end DrumProfileFromFile;
