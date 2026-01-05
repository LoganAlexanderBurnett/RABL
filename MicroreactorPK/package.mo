within ;
package MicroreactorPK
  extends Modelica.Icons.Package;
  /*
    PACKAGE OVERVIEW
    ----------------
    This package contains a small “driver + plant” simulation:

      1) Blocks.DrumProfileFromFile  (block)
         - Reads a time-varying control drum angle (deg) from a MAT-file table.
         - Produces angleDeg(t) using linear interpolation.

      2) Models.HPMicroPK (model)
         - A lumped point-kinetics reactor model with:
             * 6 delayed neutron precursor groups
             * 4-node thermal network (fuel, moderator/graphite, heat-pipe, N2)
             * drum reactivity + temperature feedback

      3) Experiments.RunOneProfile (model)
         - Wires (1) -> (2) so the drum angle profile drives the reactor.
         - Sets an experiment stop time and tolerance.

    Notes:
    - Modelica is declarative: you write equations, not a step-by-step program.
    - der(x) means dx/dt (time derivative). The solver integrates these ODEs.
  */
end MicroreactorPK;
