
from numpy import linspace, zeros, sqrt, exp
from ..ops import OperatorsOrder2
import sys

class Maxwell:
    """ The base class for evolution of Maxwell's equations.
    """

    def __init__(self, n_grid, x_out, filename_stem, n_vars):
        """Constructor sets up coordinates, memory for variables, and
        opens output file.
        """

        print(" Initializing class Maxwell with n_grid = ", n_grid,
              ", x_out = ", x_out)
        self.n_grid = n_grid
        self.filename_stem = filename_stem
        self.n_vars = n_vars
        self.delta = float(x_out) / (n_grid - 2.0)
        delta = self.delta
        print("    grid spacing delta =",delta)

        # set up cell-centered grid on interval (0, x_out) in each
        # dimension, but pad grid with one ghost zone.  Will use symmetry
        # on "inner" boundaries, and out-going wave conditions on outer
        # boundaries.
        self.x = linspace(-delta / 2.0, x_out + delta / 2.0, n_grid)
        self.y = linspace(-delta / 2.0, x_out + delta / 2.0, n_grid)
        self.z = linspace(-delta / 2.0, x_out + delta / 2.0, n_grid)
        self.r = zeros((n_grid, n_grid, n_grid))

        for i in range(0, n_grid):
            for j in range(0, n_grid):
                for k in range(0, n_grid):
                    self.r[i, j, k] = sqrt(
                        self.x[i] ** 2 + self.y[j] ** 2 + self.z[k] ** 2
                    )

        # set up derivative operators
        self.ops = OperatorsOrder2(n_grid, delta)

        # set up all variables common to both approaches
        self.E_x = zeros((n_grid, n_grid, n_grid))
        self.E_y = zeros((n_grid, n_grid, n_grid))
        self.E_z = zeros((n_grid, n_grid, n_grid))
        self.A_x = zeros((n_grid, n_grid, n_grid))
        self.A_y = zeros((n_grid, n_grid, n_grid))
        self.A_z = zeros((n_grid, n_grid, n_grid))
        self.phi = zeros((n_grid, n_grid, n_grid))
        self.constraint = zeros((n_grid, n_grid, n_grid))

        # open file that records constraint violations
        filename = self.filename_stem + "_constraints.data"
        self.constraint_file = open(filename, "w")
        if self.constraint_file:
            self.constraint_file.write(
                "# Constraint violations and errors with n = %d\n" % (n_grid)
                )
            self.constraint_file.write("# t          C_norm\n")
            self.constraint_file.write("#=========================\n")
        else:
            print(" Could not open file",filename," for output")
            print(" Check permissions?")
            sys.exit(2)
        # keep track of time
        self.t = 0.0


    def __del__(self):
        """Close output file in destructor."""
        if self.constraint_file:
            self.constraint_file.close()


    def initialize(self):
        """Set up initial data for E^i (assuming A^i = 0 initially);
        see (B.55).
        """

        for i in range(0, self.n_grid):
            for j in range(0, self.n_grid):
                for k in range(0, self.n_grid):
                    rl = self.r[i, j, k]
                    costheta = self.z[k] / rl
                    sintheta = sqrt(1.0 - costheta ** 2)
                    rho = sqrt(self.x[i] ** 2 + self.y[j] ** 2)
                    cosphi = self.x[i] / rho
                    sinphi = self.y[j] / rho
                    E_phi = -8.0 * rl * sintheta * exp(- rl ** 2)
                    self.E_x[i, j, k] = -E_phi * sinphi
                    self.E_y[i, j, k] = E_phi * cosphi
                    self.E_z[i, j, k] = 0.0

        self.check_constraint()


    def check_constraint(self):
        """Check constraint violation."""

        self.constraint = self.ops.divergence(self.E_x, self.E_y, self.E_z)
        norm_c = 0.0

        for i in range(1, self.n_grid - 1):
            for j in range(1, self.n_grid - 1):
                for k in range(1, self.n_grid - 1):
                    norm_c += self.constraint[i, j, k] ** 2

        norm_c = sqrt(norm_c * self.delta ** 3)
        self.constraint_file.write("%e %e \n" % (self.t, norm_c))
        print(" Constraint violation at time {0:.3f} : {1:.4e}".
                  format(self.t, norm_c))
        return norm_c


    def integrate(self, courant, t_max, t_check):
        """Carry out time integration."""

        print(" Integrating to time",t_max,"with Courant factor",courant)
        print("    checking results after times", t_check)
        self.write_data()
        while self.t + t_check <= t_max:
            fields = self.wrap_up_fields()
            # call stepper to intergrate from current time t to t + t_const
            self.t, fields = self.stepper(fields, courant, t_check)
            self.unwrap_fields(fields)
            self.check_constraint()
            self.write_data()


    def stepper(self, fields, courant, t_const):
        """Stepper for Runge-Kutta integrator, integrates
        from current time t to t + t_const by repeatedly calling icn.
        """
        delta_t = courant * self.delta
        time = self.t
        t_fin = time + t_const

        while time < t_fin:
            if t_fin - time < delta_t:
                delta_t = t_fin - time
            # call icn to carry out one time step
            fields = self.icn(fields, delta_t)
            time += delta_t

        return time, fields


    def icn(self, fields, dt):
        """Carry out one 2nd-order iterative Crank-Nicholson step;
        see (B.47).
        The routine derivatives(fields) is defined in the derived classes
        Original and Reformulated, and provides time derivatives of the fields
        according to the two different versions of Maxwell's equations.
        The routine update_fields(fields, fields_dot, factor, dt) adds
        factor * dt * fields_dot to fields.
        """

        # Step 1: get derivs at t0 and update fields with half-step
        fields_dot = self.derivatives(fields)
        new_fields = self.update_fields(fields, fields_dot, 0.5, dt)

        # Step 2: get derivs at t = t0 + dt and update fields temporarily
        fields_temp = self.update_fields(fields, fields_dot, 1.0, dt)
        fields_dot = self.derivatives(fields_temp)

        # Step 3: do it again...  (iterative step)
        fields_temp = self.update_fields(fields, fields_dot, 1.0, dt)
        fields_dot = self.derivatives(fields_temp)

        # Finally: update new_fields with second half-step
        new_fields = self.update_fields(new_fields, fields_dot, 0.5, dt)
        #

        return new_fields


    def update_fields(self, fields, fields_dot, factor, dt):
        """Updates all fields in the list fields,
        by adding factor * fields_dot * dt.
        """

        new_fields = []
        for var in range(0, self.n_vars):
            field = fields[var]
            field_dot = fields_dot[var]
            new_field = field + factor * field_dot * dt
            new_fields.append(new_field)

        return new_fields

    def outgoing_wave(self, f_dot, f):
        """Computes time derivatives of fields from
        outgoing-wave boundary condition;
        see (B.52)
        """
        n_grid = self.n_grid

        i = n_grid - 1  # upper x-boundary
        for j in range(0, n_grid - 1):
            for k in range(0, n_grid - 1):
                f_x, f_y, f_z = self.ops.partial_derivs(f, i, j, k)
                f_dot[i, j, k] = (-(f[i, j, k] + self.x[i] * f_x +
                                    self.y[j] * f_y + self.z[k] * f_z)
                                  / self.r[i, j, k])

        j = n_grid - 1  # upper y-boundary
        for i in range(0, n_grid):
            for k in range(0, n_grid - 1):
                f_x, f_y, f_z = self.ops.partial_derivs(f, i, j, k)
                f_dot[i, j, k] = (-(f[i, j, k] + self.x[i] * f_x +
                                    self.y[j] * f_y + self.z[k] * f_z)
                                  / self.r[i, j, k])

        k = n_grid - 1  # upper z-boundary
        for i in range(0, n_grid):
            for j in range(0, n_grid):
                f_x, f_y, f_z = self.ops.partial_derivs(f, i, j, k)
                f_dot[i, j, k] = (-(f[i, j, k] + self.x[i] * f_x +
                                    self.y[j] *  f_y + self.z[k] * f_z)
                                  / self.r[i, j, k])


    def symmetry(self, f_dot, x_sym, y_sym, z_sym):
        """Computes time derivatives on inner boundaries from symmetry,"""

        n_grid = self.n_grid

        i = 0  # lower x-boundary
        for j in range(1, n_grid - 1):
            for k in range(1, n_grid - 1):
                f_dot[i, j, k] = x_sym * f_dot[i + 1, j, k]

        j = 0  # lower y-boundary
        for i in range(0, n_grid - 1):
            for k in range(1, n_grid - 1):
                f_dot[i, j, k] = y_sym * f_dot[i, j + 1, k]

        k = 0  # lower z-boundary
        for i in range(0, n_grid - 1):
            for j in range(0, n_grid - 1):
                f_dot[i, j, k] = z_sym * f_dot[i, j, k + 1]


    def write_data(self):
        """Write data to file."""

        HEADER1 = "# x           y             E_x            E_y           "
        HEADER2 = "A_x           A_y           phi            C\n"
        WRITER = "%e  %e  %e  %e  %e  %e  %e  %e\n"

        filename = self.filename_stem + "_fields_{0:.3f}.data".format(self.t)
        out = open(filename, "w")
        if out:
            k = 1
            out.write("# data at time {0:.3f} at z = {1:e} \n".
                          format(self.t, self.z[k]))
            out.write(HEADER1)
            out.write(HEADER2)
            n_grid = self.n_grid

            for i in range(1, n_grid):
                for j in range(1, n_grid):
                    out.write(WRITER % (self.x[i], self.y[j],
                                        self.E_x[i, j, k], self.E_y[i, j, k],
                                        self.A_x[i, j, k], self.A_y[i, j, k],
                                        self.phi[i, j, k],
                                        self.constraint[i, j, k],))
                out.write("\n")
            out.close()
        else:
            print(" Could not open file", filename,"in write_data()")
            print(" Check permissions?")
        return
