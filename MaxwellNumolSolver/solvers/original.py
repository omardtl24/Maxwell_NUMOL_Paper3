from .maxwell import Maxwell

class Original(Maxwell):
    """ Derived class Original contains methods
    for the original version of Maxwell's equations.

    Note: in this version, we integrate n_vars = 7 independent variables.
    """

    def __init__(self, n_grid, x_out):
        """Constructor doesn't do much..."""

        print(" Integrating original Maxwell Equations ")
        print("    using n_grid =", n_grid, "\b^3 grid points with ")
        print("    outer boundary at x_out =", x_out)
        IND_VARS = 7
        filename_stem = "Max_orig_" + str(n_grid) + "_" + str(x_out)
        Maxwell.__init__(self, n_grid, x_out, filename_stem, IND_VARS)


    def wrap_up_fields(self):
        """Bundles up the seven variables into the list fields."""

        return (self.E_x, self.E_y, self.E_z, self.A_x, self.A_y,
                self.A_z, self.phi)


    def unwrap_fields(self, fields):
        """Extracts the independent variables from the list fields."""

        self.E_x = fields[0]
        self.E_y = fields[1]
        self.E_z = fields[2]
        self.A_x = fields[3]
        self.A_y = fields[4]
        self.A_z = fields[5]
        self.phi = fields[6]


    def derivatives(self, fields):
        """Computes the time derivative of the fields: Maxwell's equations."""

        self.unwrap_fields(fields)

        #
        # compute derivatives
        #
        Lap_A_x = self.ops.laplace(self.A_x)
        Lap_A_y = self.ops.laplace(self.A_y)
        Lap_A_z = self.ops.laplace(self.A_z)
        grad_x_phi, grad_y_phi, grad_z_phi = self.ops.gradient(self.phi)
        grad_div_x_A, grad_div_y_A, grad_div_z_A = self.ops.grad_div(
            self.A_x, self.A_y, self.A_z)
        DivA = self.ops.divergence(self.A_x, self.A_y, self.A_z)

        #
        # then compute time derivatives from original version
        # of Maxwell's equations,
        # see (4.1) and (4.2)
        #
        A_x_dot = -self.E_x - grad_x_phi
        A_y_dot = -self.E_y - grad_y_phi
        A_z_dot = -self.E_z - grad_z_phi
        E_x_dot = -Lap_A_x + grad_div_x_A
        E_y_dot = -Lap_A_y + grad_div_y_A
        E_z_dot = -Lap_A_z + grad_div_z_A
        phi_dot = -DivA

        #
        # finally fix inner boundaries from symmetry, see Table B.1...
        #
        self.symmetry(E_x_dot, 1, -1, 1)
        self.symmetry(E_y_dot, -1, 1, 1)
        self.symmetry(E_z_dot, 1, 1, -1)
        self.symmetry(A_x_dot, 1, -1, 1)
        self.symmetry(A_y_dot, -1, 1, 1)
        self.symmetry(A_z_dot, 1, 1, -1)
        self.symmetry(phi_dot, -1, -1, 1)

        #
        # ...and outer boundaries using outgoing-wave boundary condition (B.52)
        #
        self.outgoing_wave(E_x_dot, self.E_x)
        self.outgoing_wave(E_y_dot, self.E_y)
        self.outgoing_wave(E_z_dot, self.E_z)
        self.outgoing_wave(A_x_dot, self.A_x)
        self.outgoing_wave(A_y_dot, self.A_y)
        self.outgoing_wave(A_z_dot, self.A_z)
        self.outgoing_wave(phi_dot, self.phi)

        #
        return (E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot)

