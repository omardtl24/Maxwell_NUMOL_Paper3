from numpy import zeros


class OperatorsOrder2:
    """ This class contains all of the derivative operators that we will
    ever need. The operators are implemented to the second order.
    """

    def __init__(self, n_grid, delta):
        """Constructor for class OperatorsOrder2."""

        print(" Setting up 2nd-order derivative operators ")
        self.n_grid = n_grid
        self.delta = delta


    def laplace(self, fct):
        """Compute Laplace operator of function fct in interior of grid."""

        n_grid = self.n_grid
        ood2 = 1.0 / self.delta ** 2
        lap = zeros((n_grid, n_grid, n_grid))

        for i in range(1, n_grid - 1):
            for j in range(1, n_grid - 1):
                for k in range(1, n_grid - 1):
                    # see (B.26)
                    dfddx = (fct[i + 1, j, k] - 2.0 * fct[i, j, k] +
                             fct[i - 1, j, k])
                    dfddy = (fct[i, j + 1, k] - 2.0 * fct[i, j, k] +
                             fct[i, j - 1, k])
                    dfddz = (fct[i, j, k + 1] - 2.0 * fct[i, j, k] +
                             fct[i, j, k - 1])
                    lap[i, j, k] = ood2 * (dfddx + dfddy + dfddz)

        return lap

    def gradient(self, fct):
        """Compute gradient of function fct in interior of grid."""

        n_grid = self.n_grid
        oo2d = 0.5 / self.delta
        grad_x = zeros((n_grid, n_grid, n_grid))
        grad_y = zeros((n_grid, n_grid, n_grid))
        grad_z = zeros((n_grid, n_grid, n_grid))

        for i in range(1, n_grid - 1):
            for j in range(1, n_grid - 1):
                for k in range(1, n_grid - 1):
                    # see (A.12)
                    grad_x[i, j, k] = oo2d * (fct[i + 1, j, k] -
                                              fct[i - 1, j, k])
                    grad_y[i, j, k] = oo2d * (fct[i, j + 1, k] -
                                              fct[i, j - 1, k])
                    grad_z[i, j, k] = oo2d * (fct[i, j, k + 1] -
                                              fct[i, j, k - 1])

        return grad_x, grad_y, grad_z


    def divergence(self, v_x, v_y, v_z):
        """Compute divergence of vector v in interior of grid."""

        n_grid = self.n_grid
        oo2d = 0.5 / self.delta
        div = zeros((n_grid, n_grid, n_grid))

        for i in range(1, n_grid - 1):
            for j in range(1, n_grid - 1):
                for k in range(1, n_grid - 1):
                    # see (B.12)
                    v_xx = v_x[i + 1, j, k] - v_x[i - 1, j, k]
                    v_yy = v_y[i, j + 1, k] - v_y[i, j - 1, k]
                    v_zz = v_z[i, j, k + 1] - v_z[i, j, k - 1]
                    div[i, j, k] = oo2d * (v_xx + v_yy + v_zz)

        return div


    def grad_div(self, A_x, A_y, A_z):
        """Compute gradient of divergence in interior of grid."""

        n_grid = self.n_grid
        ood2 = 1.0 / self.delta ** 2
        grad_div_x = zeros((n_grid, n_grid, n_grid))
        grad_div_y = zeros((n_grid, n_grid, n_grid))
        grad_div_z = zeros((n_grid, n_grid, n_grid))

        for i in range(1, n_grid - 1):
            for j in range(1, n_grid - 1):
                for k in range(1, n_grid - 1):
                    # xx_A_x denotes \partial_x \partial_x A_x, etc...
                    xx_A_x = (A_x[i + 1, j, k] - 2.0 * A_x[i, j, k] +
                              A_x[i - 1, j, k])
                    xy_A_y = (A_y[i + 1, j + 1, k] - A_y[i + 1, j - 1, k] -
                              A_y[i - 1, j + 1, k] + A_y[i - 1, j - 1, k])
                    xz_A_z = (A_z[i + 1, j, k + 1] - A_z[i + 1, j, k - 1] -
                              A_z[i - 1, j, k + 1] + A_z[i - 1, j, k - 1])
                    grad_div_x[i, j, k] = ood2 * (xx_A_x + 0.25 * xy_A_y +
                                                  0.25 * xz_A_z)
                    yx_A_x = (A_x[i + 1, j + 1, k] - A_x[i + 1, j - 1, k] -
                              A_x[i - 1, j + 1, k] + A_x[i - 1, j - 1, k])
                    yy_A_y = (A_y[i, j + 1, k] - 2.0 * A_y[i, j, k] +
                              A_y[i, j - 1, k])
                    yz_A_z = (A_z[i, j + 1, k + 1] - A_z[i, j + 1, k - 1] -
                              A_z[i, j - 1, k + 1] + A_z[i, j - 1, k - 1])
                    grad_div_y[i, j, k] = ood2 * (0.25 * yx_A_x + yy_A_y +
                                                  0.25 * yz_A_z)
                    zx_A_x = (A_x[i + 1, j, k + 1] - A_x[i + 1, j, k - 1] -
                              A_x[i - 1, j, k + 1] + A_x[i - 1, j, k - 1])
                    zy_A_y = (A_y[i, j + 1, k + 1] - A_y[i, j + 1, k - 1] -
                              A_y[i, j - 1, k + 1] + A_y[i, j - 1, k - 1])
                    zz_A_z = (A_z[i, j, k + 1] - 2.0 * A_z[i, j, k] +
                              A_z[i, j, k - 1])
                    grad_div_z[i, j, k] = ood2 * (0.25 * zx_A_x + 0.25 *
                                                  zy_A_y + zz_A_z)

        return grad_div_x, grad_div_y, grad_div_z

    def partial_derivs(self, fct, i, j, k):
        """Compute partial derivatives. Unlike all
        other operators in this class, this function returns derivatives
        at one grid point (i,j,k) only, but including on the boundaries."""

        oo2d = 0.5 / self.delta

        # use (A.10) in interior, and (A.5) on boundaries
        # partial f / partial x
        if i == 0:
            f_x = oo2d * (-3.0 * fct[i, j, k] + 4.0 * fct[i + 1, j, k] -
                          fct[i + 2, j, k])
        elif i == self.n_grid - 1:
            f_x = oo2d * (3.0 * fct[i, j, k] - 4.0 * fct[i - 1, j, k] +
                          fct[i - 2, j, k])
        else:
            f_x = oo2d * (fct[i + 1, j, k] - fct[i - 1, j, k])

        # partial f / partial y
        if j == 0:
            f_y = oo2d * (-3.0 * fct[i, j, k] + 4.0 * fct[i, j + 1, k] -
                          fct[i, j + 2, k])
        elif j == self.n_grid - 1:
            f_y = oo2d * (3.0 * fct[i, j, k] - 4.0 * fct[i, j - 1, k] +
                          fct[i, j - 2, k])
        else:
            f_y = oo2d * (fct[i, j + 1, k] - fct[i, j - 1, k])

        # partial f / partial z
        if k == 0:
            f_z = oo2d * (-3.0 * fct[i, j, k] + 4.0 * fct[i, j, k + 1] -
                          fct[i, j, k + 2])
        elif k == self.n_grid - 1:
            f_z = oo2d * (3.0 * fct[i, j, k] - 4.0 * fct[i, j, k - 1] +
                          fct[i, j, k - 2])
        else:
            f_z = oo2d * (fct[i, j, k + 1] - fct[i, j, k - 1])
        return f_x, f_y, f_z

