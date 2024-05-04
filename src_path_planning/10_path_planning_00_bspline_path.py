
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate


#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# BSpline Path
# -----------------------------------------------------------------------------------------------------

def approximate_b_spline_path(x: list,
                              y: list,
                              n_path_points: int,
                              degree: int = 3,
                              s=None,
                              ) -> tuple:
    """
    Approximate points with a B-Spline path

    Parameters
    ----------
    x : array_like
        x position list of approximated points
    y : array_like
        y position list of approximated points
    n_path_points : int
        number of path points
    degree : int, optional
        B Spline curve degree. Must be 2<= k <= 5. Default: 3.
    s : int, optional
        smoothing parameter. If this value is bigger, the path will be
        smoother, but it will be less accurate. If this value is smaller,
        the path will be more accurate, but it will be less smooth.
        When `s` is 0, it is equivalent to the interpolation. Default is None,
        in this case `s` will be `len(x)`.

    Returns
    -------
    x : array
        x positions of the result path
    y : array
        y positions of the result path
    heading : array
        heading of the result path
    curvature : array
        curvature of the result path

    """
    distances = _calc_distance_vector(x, y)

    spl_i_x = interpolate.UnivariateSpline(distances, x, k=degree, s=s)
    spl_i_y = interpolate.UnivariateSpline(distances, y, k=degree, s=s)

    sampled = np.linspace(0.0, distances[-1], n_path_points)
    return _evaluate_spline(sampled, spl_i_x, spl_i_y)


def interpolate_b_spline_path(x, y,
                              n_path_points: int,
                              degree: int = 3) -> tuple:
    """
    Interpolate x-y points with a B-Spline path

    Parameters
    ----------
    x : array_like
        x positions of interpolated points
    y : array_like
        y positions of interpolated points
    n_path_points : int
        number of path points
    degree : int, optional
        B-Spline degree. Must be 2<= k <= 5. Default: 3

    Returns
    -------
    x : array
        x positions of the result path
    y : array
        y positions of the result path
    heading : array
        heading of the result path
    curvature : array
        curvature of the result path

    """
    return approximate_b_spline_path(x, y, n_path_points, degree, s=0.0)


def _calc_distance_vector(x, y):
    dx, dy = np.diff(x), np.diff(y)
    distances = np.cumsum([np.hypot(idx, idy) for idx, idy in zip(dx, dy)])
    distances = np.concatenate(([0.0], distances))
    distances /= distances[-1]
    return distances


def _evaluate_spline(sampled, spl_i_x, spl_i_y):
    x = spl_i_x(sampled)
    y = spl_i_y(sampled)
    dx = spl_i_x.derivative(1)(sampled)
    dy = spl_i_y.derivative(1)(sampled)
    heading = np.arctan2(dy, dx)
    ddx = spl_i_x.derivative(2)(sampled)
    ddy = spl_i_y.derivative(2)(sampled)
    curvature = (ddy * dx - ddx * dy) / np.power(dx * dx + dy * dy, 2.0 / 3.0)
    return np.array(x), y, heading, curvature,


# -----------------------------------------------------------------------------------------------------
# plot curvature
# -----------------------------------------------------------------------------------------------------

def plot_curvature(x_list, y_list, heading_list, curvature,
                   k=0.01, c="-c", label="Curvature"):
    """
    Plot curvature on 2D path. This plot is a line from the original path,
    the lateral distance from the original path shows curvature magnitude.
    Left turning shows right side plot, right turning shows left side plot.
    For straight path, the curvature plot will be on the path, because
    curvature is 0 on the straight path.

    Parameters
    ----------
    x_list : array_like
        x position list of the path
    y_list : array_like
        y position list of the path
    heading_list : array_like
        heading list of the path
    curvature : array_like
        curvature list of the path
    k : float
        curvature scale factor to calculate distance from the original path
    c : string
        color of the plot
    label : string
        label of the plot
    """
    cx = [x + d * k * np.cos(yaw - np.pi / 2.0) for x, y, yaw, d in
          zip(x_list, y_list, heading_list, curvature)]
    cy = [y + d * k * np.sin(yaw - np.pi / 2.0) for x, y, yaw, d in
          zip(x_list, y_list, heading_list, curvature)]

    plt.plot(cx, cy, c, label=label)
    for ix, iy, icx, icy in zip(x_list, y_list, cx, cy):
        plt.plot([ix, icx], [iy, icy], c)



#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# BSpline Path
# -----------------------------------------------------------------------------------------------------

# way points
way_point_x = [-1.0, 3.0, 4.0, 2.0, 1.0]
way_point_y = [0.0, -3.0, 1.0, 1.0, 3.0]
n_course_point = 50  # sampling number

plt.subplots()
rax, ray, heading, curvature = approximate_b_spline_path(
    way_point_x, way_point_y, n_course_point, s=0.5)
plt.plot(rax, ray, '-r', label="Approximated B-Spline path")
plot_curvature(rax, ray, heading, curvature)

plt.title("B-Spline approximation")
plt.plot(way_point_x, way_point_y, '-og', label="way points")
plt.grid(True)
plt.legend()
plt.axis("equal")

plt.subplots()
rix, riy, heading, curvature = interpolate_b_spline_path(
    way_point_x, way_point_y, n_course_point)
plt.plot(rix, riy, '-b', label="Interpolated B-Spline path")
plot_curvature(rix, riy, heading, curvature)

plt.title("B-Spline interpolation")
plt.plot(way_point_x, way_point_y, '-og', label="way points")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.show()

