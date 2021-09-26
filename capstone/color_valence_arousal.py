# %%
import math
import numpy as np
from matplotlib import pyplot as plt
import colorsys
from PIL import Image
import pickle


def makeColorCube():
    # full 8-bit RGB space
    bits = 8
    cube_dimension = 2 ** bits

    full_rgb_space = np.array(
        [
            np.array([i, j, k], dtype=np.float16)
            for k in range(cube_dimension)
            for j in range(cube_dimension)
            for i in range(cube_dimension)
        ]
    )

    return full_rgb_space


def convertToXYZ(cube):
    # RGB/RYB follows XZY convention, swapping G & B/Y & B to follow XYZ convention
    for i in range(len(cube)):
        g = cube[i][1]
        b = cube[i][2]
        cube[i][1] = b
        cube[i][2] = g

    return cube


def cubeToSphere(cube):
    for i, color in enumerate(cube):
        # for i in cube:

        x = color[0]
        y = color[1]
        z = color[2]

        x_sphere = x * math.sqrt(1 - y * y * 0.5 - z * z * 0.5 + y * y * z * z / 3)
        y_sphere = y * math.sqrt(1 - z * z * 0.5 - x * x * 0.5 + z * z * x * x / 3)
        z_sphere = z * math.sqrt(1 - x * x * 0.5 - y * y * 0.5 + x * x * y * y / 3)

        cube[i] = (x_sphere, y_sphere, z_sphere)

    return cube


def rotateX3D(cube, theta):
    rad = theta * np.pi / 180
    sinTheta = np.sin(rad)
    cosTheta = np.cos(rad)
    for i in range(len(cube)):
        y = cube[i][1]
        z = cube[i][2]
        cube[i][1] = y * cosTheta - z * sinTheta
        cube[i][2] = z * cosTheta + y * sinTheta

    return cube


def rotateY3D(cube, theta):
    rad = theta * np.pi / 180
    sinTheta = np.sin(rad)
    cosTheta = np.cos(rad)
    for i in range(len(cube)):
        z = cube[i][2]
        x = cube[i][0]
        cube[i][2] = z * cosTheta - x * sinTheta
        cube[i][0] = x * cosTheta + z * sinTheta

    return cube


def rotateZ3D(cube, theta):
    rad = theta * np.pi / 180
    sinTheta = np.sin(rad)
    cosTheta = np.cos(rad)
    for i in range(len(cube)):
        x = cube[i][0]
        y = cube[i][1]
        cube[i][0] = x * cosTheta - y * sinTheta
        cube[i][1] = y * cosTheta + x * sinTheta

    return cube


def rotateZ2D(point, theta, origin=(0, 0)):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    Pass negative angle for clockwise.

    """
    rad = theta * np.pi / 180
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(rad) * (px - ox) - math.sin(rad) * (py - oy)
    qy = oy + math.sin(rad) * (px - ox) + math.cos(rad) * (py - oy)
    return qx, qy


def _cubic(t, a, b):
    weight = t * t * (3 - 2 * t)
    return a + weight * (b - a)


def rybToRgbColor(color):
    # Assumption: r, y, b in [0, 1] scale
    # tri-linear interpolation

    r, y, b = color[0], color[1], color[2]

    # red
    x0, x1 = _cubic(b, 1.0, 0.163), _cubic(b, 1.0, 0.0)
    x2, x3 = _cubic(b, 1.0, 0.5), _cubic(b, 1.0, 0.2)
    y0, y1 = _cubic(y, x0, x1), _cubic(y, x2, x3)
    red = _cubic(r, y0, y1)

    # green
    x0, x1 = _cubic(b, 1.0, 0.373), _cubic(b, 1.0, 0.66)
    x2, x3 = _cubic(b, 0.0, 0.0), _cubic(b, 0.5, 0.094)
    y0, y1 = _cubic(y, x0, x1), _cubic(y, x2, x3)
    green = _cubic(r, y0, y1)

    # blue
    x0, x1 = _cubic(b, 1.0, 0.6), _cubic(b, 0.0, 0.2)
    x2, x3 = _cubic(b, 0.0, 0.5), _cubic(b, 0.0, 0.0)
    y0, y1 = _cubic(y, x0, x1), _cubic(y, x2, x3)
    blue = _cubic(r, y0, y1)

    return (red, green, blue)


def backToCenter(cube):
    # moves the cube so that the last color,
    # which should be originally white (255,255,255),
    # to (0,0,0) after a variety of transformations
    resets = cube[-1] * np.sign(cube[-1])

    for i, color in enumerate(cube):
        cube[i] = color + resets

    return cube


def dropLastCol(cube):
    return np.delete(cube, np.s_[-1:], 1)


def cubeMinMax(cube):
    # flat_list = {item for sublist in cube for item in sublist}
    return np.min(cube), np.max(cube)


def linearScaleColor(color, min, max, lower=-1, upper=1):
    x = np.interp(color[0], (min, max), (lower, upper))
    y = np.interp(color[1], (min, max), (lower, upper))
    z = np.interp(color[2], (min, max), (lower, upper))

    return (int(round(x)), int(round(y)), int(round(z)))


def convertToCircumplexXY(x_square, y_square):
    x_circle = x_square * math.sqrt(1 - 0.5 * y_square ** 2)
    y_circle = y_square * math.sqrt(1 - 0.5 * x_square ** 2)
    return x_circle, y_circle


def convertToSquareXY(u, v):
    x_square = 0.5 * math.sqrt(
        abs(2 + u ** 2 - v ** 2 + 2 * u * math.sqrt(2))
    ) - 0.5 * math.sqrt(abs(2 + u ** 2 - v ** 2 - 2 * u * math.sqrt(2)))
    y_square = 0.5 * math.sqrt(
        abs(2 - u ** 2 + v ** 2 + 2 * v * math.sqrt(2))
    ) - 0.5 * math.sqrt(abs(2 - u ** 2 + v ** 2 - 2 * v * math.sqrt(2)))
    return x_square, y_square


def plotData(data):
    matrix = np.array(data)
    f = plt.figure(figsize=(10, 3))
    f.set_figheight(7)
    f.set_figwidth(7)
    x, y = matrix.T
    x = [round(i, 3) for i in x]
    y = [round(i, 3) for i in y]
    plt.scatter(x, y)
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.axhline(y=0, color="black", linestyle="dashed", linewidth=0.5)
    plt.axvline(x=0, color="black", linestyle="dashed", linewidth=0.5)
    for i_x, i_y in zip(x, y):
        plt.text(i_x, i_y, "({}, {})".format(i_x, i_y))
    plt.show()


def distanceOrigin(square):
    distances = set()
    for i in square:
        distance = math.sqrt(((0 - i[0]) ** 2) + ((0 - i[1]) ** 2))
        if distance > 0:
            distances.add(distance)
    return distances


def flipVert(square):
    return [[i[0], -i[1]] for i in square]


def rotate90Clockwise(square):
    return [[i[1], -i[0]] for i in square]


def flipHorz(square):
    return [[-i[0], i[1]] for i in square]


def flipHorzXY(x, y):
    return -x, y


# %%
# RGB cube corners
# we can use just the corners of the cube to get rotation angles and min/max
# values for scaling, instead of processing the whole cube since
# there shouldn't be anything outside the corners of the cube

blue = np.array([0, 0, 255], dtype=np.float16)  # RGB Corner/RYB Corner
green = np.array([0, 255, 0], dtype=np.float16)  # RGB Corner/RYB Corner
yellow = np.array([255, 255, 0], dtype=np.float16)  # RGB Corner/RYB Corner
red = np.array([255, 0, 0], dtype=np.float16)  # RGB Corner/RYB Corner
orange = np.array([255, 128, 0], dtype=np.float16)  # RYB Corner
purple = np.array([128, 0, 255], dtype=np.float16)  # RYB Corner
white = np.array([255, 255, 255], dtype=np.float16)  # RGB Corner/RYB Corner
black = np.array([0, 0, 0], dtype=np.float16)  # RGB Corner/RYB Corner

cyan = np.array([0, 255, 255], dtype=np.float16)  # RGB Corner, not useful
magenta = np.array([255, 0, 255], dtype=np.float16)  # RGB Corner, not useful

# rgb_cube_corners = np.array([blue, green, yellow, orange, red, purple, white, black, cyan, magenta])
rgb_cube_corners = np.array([red, green, blue])


###############################
# NEW ORDER:
# Convert to RYB
# Convert to HSV - image size 360x360
# Get X,Y
# Convert to Square
# scale -1,1


# %%
# create a Python image canvas to map HSV colors to
im = Image.new("RGB", (4096, 4096))
radius = min(im.size) / 2.0
cx, cy = im.size[0] / 2, im.size[1] / 2
pix = im.load()


# %%
# setting some parameters for scaling
rx_max = 0 - cx
ry_max = 0 - cy
s_max = (rx_max ** 2.0 + ry_max ** 2.0) ** 0.5 / radius


# %%
# create an RGB > RYB mapping
for x in range(im.width):
    for y in range(im.height):
        rx = x - cx
        ry = y - cy
        s = (rx ** 2.0 + ry ** 2.0) ** 0.5 / radius

        # scale the saturation value
        s = np.interp(s, (0, s_max - 1), (0, 1))

        # set the hue
        h = ((math.atan2(ry, rx) / math.pi) + 1.0) / 2.0

        # rotate hue 300 degrees counterclockwise to match Stahl
        n = 300.0
        hue = math.fmod(h + n / 360.0, 1.0)

        # convert to RYB subtractive color from  HSV color space to match Itten
        # HSV to RGB requires 0-1, linearScaleColor to scale back to RGB 0-255
        rgb = linearScaleColor(
            rybToRgbColor(colorsys.hsv_to_rgb(hue, s, 1.0)), 0, 1, 0, 255
        )

        pix[x, y] = rgb

im.show()

# %%
# scale x,y and map to valence, arousal
# use RGB as keys to get (valence, arousal)
valence_arousal_dict = {}
for x in range(im.width):
    for y in range(im.height):
        color = im.getpixel((x, y))

        valence = np.interp(x, (0, im.width - 1), (-1, 1))
        arousal = np.interp(y, (0, im.height - 1), (-1, 1))

        # set RGB value as key to get (valence,arousal)
        valence_arousal_dict[color] = (valence, arousal)


# %%
# # convert to circumplex, only if needed
# round_im = Image.new("RGB", (im.width, im.height))
# round_pix = round_im.load()

# for x in range(round_im.width):
#     for y in range(round_im.height):
#         color = im.getpixel((x, y))

#         x_scale_down = np.interp(x, (0, round_im.width - 1), (-1, 1))
#         y_scale_down = np.interp(y, (0, round_im.height - 1), (-1, 1))

#         x_circle, y_circle = convertToCircumplexXY(x_scale_down, y_scale_down)

#         x_scale_up = np.interp(x_circle, (-1, 1), (0, round_im.width - 1))
#         y_scale_up = np.interp(y_circle, (-1, 1), (0, round_im.height - 1))

#         round_pix[x_scale_up, y_scale_up] = color

# round_im.show()

# %%
# serialize it to a pickle
path = "/home/bensonnd/.msds/src/msds/"
file = "color-valence-arousal.pkl"

with open(path + file, "wb") as handle:
    pickle.dump(valence_arousal_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
