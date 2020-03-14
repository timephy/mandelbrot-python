from numba import njit
from PIL import Image
from numpy import complex, array
import colorsys
import time


def rgb_conv(i):
    """Convert iteration count to RGB color tuple."""
    if i == 0:
        return (0, 0, 0)
    color = 255 * array(colorsys.hsv_to_rgb(i / 255.0, 1.0, 1.0))
    return tuple(color.astype(int))


@njit
def mandelbrot_impl(c, z, max_iter):
    """The actual Mandelbrot function."""
    # keeping track of previous values is not worth it
    # njit improves performance by 8-10x
    for i in range(max_iter):
        # print(f"z_{i} = {z}")
        z = z ** 2 + c
        if abs(z) > 2:
            return i
    return 0


@njit
def mandelbrot_impl_perf(c, z, max_iter):
    """The actual Mandelbrot function."""
    # keeping track of previous values is not worth it
    # njit improves performance by 8-10x
    z = c  # this saves 8/12s
    for i in range(max_iter):
        # print(f"z_{i} = {z}")
        if abs(z) > 2:
            return i + 1  # +1 to fix black outter circle
        z = z ** 2 + c
    return 0


def mandelbrot(c, *, z=complex(0, 0), max_iter=1000):
    # since njit-ed function cannot have kwargs with default value
    return mandelbrot_impl_perf(c, z, max_iter)


def __main__():
    RESOLUTION = (1080, 1080)
    SCALE = 2
    POS = complex(0, 0)

    img = Image.new("RGB", RESOLUTION)
    pixels = img.load()

    smaller_side = min(RESOLUTION)
    dist_per_pixel = SCALE / (smaller_side / 2)

    # doing half of the calculation out of loop
    left = POS.real - img.width / 2 * dist_per_pixel
    top = POS.imag - img.height / 2 * dist_per_pixel

    time_before = time.time()
    for x in range(img.width):
        print("x", x)
        # real = POS.real + (x - img.width / 2) * dist_per_pixel
        real = left + x * dist_per_pixel
        for y in range(img.height):
            # imag = POS.imag + (y - img.height / 2) * dist_per_pixel
            imag = top + y * dist_per_pixel
            c = complex(real, imag)
            pixels[x, y] = rgb_conv(mandelbrot(c))
    print(time.time() - time_before)

    img.show(title="mandelbrot")


if __name__ == "__main__":
    __main__()
