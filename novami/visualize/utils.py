import colorsys

import matplotlib.pyplot as plt
import numpy as np


def one_point_palette(encoding: str = 'hex'):
    """
    Terracotta.

    Parameters
    ----------
    encoding: str
        One of 'hex', 'rgb'
    """
    if encoding == 'hex':
        return ['#CB6040']
    elif encoding == 'rgb':
        return [[203, 96, 64]]
    else:
        raise ValueError(f"Invalid encoding: {encoding}")


def two_point_palette(encoding: str = 'hex'):
    """
    Teal, Terracotta.

    Parameters
    ----------
    encoding: str
        One of 'hex', 'rgb'
    """
    if encoding == 'hex':
        return ['#257180', '#CB6040']
    elif encoding == 'rgb':
        return [[37, 113, 128], [203, 96, 64]]
    else:
        raise ValueError(f"Invalid encoding: {encoding}")


def three_point_palette(encoding: str = 'hex'):
    """
    Teal, Cream, Terracotta.

    Parameters
    ----------
    encoding: str
        One of 'hex', 'rgb'
    """
    if encoding == 'hex':
        return ['#257180', '#F2E5BF', '#CB6040']
    elif encoding == 'rgb':
        return [[37, 113, 128], [242, 229, 191], [203, 96, 64]]
    else:
        raise ValueError(f"Invalid encoding: {encoding}")


def four_point_palette(encoding: str = 'hex'):
    """
    Teal, Cream, Faded Orange, Terracotta.

    Parameters
    ----------
    encoding: str
        One of 'hex', 'rgb'
    """
    if encoding == 'hex':
        return ['#257180', '#F2E5BF', '#FD8B51', '#CB6040']
    elif encoding == 'rgb':
        return [[37, 113, 128], [242, 229, 191], [253, 139, 81], [203, 96, 64]]
    else:
        raise ValueError(f"Invalid encoding: {encoding}")


def five_point_palette(encoding: str = 'hex'):
    """
    Teal, Cascade, Cream, Faded Orange, Terracotta.

    Parameters
    ----------
    encoding: str
        One of 'hex', 'rgb'
    """
    if encoding == 'hex':
        return ['#257180', '#8CABA0', '#F2E5BF', '#FD8B51', '#CB6040']
    elif encoding == 'rgb':
        return [[37, 113, 128], [140, 171, 160], [242, 229, 191], [253, 139, 81], [203, 96, 64]]
    else:
        raise ValueError(f"Invalid encoding: {encoding}")


def display_palette(palette: List[str]):
    """
    Display a palette as a rectangular box with vertical stripes.

    Parameters
    ----------
    palette: List[str]
        A list of HEX-encoded colours.
    """

    width = 8
    height = 5
    per_colour_width = width / len(palette)

    fig, ax = plt.subplots(figsize=(width, height))

    for i, colour in enumerate(palette):
        if not colour.startswith('#'):
            colour = '#' + colour

        rect = patches.Rectangle((i * per_colour_width, 0), per_colour_width, height, linewidth=0, facecolor=colour)
        ax.add_patch(rect)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.show()


def generate_color_variants(hex_color, num_variants=4, step: float = 0.25):
    """
    Generate brighter and darker variants of a given HEX color.

    Parameters
    ----------
    hex_color: str
        A base colour in HEX format to use
    num_variants: int
        Total number of NEW colours to generate
    step: float
        Step size to take on the colour wheel. Bigger step gives more different colours.

    Returns
    -------
    variants: List[str]
        Generated colours sorted from darkest to brightest

    Args:
        hex_color (str): The HEX colour code (with or without '#')
        num_variants (int): Total number of variants to generate (must be even)

    Returns:
        list: A list of color variants with original color in the middle
    """

    if num_variants % 2 != 0:
        num_variants += 1

    hex_color = hex_color.lstrip('#')

    # Convert hex to RGB
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0

    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    variants = []
    half_variants = num_variants // 2

    # Generate darker variants
    for i in range(half_variants, 0, -1):
        new_v = max(0.0, v * (1 - (i * step)))
        new_r, new_g, new_b = colorsys.hsv_to_rgb(h, s, new_v)

        hex_variant = "#{:02x}{:02x}{:02x}".format(
            int(new_r * 255),
            int(new_g * 255),
            int(new_b * 255)
        )
        variants.append(hex_variant)

    # Add original color
    variants.append(f"#{hex_color}")

    # Generate brighter variants
    for i in range(1, half_variants + 1):
        new_v = min(1.0, v * (1 + (i * step)))
        new_r, new_g, new_b = colorsys.hsv_to_rgb(h, s, new_v)

        hex_variant = "#{:02x}{:02x}{:02x}".format(
            int(new_r * 255),
            int(new_g * 255),
            int(new_b * 255)
        )
        variants.append(hex_variant)

    return variants


def set_font(font_filename: str = "arial.ttf"):
    """
    Register and set a custom TTF font globally for matplotlib.
    Works with both relative (from mpl data path) and absolute paths.
    """
    from pathlib import Path
    import matplotlib.pyplot as pl
    import matplotlib.font_manager as fm
    import matplotlib as mpl

    try:
        # Resolve font path
        font_path = (
            Path(mpl.get_data_path(), "fonts/ttf", font_filename)
            if not Path(font_filename).is_absolute()
            else Path(font_filename)
        )
        if not font_path.exists():
            raise FileNotFoundError(f"Font file not found at: {font_path}")

        fm.fontManager.addfont(str(font_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()
        pl.rcParams["font.family"] = font_name
        print(f"✔ Matplotlib is now using: {font_name}")

    except FileNotFoundError as e:
        print(f'{e}\nUsing default font instead.')
