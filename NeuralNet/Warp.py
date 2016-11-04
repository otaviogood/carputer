from PIL import Image, ImageDraw
import numpy as np
import random

# from http://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def Transform(source):
    dx = random.randint(-128, 128)
    dy = random.randint(-64, 64)
    w = source.width
    h = source.height
    rs = np.random.randn(16)
    s = 0.1
    coeffs = find_coeffs(
        [(w*s*rs[0], h*s*rs[1]),
         (w + w*s*rs[2], h*s*rs[3]),
         (w + w*s*rs[4], h + h*s*rs[5]),
         (w*s*rs[6], h + h*s*rs[7])],

        [(0, 0),
         (w, 0),
         (w, h),
         (0, h)]
    )
    target = source.transform((source.width, source.height), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    return target

def RandRects(source):
    w = source.width
    h = source.height
    draw = ImageDraw.Draw(source)
    for i in xrange(12):
        rs = np.random.rand(4)
        rc = np.random.randn(3) * 0.3
        rc = np.clip(rc, -1.0, 1.0) * 127 + 127
        draw.rectangle([(rs[0]*1.4-0.2)*w, (rs[1]*1.4-0.2)*h, (rs[0]*1.4-0.2)*w + rs[2]*40, (rs[1]*1.4-0.2)*h + rs[3]*40], fill=(int(rc[0]), int(rc[1]), int(rc[2])))
    draw.rectangle([0, 80, 128, 128], fill=(0, 32, 0))


def WhiteUnbalance(source):
    # Adjust white balance.
    min_channel_high_end = 0.25
    max_channel_low_end = 0.25
    rmin = random.random()*min_channel_high_end
    gmin = random.random()*min_channel_high_end
    bmin = random.random()*min_channel_high_end
    rmax = random.random()*max_channel_low_end + 1 - max_channel_low_end
    gmax = random.random()*max_channel_low_end + 1 - max_channel_low_end
    bmax = random.random()*max_channel_low_end + 1 - max_channel_low_end
    new_image = np.empty((source.height, source.width, 3), dtype=np.float32)
    image = np.multiply(np.array(source), 1/255.)
    new_image[:, :, 0] = np.add(np.multiply(image[:, :, 0], (rmax-rmin)), rmin)
    new_image[:, :, 1] = np.add(np.multiply(image[:, :, 1], (gmax-gmin)), gmin)
    new_image[:, :, 2] = np.add(np.multiply(image[:, :, 2], (bmax-bmin)), bmin)
    new_image = np.multiply(new_image, 255)
    image = Image.fromarray(np.uint8(new_image))
    return image
