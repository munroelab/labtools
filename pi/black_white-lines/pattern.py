"""
Make set of jpg images of resolution 1920, 1080 of black and white lines

files are lines_n.jpg where n is the number of pixels thick each black or white
line
"""
import Image
import numpy

def line_pattern(n):
    img = Image.new('L', (1920,1080))

    pattern = numpy.array(img)
    for i in range(n):
        pattern[i::2*n, :] = 255

    filename = "lines_%d.jpg" % n

    img = Image.fromarray(pattern)
    img.save(filename)


def main():

    for n in range(16):
        line_pattern(n)

    line_pattern(1080)


if __name__ == "__main__":
    main()
