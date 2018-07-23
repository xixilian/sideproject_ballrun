import math
from PIL import Image, ImageDraw


def makeRectangle(l, w, theta, offset=(0,0)):
    c, s = math.cos(theta), math.sin(theta)
    rectCoords = [(l/2.0, w/2.0), (l/2.0, -w/2.0), (-l/2.0, -w/2.0), (-l/2.0, w/2.0)]
    return [(c*x-s*y+offset[0], s*x+c*y+offset[1]) for (x,y) in rectCoords]


#L=110; W=110
#image = Image.new("1", (L, W))
#draw = ImageDraw.Draw(image)

#vertices = makeRectangle(54, 8.5, 0.5, offset=(L/2, W/2))
#draw.polygon(vertices, fill=1)

#image.save("test.png")
