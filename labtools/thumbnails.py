from PIL import Image
import labdb
import os

db = labdb.LabDB()
rows = db.execute("SELECT video_id, path FROM video")


for row in rows:
    video_id, path= row
    print video_id, path

    image = os.path.join(path, 'frame00000.png')
    if not os.path.exists(image):
        image = os.path.join(path, 'frame00000.pgm')

    filename = "thumbnails/video_%d.png" % video_id
    im = Image.open(image)
    #im.thumbnail(size, Image.ANTIALIAS)
    im.save(filename)

    print image



