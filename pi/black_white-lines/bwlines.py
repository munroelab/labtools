import pylab

# Create A4 size sheet
pylab.figure(figsize = (30 / 2, 22 / 2.))
#pylab.figure(figsize= (51 / 2, 29 / 2))
#pylab.figure(figsize=(51/1.5, 30/1.5))
#pylab.figure(figsize=(29.7 / 2, 21 / 1.5))

# Make axes the full size of the page
pylab.axes([0,0,1,1])

# Create a grid which has unit dimensions of 1mm by 3mm
z = pylab.zeros((297 // 1.5, 210))
#z= pylab.zeros(( 290 // 2, 510 ))
#z=pylab.zeros((300 // 1.5, 510))
# Make every second row black
z[::2,:] = 255

pylab.imshow(z, cmap='gray', interpolation = 'nearest', aspect='auto')

# Save the image as a PDF document
pylab.savefig('bwlines14.png')

# Show for debug purposes
pylab.show()
