
# coding: utf-8

# # Qualitative Schlieren

# Use the qt backend so that we can have animations

# In[1]:

import matplotlib
matplotlib.use('Qt4Agg')
#get_ipython().magic(u'matplotlib qt')


# Bring in some modules

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


# Images can be load with either pillow or PIL

# In[21]:

from PIL import Image


# In[22]:

expts = ['/Users/jmunroe/data/exp160622143528',
         '/Users/jmunroe/data/exp160622151225',
         '/Users/jmunroe/data/exp160622154207',
         '/Users/jmunroe/data/exp160623112912',
         '/Users/jmunroe/data/exp160623114719',
         '/Users/jmunroe/data/exp160623120410',
         ]
exp = expts[4]

# In[23]:

img = Image.open(exp + '/200.jpg')


# In[24]:

ref_image = np.array(img, dtype='int16')


# In[30]:

fig = plt.figure()
ax = plt.subplot(111)
im = plt.imshow(ref_image, vmin=-20, vmax=20, cmap=plt.get_cmap('jet'), animated=True)
ttl = plt.title('Title', animated=True)


# In[31]:

plt.colorbar()


# In[34]:

def updatefig(framenum, *args, **kwargs):
    global ref_image
    img2 = Image.open(exp + '/%d.jpg' %framenum)
    diff = ref_image - np.array(img2)
    im.set_array(diff)
    print ('Frame: %d, Time: %.1f' % (framenum, framenum / 10.0))
    return im, 


# In[35]:

ani = animation.FuncAnimation(fig, 
        updatefig, 
        frames=range(10,6000,10), 
        interval=0, blit=True)
plt.show()

# In[ ]:




# In[ ]:




# In[ ]:



