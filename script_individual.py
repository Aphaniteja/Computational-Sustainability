import requests
from sys import argv
from collections import namedtuple
import time
import shutil
from PIL import Image

#headers = {'Accept': 'application/json'}
#Directly get movies of sci-fi and above year 2000.
			 
for i in range(0,100):
	lat = 33.824504+(float(i)/100)
	if lat < 33.915881:
		for j in range(0,100):
			longi = -84.413581-(float(j)/100)
			latlong = str(lat) +","+ str(longi)
			print latlong
			if longi > -84.487346:
				params = {'key': 'AIzaSyBEo1AywD3DW6SvmxYCzmDKMnOOlqb0N4A', 'center':latlong, 'zoom':18, 'size':'640x640', 'format':'png','maptype':'satellite'}
				time.sleep(0.5)
				img = "30339/"+str(i)+str(j)+".png"
				response = requests.get('https://maps.googleapis.com/maps/api/staticmap?',params=params, timeout=7, stream=True)
				with open(img, 'wb') as out_file:
					shutil.copyfileobj(response.raw, out_file)
				del response
