import math
import numpy as np
import matplotlib
matplotlib.use('MacOSX')
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.image as mping
import pdb
import matplotlib.transforms as mtransforms
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import dialogs
#import photos iOS specific


"""set design parameters"""
alpha1 = 2 #input("Enter inital angle:")
""" DE BUG WITH 2 !!!!"""
length1 = 50# input("Enter inital horizontal length:")
radius = 250 #input("Enter radius:")
"""DE BUG WITH RADIUS 0"""
alpha2 = 15# input("Enter final angle:")
length2 = 25# input("Enter final horizontal length:")
groundx = [0,50,51,150,200,400]
groundy = [0,0,0,0,0,0]
ground = np.matrix([groundx,groundy])
head_offset = 3
a = .4 #TOI above gantry pivot
TOB = 1.0 # top of belt above ground @ tail pulley
b = 1.3 #depth of gantry

""" set lists """
gantry_list=[]
trestle_list=[]
llm_list = []
ebm_list=[]


""" helper functions"""
def rotation(theta,exist_point,origin):

	theta = np.radians(theta)
	vec = exist_point-origin
	r = np.array(( (np.cos(theta), -np.sin(theta)),
               (np.sin(theta),  np.cos(theta)) ))

	new_point = np.dot(vec,r)+origin
	return new_point

class gantry():
	def __init__(self,x2,y2,angle,length):
		# x1 and y1 are the pivot points of the lower end of the gantry
		#angle +'ve is anti-clockwise'
		self.x2 = x2
		self.y2 = y2
		self.angle = angle
		self.length = length
		if self.length ==2:
			self.take_up = True

		"""main attachment points"""
		self.x1 = x2 - length*math.cos(math.radians(angle))
		self.y1 = y2 - length*math.sin(math.radians(angle))

		"""truss"""
		self.truss = np.matrix([[i,-j*b] for i in range(length+1) for j in range(2)])
		theta = -math.radians(angle)
		c , s = math.cos(theta) , math.sin(theta)
		R = np.matrix([[c, -s],[s ,c]])
		self.truss_rotated = np.dot(self.truss,R) + np.matrix([self.x1,self.y1])

		"""idlers"""
		self.idler_coods = np.matrix([[1+(self.length-2)/4*(i),a] for i in range(5)])
		theta = -math.radians(angle)
		c , s = math.cos(theta) , math.sin(theta)
		R = np.matrix([[c, -s],[s ,c]])
		self.idler_coods_rotated = np.dot(self.idler_coods,R)+np.matrix([self.x1,self.y1])


class trestle():
	def __init__(self,x0,y0,y1):
		self.x0 = x0
		self.x1 = x0
		self.y0 = y0
		self.y1 = y1
		self.coodsx = [self.x0,self.x1]
		self.coodsy = [self.y0,self.y1]

def gradient(x):
	if x > Head_x:
		return 0
	else:
		return (belt(x)-belt(x-0.1))/(.1)

def ground(x):
	# a function to compute the y value of the ground for the given x
	return np.interp(x,groundx,groundy)

def belt(x):
	# a function to computer the y value of the belt at the given x
	if x < TP1_x:
		return x*math.tan(math.radians(alpha1))+TOB
	elif x < TP2_x:
		return CIR_y-math.sqrt(radius**2-(x-CIR_x)**2)
	elif x < Head_x and x > TP2_x:
		xcoods = [TP2_x,Head_x]
		ycoods = [TP2_y,Head_y]
		return np.interp(x,xcoods,ycoods)

def dist(x1,y1,x2,y2):
	return math.sqrt((y2-y1)**2+(x2-x1)**2)

def get_pack_heights(gan_ob): # OOP method !! gan_ob is a class
	idler_pack = []
	for coods in gan_ob.idler_coods_rotated: #coods is a matrix list of x&y coods of the gantry nodes in position
		pack_height = belt(coods[0,0])-coods[0,1]
		idler_pack.append(pack_height)
	min_pack_h2 = min(idler_pack)
	max_pack_h2 = max(idler_pack)
	return min_pack_h2, max_pack_h2

"""calculate key points"""
TP1_x = length1*math.cos(math.radians(alpha1))
TP1_y = length1*math.sin(math.radians(alpha1))+TOB
CIR_x = TP1_x - radius*math.sin(math.radians(alpha1))
CIR_y = TP1_y + radius*math.cos(math.radians(alpha1))
TP2 = rotation((alpha1-alpha2),np.matrix([TP1_x,TP1_y]),np.matrix([CIR_x,CIR_y]))
TP2_x = TP2[0,0]
TP2_y = TP2[0,1]
Head_x = TP2_x+length2*math.cos(math.radians(alpha2))
Head_y = TP2_y+length2*math.sin(math.radians(alpha2))

fig = plt.figure()
ax = fig.add_subplot(111)

#pdb.set_trace()

def plot_belt(*args, **kwargs):

	ax.plot([0,TP1_x],[TOB,TP1_y], color = 'black', linewidth =1)

	xcircle = np.linspace(TP1_x,TP2_x,50)-CIR_x
	ycircle = -np.sqrt(radius**2-xcircle**2)+CIR_y

	x_belt = [TP1_x]
	for i in xcircle:
		x_belt.append(i+CIR_x)
	x_belt.append(Head_x)

	y_belt = [TP1_y]
	for i in ycircle:
		y_belt.append(i)
	y_belt.append(Head_y)

	ax.plot(x_belt,y_belt, color ='black', linewidth =0.75)

	xcircle = np.linspace(TP1_x,TP2_x,50)-CIR_x
	ycircle = -np.sqrt(radius**2-xcircle**2)+CIR_y

	ax.plot(xcircle+CIR_x,ycircle, color ='black', linewidth = 0.75)

	ax.set_xlim(0, Head_x)
	ax.set_ylim(0, Head_y*2)
	plt.savefig('foo.png')
	plt.show()

def plot_ground():

	""" plot grounds """
	ax.plot(groundx,groundy, color ='darkred', linewidth =1.5)

def plot_truss():
	for j in range(len(gantry_list)):
		xtruss = [i[0,0] for i in gantry_list[j].truss_rotated]
		xtruss.append(gantry_list[j].truss_rotated[1,0])
		ytruss = [i[0,1] for i in gantry_list[j].truss_rotated]
		ytruss.append(gantry_list[j].truss_rotated[1,1])
		ax.plot(xtruss,ytruss, color='black', linewidth = .5)

def plot_steel():

	### steel line plotting
	x_llm = [column[0] for column in llm_list]
	y_llm = [column[1] for column in llm_list]
	plt.plot(x_llm,y_llm, color ='lightblue', linewidth =1)

	x_ebm = [column[0] for column in ebm_list]
	y_ebm = [column[1] for column in ebm_list]
	ax.plot(x_ebm,y_ebm, color ='lightgreen', linewidth =1)

	x_gan = [i.x1 for i in gantry_list]
	x_gan.insert(0,gantry_list[0].x2)
	y_gan = [i.y1 for i in gantry_list]
	y_gan.insert(0,gantry_list[0].y2)
	ax.plot(x_gan,y_gan, color ='black', linewidth =0.7)

	for i in gantry_list:
		if i.length ==2:
			continue
		ax.annotate(i.length, xy=(i.x1+i.length/2, i.y1), xycoords='data', xytext=(0, -0.5), textcoords='offset points',fontsize=8, color = 'red')

	for i in range(len(trestle_list)):
		ax.plot(trestle_list[i].coodsx,trestle_list[i].coodsy, color ='black', linewidth =0.75)

	for i in trestle_list:
		ax.annotate(int(round((i.y1-i.y0),0)), xy=(i.x0, i.y1/2), xycoords='data', xytext=(5, 0), textcoords='offset points',fontsize=8, color = 'green')


def main():

	"""plot belt"""
	plot_belt()
	plot_ground()

	"""set initial calc parameters"""
	x1 = Head_x
	y1 = Head_y
	take_up = 0

	""" start at head end and move back to offset connection"""
	angle = degrees(math.atan(gradient(Head_x-head_offset)))
	x_belt = Head_x-head_offset*cos(radians(angle))
	y_belt = Head_y-head_offset*sin(radians(angle))

	"""get pivot point of gantry position"""
	x2_gantry = x_belt+a*sin(radians(angle))
	y2_gantry = y_belt-a*cos(radians(angle))

	"""gantries"""
	while (y1 - ground(x1)) > 2.5:

		"""position gantry and check"""
		length = 24
		angle = degrees(math.atan(gradient(x2_gantry)))

		min_pack = 100
		max_pack = 100

		"""add take up tower"""
		"""DEBUG to add take up tower at least 1 gantry away from first pivot"""
		if y2_gantry-ground(x2_gantry) < 12 and take_up == 0:
			length = 2
			take_up = 1
			min_pack = 0

		while min_pack > 0.05:
			min_pack, max_pack = get_pack_heights(gantry(x2_gantry,y2_gantry,angle,length))
			if min_pack <= 0.05:
				break
			angle -= 0.1

		while max_pack > .4:
			min_pack, max_pack = get_pack_heights(gantry(x2_gantry,y2_gantry,angle,length))
			if max_pack <= 0.4:
				break
			length -= 6

		"""get parameters"""
		x1 = gantry(x2_gantry,y2_gantry,angle,length).x1
		y1 = gantry(x2_gantry,y2_gantry,angle,length).y1

		while y1 - ground(x1) < 2.5:
			length -= 6
			x1 = gantry(x2_gantry,y2_gantry,angle,length).x1
			y1 = gantry(x2_gantry,y2_gantry,angle,length).y1
			#pdb.set_trace()
			if length ==6:
				break
			if y1 - ground(x1) >= 2.5:
				length += 6
				break

		"""add gantries"""
		gantry_list.append(gantry(x2_gantry,y2_gantry,angle,length))

		"""add trestles """
		trestle_list.append(trestle(x1,ground(x1),y1-b)) #check

		""" increment to next gantry"""
		x2_gantry = x1
		y2_gantry = y1

	""" add elevated beam module"""
	ebm_list.append([x1,y1])
	while (y1 - ground(x1)) > 1.5:
		x1 -= 0.1
		y1 = belt(x1)
	ebm_list.append([x1,y1])

	""" add ground module to the end"""
	llm_list.append([x1,y1])
	llm_list.append([0,TOB])

	plot_steel()
	plot_truss()

if __name__ == '__main__':
	main()

##### plot setup

ax.set_xlim(0, Head_x)
ax.set_ylim(min(groundy), Head_y*3)
plt.savefig('foo.png')
plt.show()
