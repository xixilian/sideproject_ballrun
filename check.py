from __future__ import division

from multimethod import multimethod

import os
import subprocess
import sys
import time
import glob

import numpy as np
import scipy
from scipy.optimize import differential_evolution, basinhopping

import math

from PIL import Image, ImageDraw




BALL_DIAMETER = 0.017270193333
DIMENSION_X = 0.287351820006
DIMENSION_Y = 0.01
DIMENSION_Z = 0.0425280693609

class Obstacle():

	def __init__(self):
		self.dimensions = []
		self.pos = []
		self.angle = 0.0

def get_init_setup():

	obst = []
  	#print(str(x))
	#f1 = open("./params/dyn_setup.txt", 'r')
	#f1 = open("./tests/tofix/best_from_select.txt", 'r')
	#f1 = open("./tests/best_from_select.txt", 'r')
	f1 = open("./tests/best_setup.txt", 'r')
	#l = f1.readlines()
	#print(len(l))
  	#for i in range(0, len(l) - 3):
	lines = f1.readlines()
	for line in lines:
		#print line
		#if not line.strip():
	
			#print i 
		cont = line.split(' ')
		
		#print cont
		if len(cont) == 8:
  			o = Obstacle()
  			o.dimensions.append(float(cont[1]))
  			o.dimensions.append(float(cont[2]))
  			o.dimensions.append(float(cont[3]))
  			o.pos.append(float(cont[-4]))
  			o.pos.append(float(cont[-3]))
  			o.pos.append(float(cont[-2]))
 			o.angle = float(cont[-1])

  			obst.append(o)

	f1.close()
  	return obst 


def get_valid_output(obst):

    f1 = open('./tests/what_did_you_get.txt', 'w')

    for o in obst:
		f1.write("o ")
		f1.write(str(o.dimensions[0]) + ' ' + str(DIMENSION_Y) + ' ' + str(o.dimensions[2]) + ' ')
		f1.write(str(float(o.pos[0])) + ' 0.0 ' + str(float(o.pos[2])) + ' ' + str(float(o.angle))+'\n' )

    f1.close()

    movable = []
    fixed = []

    if len(obst) == 7 :
        #one fixed obstacle
        movable = obst[:3]
        fixed = obst[-4:]
    else :
        movable = obst[:3]
        fixed = obst[-3:]

    # from left to right in x direction
	movable.sort(key = lambda x: x.pos[0])

	collide = set([])

    cmd = []
    #cmd = "./track".split()
    cmd.append("./track")
    setup_path = './tests/what_did_you_get.txt'
    cmd.append(setup_path)
    #print cmd
    subprocess.call(cmd, shell = True)
    f = open("./tests/track.txt", 'r')

    lines = f.readlines()
    os.remove('./tests/hit_track.txt')
    f2 = open('./tests/hit_track.txt', 'a')

    counter = []

    for i in range(len(movable)):
        counter.append(0)

    for l in lines:
        track = l.split()
        pos_x = float(track[0])
        pos_z = float(track[1])

        if(len(collide) == len(movable)):
            break

        for j, o in enumerate(movable) :
            ver = makeRectangle(o.dimensions[0], o.dimensions[2], o.angle, (o.pos[0], o.pos[2]))
        
            top = under = ver[0][1]
            left =right = ver[0][0]
			
            for v in ver:
                if v[0] < left:
                    left = v[0]
                if v[0] > right :
                    right = v[0]
                if v[1] < under :
                    under = v[1]
                if v[1] > top:
                    top = v[1]

                if(pos_x >= (left- BALL_DIAMETER)  and pos_x <= (right + BALL_DIAMETER) and pos_z >= (under - BALL_DIAMETER) and pos_z <= (top+ BALL_DIAMETER)):
                    f2.write("colliding with movable %1d"%movable.index(o) + '\n')
                    f2.write(str(ver) + '\n')

                ds = []
                p = (pos_x, pos_z)

                for i in range(0,len(ver)):

                    ind2 = (i + 1)% len(ver)
                    p1 = ver[i]
                    p2 = ver[ind2]

                    f2.write(str(p1) +  ' ' + str(p2) +'\n')
                    d =  distance(p, p1, p2)
                    f2.write(str(d) +'\n')
                    ds.append(d)

                d = min(ds)

                f2.write(str(ds) + '\n')
                f2.write(str(d) + '\n')
                f2.write(str(p) + '\n')

                if (d < BALL_DIAMETER + BALL_DIAMETER/2.0):
                    counter[j] += 1
			
	
	for i, c in enumerate(counter):
		if c > 5:
			collide.add(movable[i])


	f.close()
	f2.close()
	# before everything changed, identify which obstacle is idle
	idle = []
	idle_index = []

	for o in movable:
		if not o in collide:
			idle.append(o)

	coll = []

	for i, o in enumerate( collide):
		print ("collide %1d"%i, o.pos[0] )
		coll.append(o)

	for o in idle:
		print ("idle %1d"%idle.index(o), o.pos[0] )

	coll.sort(key = lambda x: x.pos[0])
	
	right = fixed[-2]
	left = fixed[-3]

	if (right.pos[0] < left.pos[0]):
		tmp = right
		right = left
		left = tmp

	if (len(idle) > 0):
		
		ver_right = makeRectangle(right.dimensions[0], right.dimensions[2], right.angle, (right.pos[0], right.pos[2]))
		ver_left = makeRectangle(left.dimensions[0], left.dimensions[2], left.angle, (left.pos[0], left.pos[2]))
				
		if (len(idle) > 2):
		# we dont need that much neither
			for i in range(2, len(idle)):
				idle[i].pos[0] = 10

			idle[0] = move_idle_right(idle[0], right)
			idle[1] = move_idle_left(idle[1], left) 
			i = idle[0]
			j = idle[1]
			ver_0 = makeRectangle(i.dimensions[0], i.dimensions[2], i.angle, (i.pos[0], i.pos[2]))
			ver_1 = makeRectangle(j.dimensions[0], j.dimensions[2], j.angle, (j.pos[0], j.pos[2]))
			
			if len(fixed) > 3:
				for f in fixed:
					ver_f =  makeRectangle(f.dimensions[0], f.dimensions[2], f.angle, (f.pos[0], f.pos[2]))
					if overlap(ver_0, ver_f):

						idle[0].pos[0] = 0.5
						idle[0].pos[2] = 0.5	
				
					if(overlap(ver_1, ver_f)):
					
						idle[1].pos[0] = 0.5
						idle[1].pos[2] = 0.5 
		
		elif(len(idle) == 1):
			
			i = idle[0]
			
			i = move_idle_right(i, right)
			

			if len(fixed) > 3:
				ver = makeRectangle(i.dimensions[0], i.dimensions[2], i.angle, (i.pos[0], i.pos[2]))
				for f in fixed:
					ver_f =  makeRectangle(f.dimensions[0], f.dimensions[2], f.angle, (f.pos[0], f.pos[2]))
					if overlap(ver, ver_f):
						i.angle = idle[0].angle/2.0
						i = move_idle_right(i, right)
					
	
			idle[0] = i
		
			
		elif(len(idle) == 2):

			i = move_idle_right(idle[0], right)
			j = move_idle_left(idle[1], left) 
			
			if len(fixed) > 3:
				ver_0 = makeRectangle(i.dimensions[0], i.dimensions[2], i.angle, (i.pos[0], i.pos[2]))
				ver_1 = makeRectangle(j.dimensions[0], j.dimensions[2], j.angle, (j.pos[0], j.pos[2]))
			
				for f in fixed:
					ver_f =  makeRectangle(f.dimensions[0], f.dimensions[2], f.angle, (f.pos[0], f.pos[2]))
					if overlap(ver_0, ver_f):
						i.angle = i.angle/2.0
						i = move_idle_right(i, right)
					if(overlap(ver_1, ver_f)):
						j.angle = j.angle/2.0
						j = move_idle_left(j, left) 
				
			idle[0] = i
			idle[1] = j


	if (len(collide) > 1):
		
		#for o1, o2 in zip(coll, coll[1:]):
		for i in range(len(coll)):
						

			ind2 = (i + 1)% len(coll)

			o1 = coll[i]
			o2 = coll[ind2]
	
			arr  = [o1,o2]
			arr.sort(key = lambda x: x.pos[0])

			left = arr[0]
			right = arr[-1]			

			ind_1 = coll.index(o1)
			ind_2 = coll.index(o2)
			print "comparing %1d"%ind_1
			print "and %1d"%ind_2
			o1_ver = makeRectangle(o1.dimensions[0], o1.dimensions[2], o1.angle, (o1.pos[0], o1.pos[2]))
			o2_ver = makeRectangle(o2.dimensions[0], o2.dimensions[2], o2.angle, (o2.pos[0], o2.pos[2]))

			if(overlap(o1_ver, o2_ver)):
				
                print "fixing %1d"%ind_1
				print "and %1d"%ind_2
				# decide which direction should move, and we move 
				#new1, new2 = fix_movables(o1, o2, o1_ver, o2_ver, fixed)
				(new1, new2) = fix_movables((left, right), fixed)

				coll[ind_1] = new1
				coll[ind_2] = new2

	# check if the movable overlap with fixed object
	
	if (len(collide) > 0):
		for o in coll :
		
			
			ver = makeRectangle(o.dimensions[0], o.dimensions[2], o.angle, (o.pos[0], o.pos[2]))
			
			for fix in reversed( fixed ):

				
				f_ver =  makeRectangle(fix.dimensions[0], fix.dimensions[2], fix.angle, (fix.pos[0], fix.pos[2]))

				if(overlap(ver, f_ver)):
					print ("fixed index ", fixed.index(fix))
					ind =  coll.index(o)
					p1 = (o.pos[0], o.pos[2])
					p2 = (fix.pos[0], fix.pos[2])
										
					
					shift, up = get_shift_dist(o, fix, fixed)
					
                    # print ("shift" , shift)
                    #print ( "up" , up)
					
					if o.pos[0] < 0.25:
						o.pos[0] -= abs(shift)
					else:
						o.pos[0] += abs(shift)
					o.pos[2] += up
	
			
			if(len(idle) > 0):
				for i in idle :
				
					ver_i =  makeRectangle(i.dimensions[0], i.dimensions[2], i.angle, (i.pos[0], i.pos[2]))

					if (overlap(ver, ver_i)):
                        # print "hit idle"
						ind = idle.index(i)
						shift_d = get_shift_point_dist(ver, ver_i)					
						shift = shift_d * math.cos(o.angle)
						up = shift_d * math.sin(o.angle)
                        
						#left side
						if(i.angle > 0):
							
							# put it somewhere else
							i.pos[0] = 0.6
							i.pos[2] = 0.5
                        
						# right side
						elif(i.angle < 0):
							i.pos[0] += shift
							i.pos[2] -+ up
						idle[ind] = i



	return coll+idle+fixed


# distance from point p0 to line formed by p1, p2
def distance(p0, p1, p2): # p0 is the point
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    nom = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denom = math.sqrt(((y2 - y1)**2 + (x2 - x1) ** 2))
    result = nom / denom
    return result


def move_idle_right(i, right):
	i.angle = -abs(i.angle)

	ver = makeRectangle(i.dimensions[0], i.dimensions[2], i.angle, (i.pos[0], i.pos[2]))
	copy = ver
	copy.sort(key = lambda x: x[0])
	ver_right = makeRectangle(right.dimensions[0], right.dimensions[2], right.angle, (right.pos[0], right.pos[2]))
	copy_right = ver_right
	copy_right.sort(key = lambda x: x[0])

	shift = copy_right[-1][0] - copy[0][0]
			
	i.pos[0] += shift
	
	copy.sort(key = lambda x: x[1])
	copy_right.sort(key = lambda x: x[1])
	up = copy_right[-1][1] - ver[1][1]
			
	i.pos[2] += up
	
	return i 
			
def move_idle_left(i, left):
	
	#if(idle[1].angle > 0 ):
	i.angle = abs(i.angle)
	ver_l = makeRectangle(i.dimensions[0], i.dimensions[2], i.angle, (i.pos[0], i.pos[2]))

	copy_l = ver_l
	copy_l.sort(key = lambda x: x[0])
	ver_left = makeRectangle(left.dimensions[0], left.dimensions[2], left.angle, (left.pos[0], left.pos[2]))
	copy_left = ver_left
	copy_left.sort(key = lambda x: x[0])

	shift = copy_left [0][0] - copy_l[-2][0]

	copy_l.sort(key = lambda x: x[1])
	copy_left.sort(key = lambda x: x[1])
			
	i.pos[0] += shift
	up = copy_left[-1][1] - copy_l[0][1]
			
	i.pos[2] += up

	return  i
			 			

def fix_movables(o_list, fixed):
	
	o1 = o_list[0]
	o2 = o_list[-1]
	new = []
	print o1.pos[0], o2.pos[0]
	x_1 = o1.pos[0]
	x_2 =  o2.pos[0]
	dim_x_1 = o1.dimensions[0] 
	dim_x_2 = o2.dimensions[0]

	right = fixed[-2]
	
	left = fixed[-3]

	if (right.pos[0] < left.pos[0]):
		tmp = right
		right = left
		left = tmp

	ver_right = makeRectangle(right.dimensions[0], right.dimensions[2], right.angle, (right.pos[0], right.pos[2]))
	ver_left = makeRectangle(left.dimensions[0], left.dimensions[2], left.angle, (left.pos[0], left.pos[2]))
	ver_left.sort(key = lambda x: x[0])
	left_bound = ver_left[0][0]

	p1 = (o1.pos[0], o1.pos[2])
	p2 = (o2.pos[0], o2.pos[2])
	d = get_point_line_dist(p1, p2, o2.angle)

	o1_ver = makeRectangle(o1.dimensions[0], o1.dimensions[2], o1.angle, (o1.pos[0], o1.pos[2]))
	o2_2_ver = makeRectangle(o2.dimensions[0], o2.dimensions[2], o2.angle, (o2.pos[0], o2.pos[2]))
	o2_2_ver.sort(key = lambda x: x[0])
	left_x_2 = o2_2_ver[0][0]
		

	if(x_1 > -(dim_x_1)/4.0) and (x_2 < dim_x_2):
 
		shift_d = get_shift_point_dist(o1_ver, o2_2_ver)
		shift = shift_d * math.cos(o1.angle)

		o1.pos[0] -= abs(shift)
		up = shift_d * math.sin(o1.angle)
		
		if up < 0.001:
		
			up += 0.0001
			
        #print "up"

		o1.pos[2] += abs(up)

	else:
        # print "shift the right one"
	
		shift_d = get_shift_point_dist(o1_ver, o2_2_ver)
		shift = shift_d * math.cos(o1.angle)
		o2.pos[0] += abs(shift)
		up = shift_d * math.sin(o1.angle)
		
		if up < 0.001:
			o2_2_ver.sort(key = lambda x: x[1])
			o1_ver.sort(key = lambda x: x[1])
			low = o2_2_ver[-1][1]
			high = o1_ver[0][1]
			up = high - low	
		o2.pos[2] -= abs(up)
    
	new.append(o1)
	new.append(o2)
	return new

def get_overlap_count(obst):
	
	for o in obst:
		o.dimensions = np.array(o.dimensions, dtype=np.float32)
		o.pos = np.array(o.pos, dtype=np.float32)
		o.angle = float(o.angle)
	
	c = 0
	
	movable = []
	fixed = []
	
	if len(obst) == 7 :
		# one fixed obstacle
		movable = obst[:3]
		fixed = obst[-4:]
    
	if len(obst) == 6 :
        
		movable = obst[:3]
		fixed = obst[-3:]
    
    # two movable obstacles
    if len(obst) == 5 :
        
        movable = obst[:2]
        fixed = obst[-3:]
    
	# from left to right in x direction
	movable.sort(key = lambda x: x.pos[0])

	f = open('./tests/overlapcount.txt', 'a')


	for i in range(len(movable)):
						
		
		ind2 = (i + 1)% len(movable)

		o1 = movable[i]
		o2 = movable[ind2]
		
		o2_ver = makeRectangle(o2.dimensions[0], o2.dimensions[2], o2.angle, (o2.pos[0], o2.pos[2]))		

		o1_ver = makeRectangle(o1.dimensions[0], o1.dimensions[2], o1.angle, (o1.pos[0], o1.pos[2]))
		
		f.write("movable %1d, "%movable.index(o1) + "movable %1d"%movable.index(o2))
		f.write('\n')
		f.write('o1 \n')
		f.write(str(o1_ver))
		f.write('\n')
		f.write('o2 \n')
		f.write(str(o2_ver))
		f.write('\n')
		
		if(overlap(o1_ver, o2_ver)):
		
			c += 1

	for i in range(len(fixed)):
		
		o = fixed[i]
		v = makeRectangle(o.dimensions[0],o.dimensions[2], o.angle, (o.pos[0], o.pos[2]))
		
		for m in movable:
		
			ver_m =  makeRectangle(m.dimensions[0],m.dimensions[2], m.angle, (m.pos[0], m.pos[2]))	
				
			f.write(" movable %1d, "%movable.index(m) + "fixed %1d"%i)
			f.write('\n')
			f.write('movable  \n')	
			f.write(str(ver_m))
			f.write('\n')
			f.write('fixed \n')
			f.write(str(v))
			f.write('\n')
			f.write('\n')
			
			if(overlap(v, ver_m)):
				print ("hit movable %1d"%movable.index(m) + " fixed %1d "%i)
				c += 1
	return c



# it also works for convex polygons
def overlap(ver1, ver2):
	
	rects = [ver1, ver2]
	
	for r in rects :
		
		for i in range(len(r)):
			min_1 = None
			min_2 = None
			max_1 = None
			max_2 = None			

			ind2 = (i + 1)% len(r)
			p1 = r[i]
			p2 = r[ind2]

			normal = (p2[1] - p1[1] , p1[0] - p2[0])

			for v in ver1:		
				projected = normal[0] * v[0] + normal[1] * v[1]

				if ( not min_1 or projected < min_1 ):
					min_1 = projected
				if (not max_1 or projected > max_1):
					max_1 = projected

			for v in ver2:		
				projected2 = normal[0] * v[0] + normal[1] * v[1]

				if (not min_2 or projected2 < min_2):
					min_2 = projected2
				if (not max_2 or projected2 > max_2):
					max_2 = projected2

			if (max_1 < min_2 or max_2 < min_1):
				return False

	return True

# distance to shift along first obstacle's length
def get_shift_point_dist(ver1, ver2):
    # print ver1
    # print ver2
	
	ver1.sort(key = lambda x: x[1])
	under = ver1[0]
	
	len_l = ver1[1]
	tmp = ver1[2]

	# find the under length edge of the first obstacle
	d1 = math.sqrt((under[0] - len_l[0])**2 + (under[1] - len_l[1])**2)
	d2 = math.sqrt((under[0] - tmp[0])**2 + (under[1] - tmp[1])**2)
	
	if d2 > d1 :
		len_l = ver1[2]
	
	ver2.sort(key = lambda x: x[1])
	
	p = ver2[-1]

	left_up = p


	can_1 = ver2[-2]
	can_2 = ver2[-3]
	d1 = math.sqrt((p[0]- can_1[0])**2 + (p[1]- can_1[1])**2 )
	d2 = math.sqrt((p[0]- can_2[0])**2 + (p[1]- can_2[1])**2 )
	
	# find the upper length edge of second obstacle
	if(d2 > d1):
		left_up = can_2
	else:
		left_up = can_1

    '''
	print ("under ", under )
	print ("len_l ", len_l)
	print ("p ", p )
	print ("left_up ", left_up )
    '''
	intersect = get_intersect(under, len_l, p, left_up)

    #print intersect
    #print ("diff 1 : ", under[0] - intersect[0])
    #print ('diff 2 : ', under[1] - intersect[1])
	d = math.sqrt((under[0] - intersect[0])**2 + (under[1] - intersect[1])**2)
    # print ("d first calculated ", d)

	left = min(p[0], left_up[0])
	right = max(p[0], left_up[0])
	up = max(p[1], left_up[1])
	down = min(p[1], left_up[1])
	# if the intersect point is not on the line segment
	if (intersect[0] < left ) or (intersect[0] > right) or (intersect[1] < down) or (intersect[1] > up):
        # print "change"
		if p[0] < under[0]:
		
			d = math.sqrt((p[0] - under[0])**2 + (p[1] - under[1])**2)

		else:
			d = math.sqrt((left_up[0] - under[0])**2 + (left_up[1] - under[1])**2)

    #print ("d before return ", d)
	return d

# distance for movable obstacle to shift while colliding with a fixed object
def get_shift_dist(o, fix, l_fixed):

	ver = makeRectangle(o.dimensions[0], o.dimensions[2], o.angle, (o.pos[0], o.pos[2]))

	f_ver = makeRectangle(fix.dimensions[0], fix.dimensions[2], fix.angle, (fix.pos[0], fix.pos[2]))

	#if (fix.pos[0] > 0.25):
	ver.sort(key = lambda x: x[1])
	under = ver[0]
	
	len_l = ver[1]
	
	tmp = ver[2]

	# find the under length edge of the first obstacle
	d1 = math.sqrt((under[0] - len_l[0])**2 + (under[1] - len_l[1])**2)
	d2 = math.sqrt((under[0] - tmp[0])**2 + (under[1] - tmp[1])**2)
	
	if d2 > d1 :
		len_l = ver[2]

	f_ver.sort(key = lambda x: x[1])
	p = f_ver[-1]
	left_up = f_ver[-2]

	intersect = get_intersect(under, len_l, p, left_up)
	d1 = math.sqrt((under[0] - intersect[0])**2 + (under[1] - intersect[1])**2)
    # print ("d1 " , d1)
    # print ('intersect point : ', intersect)
	
	d2 = math.sqrt((len_l[0] - intersect[0])**2 + (len_l[1] - intersect[1])**2)
    # print ("d2 " , d2)
    # shift minimum distance
	d = min(d1, d2)
    # print ("d ", d)
	left = min(p[0], left_up[0])
	#print ("left ", left)
	right = max(p[0], left_up[0])
	#print ("right " , right)
	up = max(p[1], left_up[1])
	#print ("up ", up)	
	down = min(p[1], left_up[1])

    # print ("d before return ", d)
	delta_x = d* math.cos(o.angle)
	delta_z = d*math.sin(o.angle)
	shifted_x = o.pos[0] + d* math.cos(o.angle)
	shifted_z = d*math.sin(o.angle) + o.pos[2]

	shifted = makeRectangle(o.dimensions[0], o.dimensions[2], o.angle, (shifted_x, shifted_z))

	if overlap(shifted , f_ver):
        # print "not fixed yet"
		d = max(d1, d2)
		delta_x = d* math.cos(o.angle)
		delta_z = d*math.sin(o.angle)
	
	return delta_x, delta_z
	

	


# get intersect point of two lines
def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)



# get the distance from p1 to the line formed by p2 and the angle			
def get_point_line_dist(p1, p2, angle2):

	slope = math.tanh(angle2)
	
	b = p2[1] - slope*p2[0]
	

	# compute point projected on the line by p2
	denom = slope + 1/slope
	numerator = p1[1] + (1/slope)*p1[0] + slope*p2[0] - p2[1]

	x = numerator/denom
	y = slope * x + b
	d = math.sqrt((p1[0] - x)**2 + (p1[1] - y)**2)

	return d
	

def makeRectangle(l, w, theta, (x,y)):

    # avoid type error
	l = float(l)
	w = float(w)
	theta = float(theta)
	x = float(x)
	y = float(y)
	offset = (x, y)

	c, s = math.cos(theta), math.sin(theta)
	
	rectCoords = [(l/2.0, w/2.0), (l/2.0, -w/2.0), (-l/2.0, -w/2.0), (-l/2.0, w/2.0)]
	
	return [(c*x + s*y+offset[0], -s*x+c*y+offset[1]) for (x,y) in rectCoords]

