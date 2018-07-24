import numpy as np
import subprocess

from PIL import Image, ImageDraw

import draw as d

import check


BALL_DIAMETER = 0.017270193333
DIMENSION_X = 0.287351820006
DIMENSION_Y = 0.01
DIMENSION_Z = 0.0425280693609
BALL_INIT_POS_X = 0  # in grid
BALL_INIT_POS_Z = 10 # in grid

PENALTY_FACTOR = 5


class Ball_run(object):

	def __init__(self, number_obstacles):
		
		self.goal = []
		self.n_obstacles = number_obstacles
		self.obstacles = [] 
		self._setup_obstacles()
		self._read_goal_setup()
		self.grid_size = 10
		self.counter = 0
		self.printcounter = 0
		self.actions = 12
		self.reset()		
		#self.goal_len = 0.0
		#self.goal_height = 0.0
		#self.goal_pos_x = 0.0
		#self.goal_pos_z  = 0.0
		#self.goal_left_bound = 0.0
		#self.goal_right_bound = 0.0
		#self.goal_grid = 0
		self.success = False
		

	def _setup_obstacles(self):

		for i in range(self.n_obstacles):

			o = Obstacle()
			o.dimensions.append(DIMENSION_X)
			o.dimensions.append(DIMENSION_Y)
			o.dimensions.append(DIMENSION_Z)

			self.obstacles.append(o)



	# reset positions and angles
	def reset(self):

		state = []
		
		# state is an array with x_pos, z_pos and angle of each obstacles

		for i in range(0, self.n_obstacles):
			
			#print i
			x = np.random.randint(self.grid_size, size = 1) * 0.05
			z = np.random.randint(1, self.grid_size, size = 1) * 0.05
			a = (np.random.randint(-5, 5, size = 1) * 0.2)

			self.obstacles[i].pos.append(x) 
			self.obstacles[i].pos.append(0.0)
			self.obstacles[i].pos.append(z) 
			
			self.obstacles[i].angle = a

			state.append(x)
			state.append(z)
			state.append(a)

		# rounded, reduce case number
		state.append(self.goal_x_grid)
		state.append(self.goal_z_grid)
		
		state.append(BALL_INIT_POS_X)
		state.append(BALL_INIT_POS_Z)

		self.state = state

		# write change to file
		self._write_setup()
        	self.counter = 0
		



	def _read_goal_setup(self):

            f = open("./params/goal_setup.txt", 'r')

            for line in f :
                o = Obstacle()
                contents = line.split(' ')
                
                if len(contents) == 7:
                    o.dimensions.append(contents[0])
                    o.dimensions.append(contents[1])
                    o.dimensions.append(contents[2])
                    o.pos.append(contents[3])
                    o.pos.append(contents[4])
                    o.pos.append(contents[5])
                    o.angle = contents[6]
                        
                self.goal.append(o)

            f.close()
            self.goal.sort(key = lambda x: x.dimensions[0])

            # the sort is in ascending order
            mid = self.goal[-1]
            f = open("./text.txt",'w')
            f.write(str(mid))
            self.goal_len = float(mid.dimensions[0])
            f.write(str(self.goal_len))
            f.close()
            self.goal_height = float(mid.dimensions[2])
            self.goal_pos_x = float(mid.pos[0])
            self.goal_pos_z = float(mid.pos[2])
            self.goal_left_bound = self.goal_pos_x - (self.goal_len/2.0 )
            self.goal_right_bound = self.goal_left_bound + self.goal_len
            self.goal_x_grid = (int(self.goal_pos_x / 0.05)) * 0.05
            self.goal_z_grid = (int(self.goal_pos_z // 0.05)) * 0.05



	def _get_reward(self):

            overlap_count = check.get_overlap_count(self.obstacles)
    
            rw = 0

            ball_x = self.state[-2]
            ball_z = self.state[-1]

            if self.success :
                # we don't want overlapping
                # the reward will be 1 if the overlap_count is 0
                rw = 1 - overlap_count
            
            # optimize the obstacles position w.r.t. ball pos
            # need to push the ball to the goal position
            # which usually locates at the right end of the grid
            
            # the ball is on the ground
            elif ball_x < 0:
                penalty =  (ball_x - self.goal_pos_x + 1)*(ball_x - self.goal_pos_x + 1) + overlap_count
                rw = -(penalty *PENALTY_FACTOR)
            else :
                # minimize the dist between ball and leftmost obstacle
                penalty = (ball_x - self.goal_pos_x + 1)*(ball_x - self.goal_pos_x + 1) + (ball_x - self.obstacles[0].pos[0])*(ball_x - self.obstacles[0].pos[0]) + overlap_count

                rw  = -(penalty *PENALTY_FACTOR)

            return rw

	def _update_state(self, action):
            """
            Input: action and states
            Ouput: new states and reward
            """
            #state = self.state
            out = []
            #print self.state
            
            if action == 0:
                # no greater than 0.5
                self.obstacles[0].pos[0] = min(self.obstacles[0].pos[0] + 0.05 , 0.5)
    
            elif action == 1 :
                # no smaller than 0
                self.obstacles[0].pos[0] = max(0, self.obstacles[0].pos[0] - 0.05)
               
            elif action == 2 :
                self.obstacles[0].pos[2] = min(self.obstacles[0].pos[2] + 0.05 , 0.5)

            elif action == 3 :
                self.obstacles[0].pos[2] = max(0, self.obstacles[0].pos[2] - 0.05)

            elif action == 4 :
                # no greater than 1
                self.obstacles[0].angle = min(self.obstacles[0].angle + 0.2 , 1)

            elif action == 5 :
                # no smaller than -1
                self.obstacles[0].angle = max(self.obstacles[0].angle - 0.2, -1)
                
            elif action == 6 :
                self.obstacles[1].pos[0] = min(self.obstacles[1].pos[0] + 0.05 , 0.5)
            elif action == 7 :
                self.obstacles[1].pos[0] = max(0, self.obstacles[1].pos[0] - 0.05)
            elif action == 8 :
                self.obstacles[1].pos[2] = min(self.obstacles[1].pos[2] + 0.05 , 0.5)
            elif action == 9 :
                self.obstacles[1].pos[2] = max(0, self.obstacles[1].pos[2] - 0.05)
            elif action == 10 :
                self.obstacles[1].angle = min(self.obstacles[1].angle + 0.2 , 1)
            elif action == 11 :
                self.obstacles[1].angle = max(self.obstacles[1].angle - 0.2, -1)
            elif action == 12 :
                self.obstacles[2].pos[0] = min(self.obstacles[2].pos[0] + 0.05 , 0.5)
            elif action == 13 :
                self.obstacles[2].pos[0] = max(0, self.obstacles[2].pos[0] - 0.05)
            elif action == 14 :
                self.obstacles[2].pos[2] = min(self.obstacles[2].pos[2] + 0.05 , 0.5)
            elif action == 15 :
                self.obstacles[2].pos[2] = max(0, self.obstacles[2].pos[2] - 0.05)
            elif action == 16 :
                self.obstacles[2].angle = min(self.obstacles[2].angle + 0.2 , 1)
            elif action == 17 :
                self.obstacles[2].angle = max(self.obstacles[2].angle - 0.2, -1)

            # fix the overlapping problem
            all_obst = self.obstacles + self.goal
            valid_obs = check.get_valid_output(all_obst)

            self.obstacles = valid_obs[:len(self.obstacles)]

            self.obstacles.sort(key = lambda x: x.pos[0])
            self._write_setup()

            # call the dynamic
            subprocess.call("./dynamics", shell= True)

            for o in self.obstacles:
                out.append(round(float( o.pos[0]),2))
                out.append(round(float( o.pos[2]),2))
                out.append(round(float( o.angle),2))

            out.append(self.goal_x_grid)
            out.append(self.goal_z_grid)

            f = open('./results/result.txt', 'r')
            l = f.readlines()
            result = l[0].split(" ")
            f.close()

            x = int(float(result[0])/0.05) * 0.05
            z = int(float(result[1])/0.05) * 0.05
            
            out.append(x)
            out.append(z)
            print out

            assert(len(out)) == self.n_obstacles * 3 + 4
            self.state = out


	def _write_setup(self):

            f1 = open("./params/dyn_setup.txt", 'w')

            for o in self.obstacles:
                
                f1.write(str(DIMENSION_X) + ' ' + str(DIMENSION_Y) + ' ' + str(DIMENSION_Z) + ' ')
                f1.write(str(float(o.pos[0])) + ' 0.0 ' + str(float(o.pos[2])) + ' ' + str(float(o.angle))+'\n' )

            for o in self.goal:
                
                f1.write(str(o.dimensions[0]) + ' ' + str(o.dimensions[1]) + ' ' + str(o.dimensions[2]) + ' ' )
                f1.write(str(o.pos[0]) + ' ' + str(o.pos[1]) + ' ' + str(o.pos[2]) + ' ')
                f1.write(str(o.angle))

            f1.close()


	def _draw_state(self):

		self.printcounter +=1
        # 110 : the range of the "wall" in simulation is 0 - 0.55
		L=110; W=110
		image = Image.new("1", (L, W))
		draw = ImageDraw.Draw(image)

		state = self.state
		
		for i in range(0, self.n_obstacles):
		    vertices = d.makeRectangle(56, 8.5, float(state[i*3+2]), offset=(int(state[i*3]*200) , 110 - int(state[i*3+1]*200)))
		    draw.polygon(vertices, fill=1)
		
		# draw goal
		goal_l = int(self.goal_len*200)
		goal_w = int(self.goal_height * 200)

		ver = d.makeRectangle(goal_l, goal_w, 0, offset=(int(self.goal_x_grid*200) ,110 - int(self.goal_z_grid * 200)))
		draw.polygon(ver, fill=1)

		# draw the ball
		ball_x = int(state[-2]*200)
		ball_z = 110 - int(state[-1]*200)

		v = d.makeRectangle(5,5,0,offset=(ball_x, ball_z))
		draw.polygon(v, fill =1)

		if (self.counter >= 49):
			image.save("./frames/%5d.png"%self.printcounter)
		image.save("./test.png")
		
        	arr = np.fromiter(iter(image.getdata()), np.uint8)
        	arr.resize(110, 110)

		return arr
        	'''
		im_size = (self.grid_size,)*2
		state = self.state
		canvas = np.zeros(im_size)
		#canvas[state[-2], state[-1]] = 1  # draw ball
		# draw obstacles
		for i in range(self.n_obstacles):
			canvas[state[i * 3], state[i*3 + 2] - 2 : state[1*3 +2] + 3 ] = 1

		return canvas
        	'''

	def _is_over(self):

		f = open('./results/rstring.txt', 'r')
		l = f.readlines()
		result = l[0]
		self.counter += 1
		f.close()
		if result[0] == 's':
			f1 = open("./results/success.txt", 'a')
			f1.write('hit at %d'%self.printcounter)
			f1.close()
			
			self.success = True
            		self.counter = 0
		   	return True
		elif self.counter > 50:
            		self.counter = 0
			return False
		else :
			return False
	

	def observe(self):
		canvas = self._draw_state()
		return canvas.reshape((1, -1))


	def act(self, action):
		self._update_state(action)
		
		game_over = self._is_over()
		reward = self._get_reward()
		return self.observe(), reward, game_over


class Obstacle():

	def __init__(self):
		self.dimensions = []
		self.pos = []
		self.angle = 0.0


