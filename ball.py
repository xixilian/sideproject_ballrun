import numpy as np






BALL_DIAMETER = 0.017270193333
DIMENSION_X = 0.287351820006
DIMENSION_Y = 0.01
DIMENSION_Z = 0.0425280693609


class Ball_run(object);

	def _init_(self, number_obstacles):
		
		self.goal = []
		self.n_obstacles = number_obstacles
		self.obstacles = [] 
		_read_goal_setup()
		self.reset()
		self.left_bound = 0.0
		self.right_bound = 0.0
		# using grid to simplifie the env
		self.grid_size = 10


	def reset(self):
		self.obstacles = []


		for i in range(0, self.n_obstacles):
			o = Obstacle()

			x = np.random.randint(0, self.grid_size-1, size = 1)
			z = np.random.randint(1, self.grid_size-1, size = 1)
			a = np.random.uniform(-1, 1, size = 1)

			o.pos[0] = x*0.05
			o.pos[1] = 0.0
			o.pos[2] = z*0.05
			o.dimensions[0] = DIMENSION_X
			o.dimensions[1] = DIMENSION_Y
			o.dimensions[2] = DIMENSION_Z
			o.angle = a

			self.obstacles.append(o)

		f =  open("./params/dyn_setup.txt", 'w')
		f2 =  open("./setup.txt", 'w')

		for o in self.obstacles:
			f.write('o')
			f2.write('o')
			
			
			for d in o.dimensions:
				f.write( " " + str(d))
				f2.write( " " + str(d))
			for p in o.pos:
				f.write(" " + str(p)) 
				f2.write(" " + str(p))

			f.write( " " + o.angle+ '\n')
			f2.write( " " + o.angle+ '\n')

		for o in self.goal:
			f.write('o')
			
			for d in o.dimensions:
				f.write( " " + str(d))
			for p in o.pos:
				f.write(" " + str(p))
			f.write(" " + o.angle+ '\n')

		f.close()
		f2.close()
 

	def _read_goal_setup(self):

		f = open("./params/goal_setup.txt", 'r')

		for line in f :
			o = Obstacle()
			contents = line.split(' ')
			if len(contents) == 7:
				o.dimensions[0] = contents[0]
				o.dimensions[1] = contents[1]
				o.dimensions[2] = contents[2]
				o.pos[0] = contents[3]
				o.pos[1] = contents[4]
				o.pos[2] = contents[5]
				o.angle = contents[6]
			self.goal.append(o)


		f.close()

		max_x_len = goal[np.argsort(goal.dimensions[:, 0]), :][0].dimensions[0]
		self.left_bound = goal[0].pos[0] - max_x_len/2.0
		self.right_bound = self.left_bound + max_x_len





class Obstacle():

	def _init_(self):
		self.dimensions = []
		self.pos = []
		self.angle = 0.0


