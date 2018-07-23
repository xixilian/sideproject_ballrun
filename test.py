#matplotlib for rendering
import matplotlib.pyplot as plt
#numpy for handeling matrix operations
import numpy as np
#time, to, well... keep track of time
import time


import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc

import dqn
import ball_run


#last frame time keeps track of which frame we are at
last_frame_time = 0
#translate the actions to human readable words
translate_action = ["1.ob left","1.ob right","1.ob up","1.ob down",
					"1.ob turn clockwise","1.ob turn anti clockwise","2.ob left",
					"2.ob right","2.ob up","2.ob down","2.ob clockwise","2.ob anti clockwise","End Test"]
#size of the game field
grid_size = 10
# number of obstacles
n_obstacles = 2
img_size = 110

max_epLength = 50

pred = []

################# show ################
def display_screen(action,points,input_t):
    #Function used to render the game screen
    #Get the last rendered frame
    global last_frame_time
    print("Action %s, Points: %d" % (translate_action[action],points))
    #Only display the game screen if the game is not over
    if("End" not in translate_action[action]):
        #Render the game with matplotlib
        plt.imshow(input_t.reshape((img_size,)*2),
               interpolation='none', cmap='gray')
	plt.pause(0.0001)
        #Clear whatever we rendered before
        display.clear_output(wait=True)
        #And display the rendering
        plt.gcf()
	plt.savefig('./test/%3d.png', last_frame_time)
    #Update the last frame time
    last_frame_time = set_max_fps(last_frame_time)
    
    
def set_max_fps(last_frame_time,FPS = 1):
    current_milli_time = lambda: int(round(time.time() * 1000))
    sleep_time = 1./FPS - (current_milli_time() - last_frame_time)
    if sleep_time > 0:
        time.sleep(sleep_time)
    return current_milli_time()

################ test ########################################

def test(path):
    #This function lets a pretrained model play the game to evaluate how well it is doing
    global last_frame_time
    plt.ion()
    # Define environment, game
    env = ball_run.Ball_run(n_obstacles)
    #c is a simple counter variable keeping track of how much we train
    c = 0
    #Reset the last frame time (we are starting from 0)
    last_frame_time = 0
    #Reset score
    points = 0
    #For training we are playing the game 10 times
    for e in range(10):
        loss = 0.
        #Reset the game
        env.reset()
        #The game is not over
        game_over = False
        # get initial input

        input_t = env.observe()
        #display_screen(3,points,input_t)
        c += 1
        while not game_over:
            #The learner is acting on the last observed game screen
            #input_t is a vector containing representing the game screen
            input_tm1 = input_t
            #Feed the learner the current status and get the expected rewards for different actions from it
            q = model.predict(input_tm1)
            #Select the action with the highest expected reward
            action = np.argmax(q[0])
            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            #Update our score
            points += reward
            display_screen(action,points,input_t)
            c += 1
	#create lists to contain total rewards and steps per episode

    sess.run(init)

    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess,ckpt.model_checkpoint_path)
    for i in range(10):
        episodeBuffer = experience_buffer()
        #Reset environment and get first new observation
        env.reset()
	s = env.observe()
        s = dqn.processState(s)
        d = False
        rAll = 0
        j = 0
        #The Q-Network
        while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j+=1
           
            a = sess.run(pred,feed_dict={mainQN.scalarInput:[s]})[0]
	    
	    s1,r,d = env.act(a)
           
            s1 = processState(s1)
            total_steps += 1
           
            rAll += r
            s = s1
            
            if d == True:

                break
        
        
        jList.append(j)
        rList.append(rAll)
        #Periodically save the model. 
        if i % 1000 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.ckpt')
            print("Saved Model")
        if len(rList) % 10 == 0:
            print(total_steps,np.mean(rList[-10:]), e)
    saver.save(sess,path+'/model-'+str(i)+'.ckpt')
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

########################################

path = './dqn'

sess = tf.Session()
saver = tf.train.Saver()
#ckpt = tf.train.get_checkpoint_state(path)
#saver.restore(sess,ckpt.model_checkpoint_path)

jList = []
rList = []
total_steps = 0

#saver.restore(sess, 'model/model.ckpt')

