import time
import numpy as np
import random
import tkinter as tk
from PIL import ImageTk, Image

np.random.seed(7)
random.seed(7)

PhotoImage = ImageTk.PhotoImage
UNIT = 100  # pixels
HEIGHT = 6  # grid height
WIDTH = 6  # grid width

REWARD = 100
PENALTY = -100


class Env(tk.Tk):
    def __init__(self, noise_level=0.0):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Práctica 1.- Q Learning')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT))
        (self.up, self.down, self.left, self.right), self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.noise = noise_level
        self.texts = []
        self.arrows = []

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',height=HEIGHT * UNIT,width=WIDTH * UNIT)
        
        # Create grid
        for c in range(0, WIDTH * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, WIDTH * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        # Images to be displayed
        
        x_start, y_start = 0, 0       
        self.agent_position = canvas.create_image(x_start*UNIT + UNIT*0.5, y_start*UNIT + UNIT*0.5, image=self.shapes[0])

        x_penalty_a, y_penalty_a = 0, 2
        self.penalty_a = canvas.create_image(x_penalty_a*UNIT + UNIT*0.5, y_penalty_a*UNIT + UNIT*0.5, image=self.shapes[1])

        #x_penalty_b, y_penalty_b = 2, 2
        #self.penalty_b = canvas.create_image(x_penalty_b*UNIT + UNIT*0.5, y_penalty_b*UNIT + UNIT*0.5, image=self.shapes[1])

        x_reward, y_reward = 1, 4       
        self.gem = canvas.create_image(x_reward*UNIT + UNIT*0.5, y_reward*UNIT + UNIT*0.5, image=self.shapes[2])
        
        # Set rewards and penalty states
        
        self.rewards = [canvas.coords(self.gem)]
        #self.penalties = [canvas.coords(self.penalty_a), canvas.coords(self.penalty_b)]
        self.penalties = [canvas.coords(self.penalty_a)]
        
        self.rewards_pos = [[x_reward, y_reward]]
        #self.penalties_pos = [[x_penalty_a, y_penalty_a],[x_penalty_b, y_penalty_b]]
        self.penalties_pos = [[x_penalty_a, y_penalty_a]]

        # pack all
        canvas.pack()

        return canvas

    def load_images(self):
    
        up = PhotoImage(Image.open("img/up.png").resize((int(UNIT*0.13), int(UNIT*0.13))))
        right = PhotoImage(Image.open("img/right.png").resize((int(UNIT*0.13), int(UNIT*0.13))))
        left = PhotoImage(Image.open("img/left.png").resize((int(UNIT*0.13), int(UNIT*0.13))))
        down = PhotoImage(Image.open("img/down.png").resize((int(UNIT*0.13), int(UNIT*0.13))))
        
        bot = PhotoImage(Image.open("img/bot.png").resize((int(UNIT*0.65), int(UNIT*0.65))))
        wrong = PhotoImage(Image.open("img/wrong.png").resize((int(UNIT*0.65), int(UNIT*0.65))))
        gem = PhotoImage(Image.open("img/gem.png").resize((int(UNIT*0.65), int(UNIT*0.65))))

        return (up, down, left, right), (bot, wrong, gem)

    def text_value(self, row, col, contents, action, font='Helvetica', size=int(UNIT*0.065), style='normal', anchor="nw"):
    
        # Method to print values for tables

        if action == 0:
            origin_x, origin_y = int(UNIT*0.07), int(UNIT*0.42)
        elif action == 1:
            origin_x, origin_y = int(UNIT*0.85), int(UNIT*0.42)
        elif action == 2:
            origin_x, origin_y = int(UNIT*0.42), int(UNIT*0.05)
        else:
            origin_x, origin_y = int(UNIT*0.42), int(UNIT*0.75)

        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)        
        text = self.canvas.create_text(x, y, fill="black", text=contents, font=font, anchor=anchor)
        return self.texts.append(text)

    def print_value_all(self, q_table):
    
        # Print all values from q_tables
            
        for i in self.texts:
            self.canvas.delete(i)
        self.texts.clear()
        for i in range(WIDTH):
            for j in range(HEIGHT):
                for action in range(0, 4):
                    state = [i, j]                    
                    if state not in self.rewards_pos and state not in self.penalties_pos:
                        if str(state) in q_table.keys():
                            temp = q_table[str(state)][action]
                            self.text_value(j, i, round(temp, 2), action)
                        
    def print_policy_all(self, agent):
    
        # Print best policy for every state
            
        for i in self.texts:
            self.canvas.delete(i)
        self.texts.clear()
        
        for i in range(WIDTH):
            for j in range(HEIGHT):
                state = [i, j]
                if state not in self.rewards_pos and state not in self.penalties_pos:                    
                    if str(state) in agent.q_table.keys():
                        best_action, full_draw = agent.arg_max(agent.q_table[str(state)])
                        if not full_draw:
                            self.draw_one_arrow(i, j, best_action)
                        else:
                            for action in range(0, 4):
                                self.draw_one_arrow(i, j, action)
        self.render()
                        
    def draw_one_arrow(self, row, col, action):

        # Method for printing arrows and therefore display policies

        if action == 0:  # up
            origin_x, origin_y = UNIT*0.5 + (UNIT * row), UNIT*0.1 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,image=self.up))
        if action == 1:  # down
            origin_x, origin_y = UNIT*0.5 + (UNIT * row), UNIT*0.9 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,image=self.down))
        if action == 2:  # left
            origin_x, origin_y = UNIT*0.1 + (UNIT * row), UNIT*0.5 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.left))
        if action == 3:  # right
            origin_x, origin_y = UNIT*0.9 + (UNIT * row), UNIT*0.5 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.right))  

    def coords_to_state(self, coords):
        x = int((coords[0] - UNIT*0.5) / UNIT)
        y = int((coords[1] - UNIT*0.5) / UNIT)
        return [x, y]

    def state_to_coords(self, state):
        x = int(state[0] * UNIT + UNIT*0.5)
        y = int(state[1] * UNIT + UNIT*0.5)
        return [x, y]

    def reset(self, do_render=True):
        # Reset display and move agent to starting posicion
        self.update()
        for i in self.arrows:
                self.canvas.delete(i)
        x, y = self.canvas.coords(self.agent_position)
        self.canvas.move(self.agent_position, UNIT / 2 - x, UNIT / 2 - y)
        if do_render:
            self.render()
        
        # return observation
        return self.coords_to_state(self.canvas.coords(self.agent_position))


    def step(self, action, do_render):
    
        # Apply a step based on action and current agent position
        
        state = self.canvas.coords(self.agent_position)
        base_action = np.array([0, 0])
        if do_render:
            self.render()
        
        real_action = action
        
        # Consider noise 
        # a) --> up/down can be transformed into left/right
        
        if real_action in [0,1]:
            selection = random.choices([real_action,2,3],weights=(1-self.noise,self.noise*0.5,self.noise*0.5), k=1)	
            real_action = selection[0]
        
        # b) --> left/right can be transformed into up/down
            
        if real_action in [2,3]:
            selection = random.choices([real_action,0,1],weights=(1-self.noise,self.noise*0.5,self.noise*0.5), k=1)	
            real_action = selection[0]
        
        # Perform the real action, which can differ from the parameter used in the invocation

        if real_action == 0:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif real_action == 1:  # down
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif real_action == 2:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif real_action == 3:  # right
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT

        # Display agent in destination position
        self.canvas.move(self.agent_position, base_action[0], base_action[1])
        
        # Set agent in top level of canvas
        self.canvas.tag_raise(self.agent_position)
        
        next_state = self.canvas.coords(self.agent_position)

        # Reward function and identify it is a final state
        if next_state in self.rewards:
            reward = REWARD
            done = True
        elif next_state in self.penalties:
            reward = PENALTY
            done = True
        else:
            reward = 0
            done = False

        next_state = self.coords_to_state(next_state)
        return next_state, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()
