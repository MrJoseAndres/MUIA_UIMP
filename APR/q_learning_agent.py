import time
import numpy as np
from optparse import OptionParser
from environment import Env
from collections import defaultdict

np.random.seed(7)

DEF_Q_V = 0.0

class QLearningAgent:
    def __init__(self, actions=[0,1,2,3]):
        self.actions = actions
        self.learning_rate = 0.25
        self.discount_factor = 0.9
        self.epsilon = 0.15        	# TODO --> SE MODIFICA PARA LA PREGUNTA 1E y 2B
        self.q_table = defaultdict(lambda: [DEF_Q_V, DEF_Q_V, DEF_Q_V, DEF_Q_V])

        # La variable q_table es una estructura de tipo diccionario, donde la clave es una entrada del tipo [x, y]
        # y cada valor es un array con cuatro valores, una por acción (up,down,left,right). El valor por defecto
        # de estas entradas es DEF_Q_V (inicialmente a 0)
        
    def list_q_values(self):             # ESTE MÉTODO NOS MUESTRA COMO ACCEDER A LA Q TABLE
        for key, value in self.q_table.items():
            print(f"Los valores del estado {key} son: {value}")

    # update q function with sample <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        # using Bellman Optimality Equation to update q function
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)


    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # take random action
            action = np.random.choice(self.actions)
        else:
            # take action according to the q function table
            state_action = self.q_table[state]
            action, _ = self.arg_max(state_action)
        return action

    @staticmethod
    #  Select the action with highest entry in q-table, and in case of draw, 
    # it performsn a random choice between the array of best actions     
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return np.random.choice(max_index_list), len(max_index_list)==4
        

if __name__ == "__main__":
    parser = OptionParser(description='Aprender por refuerzo mediante Q Learning')    
    parser.add_option('-e', '--numEpisodes', dest='numEpisodes', type='int', help='Número de episodios a realizar (50 por defecto)', default=50)
    parser.add_option('-n', '--noiseLevel', dest='noise', type='float', help='Probabilidad de que el movimiento no sea el realizado, en tanto por uno (0.0 por defecto)', default=0.0)
    parser.add_option('-s', '--skip', dest='skip', action='store_true', help='Saltar la visualización durante el aprendizaje',default=False)

    args, _ = parser.parse_args()
    MOSTRAR_PROCESO = not args.skip
    NUM_EPISODIOS = args.numEpisodes
    NOISE_LEVEL = args.noise
                
    env = Env(NOISE_LEVEL)
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    for episode in range(NUM_EPISODIOS):
        state = env.reset(MOSTRAR_PROCESO)  
        
        ### INICIO DE VARIABLES ÚTILES PARA MODIFICACION 1A ###
        prev_q_table = agent.q_table.copy()	# Guardar la q-table antes de aprender
        converged = False
        ### FIN DE VARIABLES ÚTILES PARA MODIFICACION 1A ###
        while True:            
            # 1.- Get the action based on QTable
               # Epsilon will determine the exploration probabily, and in this case, the action is random
            action = agent.get_action(str(state))
            
            # 2.- Perform the selected action and get the new state and reward
               # If noise is considered, the action to be performed can differ from the selected one
            next_state, reward, done = env.step(action, MOSTRAR_PROCESO)

	    # 3.- Learn from the experience using the formular
	       # with sample <s,a,r,s'>, agent learns new q function
	       # We can learn from a experience comming from the application of an action different from the selected one
	    	
            agent.learn(str(state), action, reward, str(next_state))

            # 4.- Set the new state as current one
            state = next_state
            
            # 5.- Print policies in the screen
            if MOSTRAR_PROCESO:
                env.print_value_all(agent.q_table)
            
            # if episode ends, then break
            if done:
                break
        if MOSTRAR_PROCESO:        
            env.print_policy_all(agent)
            time.sleep(0.5)
                
        ### INICIO DE MODIFICACION 1A ###
        if not converged:
        
             ### AQUÍ SE DEBE INTEGRAR LA LÓGICA QUE COMPRUEBE QUE SE HAN CONVERGIDO ##
             
             if converged:
                 print(f"El algoritmo ha convergido tras {episode} episodios")
                 converged = True
             converged = False
        ### FINAL DE MODIFICACION 1A ###                
      
    env.print_policy_all(agent)
    guess = input("Pulsa cualquier tecla para salir...")
    exit(0)