import sys
import random
import requests
import numpy as np
import argparse
import gym
from gym_connect_four import ConnectFourEnv
import time

env: ConnectFourEnv = gym.make("ConnectFour-v0")

#SERVER_ADRESS = "http://localhost:8000/"
SERVER_ADRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["fi1231ka-s"] # TODO: fill this list with your stil-id's

def call_server(move):
   res = requests.post(SERVER_ADRESS + "move",
                       data={
                           "stil_id": STIL_ID,
                           "move": move, # -1 signals the system to start a new game. any running game is counted as a loss
                           "api_key": API_KEY,
                       })
   # For safety some respose checking is done here
   if res.status_code != 200:
      print("Server gave a bad response, error code={}".format(res.status_code))
      exit()
   if not res.json()['status']:
      print("Server returned a bad status. Return message: ")
      print(res.json()['msg'])
      exit()
   return res

def check_stats():
   res = requests.post(SERVER_ADRESS + "stats",
                       data={
                           "stil_id": STIL_ID,
                           "api_key": API_KEY,
                       })

   stats = res.json()
   return stats

"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""
def opponents_move(env):
   env.change_player() # change to oppoent
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   # TODO: Optional? change this to select actions with your policy too
   # that way you get way more interesting games, and you can see if starting
   # is enough to guarrantee a win
   action = student_move(env)
   # action = random.choice(list(avmoves))

   state, reward, done, _ = env.step(action)
   if done:
      if reward == 1: # reward is always in current players view
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done

def student_move(env):
   """
   TODO: Implement your min-max alpha-beta pruning algorithm here.
   Give it whatever input arguments you think are necessary
   (and change where it is called).
   The function should return a move from 0-6
   """
   depth = 5
   alpha = float('-inf')
   beta = float('inf')

   t = time.time()
   # return alpha_beta(env, depth, alpha, beta, True, 0, 0.9)[1]
   move = alpha_beta1(env, 0, alpha, beta, True, depth, 0)[1]
   print('Chose move', move, "took", time.time() - t)
   return move

def alpha_beta1(env: ConnectFourEnv, current_depth, alpha, beta, maximizing_player, max_depth, reward):
   if current_depth == max_depth or reward != 0:
      if maximizing_player:
         return -reward, 0
      else:
         return reward, 0
 
   # self.node_expanded += 1
 
   possible_moves = env.available_moves()
   best_value = float('-inf') if maximizing_player else float('inf')
   move_target = next(iter(env.available_moves()))
   for move in possible_moves:
      env_copy: ConnectFourEnv = gym.make("ConnectFour-v0")
      env_copy.reset(board = env.board)
      if not maximizing_player: env_copy.change_player()
      (_, reward, done, _) = env_copy.step(move)

      eval_child = alpha_beta1(env_copy, current_depth+1, alpha, beta, not maximizing_player, max_depth, reward)[0]

      if maximizing_player and best_value < eval_child:
            best_value = eval_child
            move_target = move
            alpha = max(alpha, best_value)
            if beta <= alpha:
               break

      elif (not maximizing_player) and best_value > eval_child:
            best_value = eval_child
            move_target = move
            beta = min(beta, best_value)
            if beta <= alpha:
               break

   return best_value, move_target

def alpha_beta(env: ConnectFourEnv, depth, alpha, beta, maximizing_player, reward, penalty_factor):
   terminal_node = reward != 0
   if(depth == 0 or terminal_node):
      print(env.board)
      if maximizing_player:
         print('reward', -reward, 'depth', depth)
         return -reward * penalty_factor, 0
      else:
         print('reward', reward, 'depth', depth)
         return reward * penalty_factor, 0

   env_copy: ConnectFourEnv = gym.make("ConnectFour-v0")
   env_copy.reset(board = env.board)
   
   if maximizing_player:
      max_value = float('-inf')
      best_move = next(iter(env.available_moves()))

      # print('1: maximizing player', env.available_moves())
      for move in env.available_moves():
         (_, reward, done, _) = env_copy.step(move)
         # print('1: testing move', move, 'got reward', reward, 'done', done)
         
         value = alpha_beta(env_copy, depth-1, alpha, beta, False, reward, penalty_factor**2)[0]
         if max_value < value:
            max_value = value
            # print('1: assigning best move for 1!', move)
            best_move = move
         print(''.join(['   ' for i in range(4-depth)]),"1: max_value", max_value, 'depth', depth, 'best_move', best_move, 'move', move)
         alpha = max(alpha, value)
         if beta <= alpha:
            break

      return max_value, best_move

   else:
      min_value = float('inf')
      best_move = next(iter(env.available_moves()))

      # print('-1:   minimizing player', env.available_moves())
      for move in env.available_moves():
         env_copy.change_player()
         (_, reward, done, _) = env_copy.step(move)
         # print('-1:   testing move', move, 'got reward', reward, 'done', done)
         
         value = alpha_beta(env_copy, depth-1, alpha, beta, True, reward, penalty_factor**2)[0]
         if min_value > value:
            min_value = value
            # print('-1:   assigning best move for -1!', move)
            best_move = move

         print(''.join(['   ' for i in range(4-depth)]), "-1: min_value", min_value, 'depth', depth, 'best_move', best_move, 'move', move)
         beta = min(beta, value)
         if beta <= alpha:
            break
      return min_value, move

def play_game(vs_server = False):
   """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """

   # default state
   # state = np.zeros((6, 7), dtype=int)
   state = np.array( 
      [
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [-1, 0, 0, 0, 0, 1, 0],
         [ 1, 1, 0, 0, 0, 1, 0],
         [ 1,-1, 0, 0, 0, 1, 0],
         [ 1, 1, 0, 0, 1,-1,-1]
      ]
    )
   # -1 signals the system to start a new game. any running game is counted as a loss
   res = call_server(-1)

   # setup new game
   if vs_server:
      # Start a new game
      res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss

      # This should tell you if you or the bot starts
      print(res.json()['msg'])
      botmove = res.json()['botmove']
      state = np.array(res.json()['state'])
      env.reset(board=state)
   else:
      # reset game to starting state
      # env.reset(board=None)
      env.reset(board=state)
      # determine first player
      # student_gets_move = random.choice([True, False])
      student_gets_move = True
      if student_gets_move:
         print('You start!')
         print()
      else:
         print('Bot starts!')
         print()

   # Print current gamestate
   print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
   print(state)
   print()

   done = False
   while not done:
      # Select your move
      stmove = student_move(env) # TODO: change input here

      # make both student and bot/server moves
      if vs_server:
         # Send your move to server and get response
         res = call_server(stmove)
         print(res.json()['msg'])

         # Extract response values
         result = res.json()['result']
         botmove = res.json()['botmove']
         state = np.array(res.json()['state'])
         env.reset(board=state)
      else:
         if student_gets_move:
            # Execute your move
            avmoves = env.available_moves()
            if stmove not in avmoves:
               print("You tied to make an illegal move! You have lost the game.")
               break
            state, result, done, _ = env.step(stmove)

         student_gets_move = True # student only skips move first turn if bot starts

         # print or render state here if you like

         # select and make a move for the opponent, returned reward from students view
         if not done:
            state, result, done = opponents_move(env)

      # Check if the game is over
      if result != 0:
         done = True
         if not vs_server:
            print("Game over. ", end="")
         if result == 1:
            print("You won!")
         elif result == 0.5:
            print("It's a draw!")
         elif result == -1:
            print("You lost!")
         elif result == -10:
            print("You made an illegal move and have lost!")
         else:
            print("Unexpected result result={}".format(result))
         if not vs_server:
            print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
      else:
         print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

      # Print current gamestate
      print(state)
      print()
      # break

def main():
   # Parse command line arguments
   parser = argparse.ArgumentParser()
   group = parser.add_mutually_exclusive_group()
   group.add_argument("-l", "--local", help = "Play locally", action="store_true")
   group.add_argument("-o", "--online", help = "Play online vs server", action="store_true")
   parser.add_argument("-s", "--stats", help = "Show your current online stats", action="store_true")
   args = parser.parse_args()

   # Print usage info if no arguments are given
   if len(sys.argv)==1:
      parser.print_help(sys.stderr)
      sys.exit(1)

   if args.local:
      play_game(vs_server = False)
   elif args.online:
      print('playing online')
      play_game(vs_server = True)

   if args.stats:
      stats = check_stats()
      print(stats)

   # TODO: Run program with "--online" when you are ready to play against the server
   # the results of your games there will be logged
   # you can check your stats bu running the program with "--stats"

if __name__ == "__main__":
    main()
