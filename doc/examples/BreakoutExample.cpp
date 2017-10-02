/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare,
 *  Matthew Hausknecht, and the Reinforcement Learning and Artificial Intelligence 
 *  Laboratory
 * Released under the GNU General Public License; see License.txt for details. 
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  sharedLibraryInterfaceExample.cpp 
 *
 *  Sample code for running an agent with the shared library interface. 
 **************************************************************************** */

//lives: B9
//score: CD, CE
//brick addresses from: 80-A3
//paddle position: C6, C8
//ball info: E1, E3, E4, E5, E7, E8, E9, EB, EC, D2, D4, DB, DF, F9, FA

#include <iostream>
#include <ale_interface.hpp>
#include "object_model/sarsa.hpp"

#ifdef __USE_SDL
#include <SDL.h>
#endif

using namespace std;
using namespace object_model;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " rom_file" << std::endl;
        return 1;
    }

    ALEInterface ale;
    Sarsa sarsa;

    // Get & Set the desired settings
    ale.setInt("random_seed", 123);
    ale.setInt("frame_skip", 1);
    ale.setFloat("repeat_action_probability", 0.0f);

#ifdef __USE_SDL
    ale.setBool("display_screen", true);
    ale.setBool("sound", true);
#endif

    // Load the ROM file. (Also resets the system for new settings to
    // take effect.)
    ale.loadROM(argv[1]);
    if (argc > 2)
        sarsa.LoadFromDisk(argv[2]);

    // Get the vector of legal actions
    ActionVect legal_actions = ale.getLegalActionSet();

    // Play 10 episodes
    for (int episode = 0;; episode++)
    {
        float totalReward = 0;
        Action chosenAction = PLAYER_A_NOOP;
        float reward = 0;
        int lives = 5;
        int score = 0;
        if (argc > 2 && !(episode % 10))
            sarsa.FlushToDisk(argv[2]);
        while (!ale.game_over())
        {
            ale.act(PLAYER_A_FIRE);
            sarsa.ReadFeatures(ale.getRAM());

            sarsa.UpdateWeights(reward, (Action)(chosenAction / 2), ale.game_over());

            if (rand() / ((double)RAND_MAX) < EPSILON)
            {
                int index = rand() % NUMBER_OF_ACTIONS;
                chosenAction = (Action)(index * 2 + index % 2);
            }
            else
            {
                chosenAction = sarsa.GetAction();
               // if (chosenAction != 0)
               //     cout << "Eureka: " << chosenAction << endl;
            }

            // Apply the action and get the resulting reward
            reward = ale.act(chosenAction);

            int x = ale.getRAM().get(77);
            int y = ale.getRAM().get(76);
            int byte_val = ale.getRAM().get(57);
            int scr = 1 * (x & 0x000F) + 10 * ((x & 0x00F0) >> 4) + 100 * (y & 0x000F);

            if (byte_val != lives)
                reward = -1;
            else
                reward = (float)(scr - score);

            // if (reward != 0)
            // {
            //     cout << "scr: " << scr << ", score: " << score << ", lives: " << lives << ", byte_val: " << byte_val << endl;
            //     cout << "Reward: " << reward << ", Action = " << chosenAction << endl;
            // } //cout << "Main reward" << reward;

            lives = byte_val;
            score = scr;
            sarsa.StoreFeatures(chosenAction / 2, reward);
            totalReward += reward;
        }

        // sarsa.PrintWeights();
        if (totalReward > -5)
        {
            cout << "Episode " << episode << " ended with score: " << totalReward << endl;
            //if (totalReward > 0)
            //    sarsa.PrintWeights();
        }
        ale.reset_game();
    }

    return 0;
}
