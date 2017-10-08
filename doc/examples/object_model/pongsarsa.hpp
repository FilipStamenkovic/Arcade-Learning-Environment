#ifndef __PONGSARSA_HPP__
#define __PONGSARSA_HPP__

#include <iostream>
#include <vector>
#include <ale_interface.hpp>
#include <ale_ram.hpp>

#define NUMBER_OF_ACTIONS 3
#define NUMBER_OF_FEATURES 4
#define TD_N 10
#define EPSILON 0.05

namespace object_model
{

  static const unsigned char addresses[] = {0xC6, 0xC8, 0xE3, 0xE5}; 
class PongSarsa
{
private:
  bool _updated;

  float weights[NUMBER_OF_ACTIONS][NUMBER_OF_FEATURES];
  float features[NUMBER_OF_FEATURES];
  float alpha;
  float discount;
  float history[TD_N][NUMBER_OF_FEATURES];
  float rewards[TD_N];
  int featureHistory[NUMBER_OF_FEATURES][256];
  int actions[TD_N];
  int historyIndex;

public:
  PongSarsa();

  Action GetAction();
  void UpdateWeights(float reward, Action chosenAction, bool isFinal);
  void StoreFeatures(int action, float reward)
  {
    for (int i = 0; i < NUMBER_OF_FEATURES; i++)
    {
      history[historyIndex][i] = features[i];
    }
    actions[historyIndex] = action;
    rewards[historyIndex] = reward;
    historyIndex++;
  };
  void ReadFeatures(ALERAM aleRam)
  {
    unsigned int ball_x_pos_addr = 0x31;
    unsigned int ball_y_pos_addr = 0x36;
    unsigned int ball_x_vel_addr = 0x3A;
    unsigned int ball_y_vel_addr = 0x38;
    unsigned int player_y_pos_addr = 0x3C;

    int ball_x_pos = (aleRam.get(ball_x_pos_addr));
    int ball_y_pos = (aleRam.get(ball_y_pos_addr));
    int ball_x_vel = (char)aleRam.get(ball_x_vel_addr);
    int ball_y_vel = (char)aleRam.get(ball_y_vel_addr);
    int player_y_pos = (aleRam.get(player_y_pos_addr));

    int hdiff = player_y_pos - ball_y_pos;
    
     features[3] = hdiff / 255.0f;

     features[1] = player_y_pos / 255000.0f;
     features[2] = ball_y_pos / 255000.0f;
  };
  void PrintWeights();
  void FlushToDisk(char *filename);
  void LoadFromDisk(char *filename);

private:
  void StoreHistory();
};
}

#endif // __PongSarsa_HPP__