#ifndef __SARSA_HPP__
#define __SARSA_HPP__

#include <iostream>
#include <vector>
#include <ale_interface.hpp>
#include <ale_ram.hpp>

#define NUMBER_OF_ACTIONS 3
#define NUMBER_OF_FEATURES 46
#define TD_N 10

//score: CD, CE
//brick addresses from: 80-A3
//paddle position: C6, C8
//ball info: E3, E5, E7, E9

using namespace std;

namespace object_model
{
static const unsigned char addresses[] = {0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F,
                                          0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F,
                                          0xA0, 0xA1, 0xA2, 0xA3, 0xC6, 0xC8, 0xE1, 0xE3, 0xE5};

static const unsigned char ballSpeed[] = {0xE7, 0xE9};
static const unsigned char paddleWidth = 0xEC;
class Sarsa
{
private:
  double weights[NUMBER_OF_ACTIONS][NUMBER_OF_FEATURES];
  double features[NUMBER_OF_FEATURES];
  double alpha;
  double discount;
  double history[TD_N][NUMBER_OF_FEATURES];
  double rewards[TD_N];
  int featureHistory[NUMBER_OF_FEATURES][256];
  int actions[TD_N];
  int historyIndex;

public:
  Sarsa();

  Action GetAction();
  void UpdateWeights(double reward, Action chosenAction, bool isFinal);
  void StoreFeatures(int action, double reward)
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
    unsigned char romVal = 0;
    int i = 0;
    for (; i < NUMBER_OF_FEATURES - 5; i++)
    {
      romVal = aleRam.get(addresses[i]);
      featureHistory[i][romVal]++;
      features[i] = romVal / 255000.0f;
    }

    char paddleWidthVal = (char) aleRam.get(paddleWidth);
    char speedX = (char)aleRam.get(ballSpeed[0]);
    char speedY = (char)aleRam.get(ballSpeed[1]);

    features[i++] = speedX / 1000.0f;
    features[i++] = speedY / 1000.0f;

    int player_y_pos = aleRam.get(0xC8);
    int ball_y_pos = aleRam.get(0xE3) - paddleWidthVal;
    int hdiff = player_y_pos - ball_y_pos;

    features[i++] = paddleWidthVal / 2000.0f;
    features[i] = hdiff / 255.0f;
  };
  void PrintWeights();
  void FlushToDisk(char *filename);
  void LoadFromDisk(char *filename);

private:
  void StoreHistory();
};
}

#endif // __SARSA_HPP__