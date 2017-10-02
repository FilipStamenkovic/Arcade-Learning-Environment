#ifndef __SARSA_HPP__
#define __SARSA_HPP__

#include <iostream>
#include <vector>
#include <ale_interface.hpp>
#include <ale_ram.hpp>

#define NUMBER_OF_ACTIONS 3
#define NUMBER_OF_FEATURES 54
#define TD_N 2
#define EPSILON 0.05

namespace object_model
{
static const unsigned char addresses[] = {0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F,
                                          0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F,
                                          0xA0, 0xA1, 0xA2, 0xA3, 0xC6, 0xC8, 0xE1, 0xE3, 0xE4, 0xE5, 0xE7, 0xE8, 0xE9, 0xEB, 0xEC, 0xD2,
                                          0xD4, 0xDB, 0xDF, 0xF9, 0xFA};
class Sarsa
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
  Sarsa();
  void ClearHistory();

  bool EvaluateAndImprovePolicy(double reward, bool isFinal);
  
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
    for (int i = 0; i < NUMBER_OF_FEATURES - 1; i++)
    {
      unsigned char romVal = aleRam.get(addresses[i]);
      featureHistory[i][romVal]++;
      features[i + 1] = romVal / 255.0f;
      // std::cout << (int)romVal << ", ";
    }
    // std::cout << std::endl;
  };
  void PrintWeights();
  void FlushToDisk(char *filename);
  void LoadFromDisk(char *filename);

private:
  int EvalueTD(double reward, int index, int prevIndex, double gama);
  void StoreHistory();
};
}

#endif // __SARSA_HPP__