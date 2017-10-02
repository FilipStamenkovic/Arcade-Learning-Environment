#include <iostream>
#include <vector>
#include <limits>
#include "sarsa.hpp"

using namespace std;

namespace object_model
{
Sarsa::Sarsa()
{
    for (int i = 0; i < NUMBER_OF_FEATURES; i++)
    {
        features[i] = 1.0;
        for (int j = 0; j < NUMBER_OF_ACTIONS; j++)
        {
            weights[j][i] = 0.0;
        }

        for (int j = 0; j < 256; j++)
        {
            featureHistory[i][j] = 0;
        }
    }

    for (int i = 0; i < TD_N; i++)
    {
        actions[i] = 0;
        rewards[i] = 0.0;
        for (int j = 0; j < NUMBER_OF_FEATURES; j++)
        {
            history[i][j] = 0.0;
        }
    }

    historyIndex = 0;
    alpha = 0.01;
    discount = 0.9;
}
void Sarsa::ClearHistory()
{
    _updated = false;
}

bool Sarsa::EvaluateAndImprovePolicy(double reward, bool isFinal)
{
    // int numOfPrevSteps = _history.size();
    // pair<State, Action> currStateAction = _history[numOfPrevSteps - 1];
    // Action currAction = (Action)currStateAction.second;
    // State currState = (State)currStateAction.first;
    // int currStateIndex = currState.ToIndex();
    // int currQIndex = currAction * NUMBER_OF_STATES + currStateIndex;
    // Action previousAction;

    // if (numOfPrevSteps > TD_N && !isFinal)
    // {
    //     double gama = 1;
    //     int currentIndex = currQIndex;
    //     for (int i = 0; i < TD_N; i++)
    //     {
    //         gama *= discount;
    //         int index = numOfPrevSteps - 2 - i;

    //         currentIndex = EvalueTD(reward, index, currentIndex, gama);
    //     }
    // }
    // else if (isFinal)
    // {
    //     //eval
    //     q[currQIndex] = q[currQIndex] + alpha * (reward - q[currQIndex]);
    //     //improve
    //     previousAction = _actions[currStateIndex];
    //     _updated = (previousAction != (_actions[currStateIndex] = q[currStateIndex] > q[currStateIndex + NUMBER_OF_STATES] ? NOOP : JUMP));
    // }

    // return _updated;
    return true;
}

int Sarsa::EvalueTD(double reward, int index, int currentIndex, double gama)
{
    // pair<State, Action> prevStateAction = _history[index];
    // Action prevAction = (Action)prevStateAction.second;
    // State prevState = (State)prevStateAction.first;
    // int stateIndex = prevState.ToIndex();
    // int prevQIndex = prevAction * NUMBER_OF_STATES + stateIndex;

    // //eval
    // q[prevQIndex] = q[prevQIndex] + alpha * ((reward + gama * q[currentIndex]) - q[prevQIndex]);

    // //improve
    // Action previousAction = _actions[stateIndex];
    // _updated = (previousAction != (_actions[stateIndex] = q[stateIndex] > q[stateIndex + NUMBER_OF_STATES] ? NOOP : JUMP));

    // return prevQIndex;
    return 0;
}

Action Sarsa::GetAction()
{
    float maxQ = -std::numeric_limits<float>::max();
    int maxIndex = 0;

    for (int i = 0; i < NUMBER_OF_ACTIONS; ++i)
    {
        float q = 0.0f;
        for (int of = 0; of < NUMBER_OF_FEATURES; ++of)
        {
            q += weights[i][of] * features[of];
        }

        //cout << "Q = " << q << ", i = " << i << endl;
        if (q >= maxQ)
        {
            maxQ = q;
            maxIndex = i;
        }
    }

    return (Action)(maxIndex * 2 + maxIndex % 2);
}
void Sarsa::UpdateWeights(float reward, Action chosenAction, bool isFinal)
{
    //todo: avoid double calculations like this!
    if (historyIndex < TD_N)
        return;

    float q = 0.0f;

    historyIndex = 0;

    //PrintWeights();
    for (int of = 0; of < NUMBER_OF_FEATURES; ++of)
    {
        if (weights[chosenAction][of] != 0.0)
        {
            //std::cout << "baasd = " << chosenAction << ", q = " << q << ", Reward = " << reward << std::endl;
        }
        q += weights[chosenAction][of] * features[of];
    }
   // cout << "QQQ = " << q << ",  Action = " << chosenAction << endl;
    float diff = 0.0f;

    if (isFinal)
    {
        diff = alpha * (reward - q);

        //if (reward != 0.0)
        //    std::cout << "aAction = " << chosenAction << ", Diff = " << diff << ", Reward = " << reward << std::endl;
        for (int i = 0; i < NUMBER_OF_FEATURES; ++i)
        {
            weights[chosenAction][i] += diff * features[i];
            if (weights[chosenAction][i] != 0.0)
            {
                //std::cout << "aaaAction = " << chosenAction << ", Diff = " << diff << ", Reward = " << reward << std::endl;
            }
            // if (weights[chosenAction][i] < -10000)
            //     weights[chosenAction][i] = -10000;
            // else if (weights[chosenAction][i] > 10000)
            //     weights[chosenAction][i] = 10000;
        }
        historyIndex = 0;
    }
    else
    {
        //todo: avoid double calculations like this!
        float gama = 1;
        float q_ = 0.0f;
        for (int of = 0; of < NUMBER_OF_FEATURES; ++of)
        {
            q_ += weights[chosenAction][of] * features[of];
        }

        //cout << "Q_ = " << q_ << endl;
        for (int n = TD_N - 1; n >= 0; n--)
        {
            gama = gama * discount;
            int action = actions[n];

            q = 0.0;

            for (int of = 0; of < NUMBER_OF_FEATURES; ++of)
            {
                q += weights[action][of] * features[of];
            }

            diff = alpha * (reward + gama * q_ - q);

            //if (reward != 0.0)
            //    std::cout << "Action = " << action << ", Diff = " << diff << ", Reward = " << reward << std::endl;
            reward += rewards[n];

            for (int i = 0; i < NUMBER_OF_FEATURES; ++i)
            {
                weights[action][i] += diff * history[n][i];
                if (weights[action][i] != 0.0)
                {
                    // std::cout << "Action = " << action << ", Diff = " << diff << ", Reward = " << reward << std::endl;
                }

                // if (weights[action][i] < -10000)
                //     weights[action][i] = -10000;
                // else if (weights[action][i] > 10000)
                //     weights[action][i] = 10000;

                if (std::isnan(weights[action][i]))
                {
                    std::cout << "History: " << history[n][i] << std::endl;
                }
            }
        }
    }
}

void Sarsa::StoreHistory()
{
}

void Sarsa::PrintWeights()
{
    for (int a = 0; a < NUMBER_OF_ACTIONS; ++a)
    {
        std::cout << "Weights [" << a << "] are: [" << weights[a][0];

        for (int i = 1; i < NUMBER_OF_FEATURES; ++i)
        {
            std::cout << ", " << weights[a][i];
        }
        std::cout << "]" << std::endl;
    }
}

void Sarsa::FlushToDisk(char *filename)
{
    if (std::isnan(weights[0][0]))
        return;
    ofstream f(filename);

    if (!f)
        return;

    ostringstream o;
    for (int i = 0; i < NUMBER_OF_ACTIONS; i++)
    {
        for (int j = 0; j < NUMBER_OF_FEATURES; ++j)
        {
            o << weights[i][j] << endl;
        }
    }
    f << o.str();

    f.close();
}
void Sarsa::LoadFromDisk(char *filename)
{
    ifstream f(filename);
    if (!f || !f.is_open())
        return;

    int i = 0;
    int j = 0;
    while (!f.eof() && i < NUMBER_OF_FEATURES * NUMBER_OF_ACTIONS)
    {
        string line;

        getline(f, line);

        double w = atof(line.c_str());

        weights[i][j++] = (float)w;

        if (j == NUMBER_OF_FEATURES)
        {
            j = 0;
            ++i;
        }
    }

    f.close();
}
}
