#pragma once

class Planning
{

public:
    Planning();
    ~Planning();
    void RunStateHandler();

private:
    enum State{
        WAIT,
        DRIVE,
        STOP
    };

    State state;
    int desired_speed;
    int desired_steering_angle;
};