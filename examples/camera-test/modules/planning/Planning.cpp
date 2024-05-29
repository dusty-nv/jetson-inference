#include "Planning.hpp"

Planning::Planning()
{
    state = WAIT;
}

void Planning::RunStateHandler()
{
    switch (state)
    {
    case WAIT:
        desired_speed = 0;
        desired_steering_angle = 0;
        break;
    case DRIVE:
        break;
    case STOP:
        break;
    }
}
