#pragma once

#define NO_COMM_TYPE 0
#define UART_COMM_TYPE 1
#define SOCKET_COMM_TYPE 2

class Comm
{
    public:
    Comm(int commType);
    ~Comm();
    bool publishMessage();

    int speedSetpoint;
    int lateralSetpoint;

    private:
    bool isCommInit;            /* Used for closing */
    int fd;                     /* Socket/Uart file descriptor */
    int commType;               /* Communication type UART or SOCKET*/
};