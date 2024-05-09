#include "Comm.hpp"

#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <fcntl.h>  // File control
#include <termios.h> // Terminal control
#include <unistd.h>  // UNIX standard
#include <cstring>   // For string functions
#include <stdio.h>   // for sprintf and perror
#include <errno.h>

/*
 * Local function declarations
 */
static int open_uart(const char *port, int baud_rate);
static bool write_uart(int fd, const char *data);
static bool read_uart(int fd, char *buffer, int buffer_size);
static int uart_reader(int fd);

Comm::Comm(int commType){
	isCommInit = false;
	speedSetpoint = 0;
	lateralSetpoint = 50;

	if(commType == UART_COMM_TYPE){
		fd = open_uart("/dev/ttyTHS1", B115200);
		if (fd == -1)
		{
			perror("Comm: Failed to open uart.\n");
		}else{
            isCommInit = true;
        }
	}else if(fd == SOCKET_COMM_TYPE){
		fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd == -1)
		{
			perror("Comm: Failed to create socket.\n");
		}else{
            isCommInit = true;
            sockaddr_in serverAddress;
            serverAddress.sin_family = AF_INET;
            serverAddress.sin_port = htons(25001);
            serverAddress.sin_addr.s_addr = inet_addr("192.168.100.122");

            connect(fd, (struct sockaddr *)&serverAddress, sizeof(serverAddress));
        }
	}else{
		
	}
	this->commType = commType;
}

Comm::~Comm(){
	if(isCommInit) close(fd);
}

bool Comm::publishMessage(){
	char message[50];
	sprintf(message, "%d %d\n", speedSetpoint, lateralSetpoint);
	if(commType==SOCKET_COMM_TYPE){
		int ret = send(fd, message, strlen(message), 0);
		if (ret == -1)
		{
			return false;
		}
	}else if(commType==UART_COMM_TYPE){
		bool ret = write_uart(fd, message);
		if (ret == false)
		{
			return false;
		}
	}else{

	}
	return true;
}

/*
 * Local function definitions
 */

static int open_uart(const char *port, int baud_rate)
{
    int fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1)
    {
        perror("Failed to open UART");
        return -1;
    }

    struct termios options;
    tcgetattr(fd, &options);

    cfsetispeed(&options, baud_rate);
    cfsetospeed(&options, baud_rate);

    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~PARENB; // No parity
    options.c_cflag &= ~CSTOPB; // 1 stop bit
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8; // 8 bits

    tcsetattr(fd, TCSANOW, &options);

    return fd;
}

static bool write_uart(int fd, const char *data)
{
    int len = strlen(data);
    int n = write(fd, data, len);

    if (n < 0)
    {
        perror("Write failed");
        return false;
    }
    return true;
}

static bool read_uart(int fd, char *buffer, int buffer_size)
{
    int n = read(fd, buffer, buffer_size);
    if (n < 0)
    {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
        {
            return false;
        }
        else
        {
            perror("Read failed");
            return false;
        }
    }
    buffer[n] = '\0'; // Null-terminate the received data
    return true;
}

static int uart_reader(int fd)
{
    const int buffer_size = 256;
    char read_buffer[buffer_size];

    memset(read_buffer, 0, buffer_size);

    if (read_uart(fd, read_buffer, buffer_size - 1))
    {
        //std::cout << "Received data: " << read_buffer << std::endl;
        if(strcmp(read_buffer,"50")==0){
            return 50;
        }
    }
}