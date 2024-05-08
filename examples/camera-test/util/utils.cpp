#include "utils.h"

int open_uart(const char *port, int baud_rate)
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

bool write_uart(int fd, const char *data)
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

bool read_uart(int fd, char *buffer, int buffer_size)
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

bool sendMetaData(int fd, int laneCenter, std::vector<Yolo::Detection> res)
{
    char data[100];
    sprintf(data, "98 %d\n", laneCenter); //0 - 1024
    std::cout<<"Lane center: "<<laneCenter;
    bool ret = write_uart(fd, data);
    if (ret == false)
    {
        return false;
    }
    // class_id,class_confidence,det_confidence,x,y,width,height
    for (int i = 0; i < res.size(); i++)
    {
        int class_id = (int)res[i].class_id;
        int class_confidence = (int)(res[i].class_confidence * 99);
        int det_confidence = (int)(res[i].det_confidence * 99);
        int x = (int)res[i].bbox[0];
        int y = (int)res[i].bbox[1];
        int width = (int)res[i].bbox[2];
        int height = (int)res[i].bbox[3];
        if(class_confidence * det_confidence >0.7)
        sprintf(data, "99 %02d %02d %02d %04d %04d %04d %04d\n", class_id, class_confidence, det_confidence, x, y, width, height);
        ret = write_uart(fd, data);
        if (ret == false)
        {
            return false;
        }
        std::cout << " Class_id: " << class_id;
    }
    std::cout<<std::endl;
    return true;
}
int uart_reader(int fd)
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
bool getParkCommand(int fd){
    if(uart_reader(fd)==50){
        return true;
    }
    return false;
}
bool getParkingDoneCommand(int fd){
    if(uart_reader(fd)==51){
        return true;
    }
    return false;
}