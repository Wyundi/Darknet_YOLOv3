#include "serial.h"

//设置波特率
static void set_baudrate (struct termios *opt, unsigned int baudrate)
{
	cfsetispeed(opt, baudrate);
	cfsetospeed(opt, baudrate);
}

// 设置数据位
static void set_data_bit (struct termios *opt, unsigned int databit)
{
    opt->c_cflag &= ~CSIZE;
    switch (databit) {
    case 8:
        opt->c_cflag |= CS8;
        break;
    case 7:
        opt->c_cflag |= CS7;
        break;
    case 6:
        opt->c_cflag |= CS6;
        break;
    case 5:
        opt->c_cflag |= CS5;
        break;
    default:
        opt->c_cflag |= CS8;
        break;
    }
}

// 设置校验位
static void set_parity (struct termios *opt, char parity)
{
    switch (parity) {
    case 'N':                  /* no parity check */
        opt->c_cflag &= ~PARENB;
        break;
    case 'E':                  /* even */
        opt->c_cflag |= PARENB;
        opt->c_cflag &= ~PARODD;
        break;
    case 'O':                  /* odd */
        opt->c_cflag |= PARENB;
        opt->c_cflag |= ~PARODD;
        break;
    default:                   /* no parity check */
        opt->c_cflag &= ~PARENB;
        break;
    }
}

// 设置停止位
static void set_stopbit (struct termios *opt, const char *stopbit)
{
    if (0 == strcmp (stopbit, "1")) {
        opt->c_cflag &= ~CSTOPB; /* 1 stop bit */
    }	else if (0 == strcmp (stopbit, "1")) {
        opt->c_cflag &= ~CSTOPB; /* 1.5 stop bit */
    }   else if (0 == strcmp (stopbit, "2")) {
        opt->c_cflag |= CSTOPB;  /* 2 stop bits */
    } else {
        opt->c_cflag &= ~CSTOPB; /* 1 stop bit */
    }
}

// 串口设置
int set_port_attr (
              int fd,
              int  baudrate,          // B1200 B2400 B4800 B9600 .. B115200
              int  databit,           // 5, 6, 7, 8
              const char *stopbit,    //  "1", "1.5", "2"
              char parity,            // N(o), O(dd), E(ven)
              int vtime,
              int vmin )
{
    struct termios opt;
    tcgetattr(fd, &opt);
    //设置波特率
    set_baudrate(&opt, baudrate);
    opt.c_cflag 		 |= CLOCAL | CREAD;      /* | CRTSCTS */
    //设置数据位
    set_data_bit(&opt, databit);
    //设置校验位
    set_parity(&opt, parity);
    //设置停止位
    set_stopbit(&opt, stopbit);
    //其它设置
    opt.c_oflag 		     = 0;
    opt.c_lflag            	|= 0;
    opt.c_oflag          	&= ~OPOST;
    opt.c_cc[VTIME]     	 = vtime;
    opt.c_cc[VMIN]         	 = vmin;
    tcflush (fd, TCIFLUSH);
    return (tcsetattr (fd, TCSANOW, &opt));
}

int open_serial()
{
    #define DEV_NAME  "/dev/ttyUSB0"

    int fd = open(DEV_NAME, O_RDWR | O_NOCTTY);
    if(fd < 0) {
            perror(DEV_NAME);
            return -1;
    }

    set_port_attr(fd, B115200, 8, "2", 'N', 1, 0.5);

    return fd;
}
