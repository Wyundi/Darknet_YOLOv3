#ifndef SERIAL_H
#define SERIAL_H

#include "darknet.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

//设置波特率
static void set_baudrate (struct termios *opt, unsigned int baudrate);
// 设置数据位
static void set_data_bit (struct termios *opt, unsigned int databit);
// 设置校验位
static void set_parity (struct termios *opt, char parity);
// 设置停止位
static void set_stopbit (struct termios *opt, const char *stopbit);
// 串口设置
int set_port_attr (
              int fd,
              int  baudrate,          // B1200 B2400 B4800 B9600 .. B115200
              int  databit,           // 5, 6, 7, 8
              const char *stopbit,    //  "1", "1.5", "2"
              char parity,            // N(o), O(dd), E(ven)
              int vtime,
              int vmin );

#endif