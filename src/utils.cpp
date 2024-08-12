#include "../include/utils.hpp"

unsigned long long myCPUTimer()
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec);
}