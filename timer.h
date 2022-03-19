#pragma once

#ifdef _WIN32

#include <Windows.h>
constexpr size_t TIMER_STARTNOW = 0;
typedef size_t T_time;
class timer
{
private:
    double timepasttotal;
    T_time freq;
    T_time tstart;
    bool running;
    bool supported;
    inline double gettimepast() const
    {
        T_time tend;
        QueryPerformanceCounter((LARGE_INTEGER*)& tend);
        return (tend - tstart) / (double(freq));
    }
public:
    inline explicit timer() :timepasttotal(0), running(false), tstart(0)
    {
        supported = QueryPerformanceFrequency((LARGE_INTEGER*)& freq);
    }
    inline explicit timer(size_t startOp) :timer()
    {
        switch (startOp)
        {
        case TIMER_STARTNOW:
            start();
            break;
        default:
            break;
        }
        
    }
    inline bool start()
    {
        if (!supported)
            return false;
        if (running)
            return false;
        running = true;
        QueryPerformanceCounter((LARGE_INTEGER*)& tstart);
        return true;
    }
    inline bool pause()
    {
        if (!supported)
            return false;
        if (!running)
            return false;
        timepasttotal += gettimepast();
        running = false;
        return true;
    }
    inline double gettime() const
    {
        if (!supported)
            return false;
        return timepasttotal + running ? gettimepast() : 0;
    }
    inline void clear()
    {
        running = false;
        timepasttotal = 0;
    }
    inline double restart()
    {
        auto ret=gettime();
        clear();
        start();
        return ret;
    }
};
#else //WIN32

#include <sys/time.h>
constexpr size_t TIMER_STARTNOW = 0;
typedef timeval T_time;
class timer
{
private:
    double timepasttotal;
    T_time tstart;
	double freq;
    bool running;
    bool supported;
    inline double gettimepast() const
    {
        T_time tend;
        gettimeofday(&tend,0);
		double usec=1000000*(tend.tv_sec-tstart.tv_sec)+tend.tv_usec-tstart.utv_sec;
        return usec / (double(freq));
    }
public:
    inline explicit timer() :timepasttotal(0), running(false), tstart(0)
    {
		freq=1000000;
        supported = true;
    }
    inline explicit timer(size_t startOp) :timer()
    {
        switch (startOp)
        {
        case TIMER_STARTNOW:
            start();
            break;
        default:
            break;
        }
        
    }
    inline bool start()
    {
        if (!supported)
            return false;
        if (running)
            return false;
        running = true;
        gettimeofday(&tstart,0);
        return true;
    }
    inline bool pause()
    {
        if (!supported)
            return false;
        if (!running)
            return false;
        timepasttotal += gettimepast();
        running = false;
        return true;
    }
    inline double gettime() const
    {
        if (!supported)
            return false;
        return timepasttotal + running ? gettimepast() : 0;
    }
    inline void clear()
    {
        running = false;
        timepasttotal = 0;
    }
    inline double restart()
    {
        auto ret=gettime();
        clear();
        start();
        return ret;
    }
};
#endif //WIN32