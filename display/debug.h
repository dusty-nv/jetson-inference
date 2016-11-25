#ifndef __DEBUG_H__
#define __DEBUG_H__
#define DEBUG 0

#if !DEBUG
#define debug_print(fmt, ...) 
#else

#if 1
#define debug_print(fmt, args...) \
    fprintf(stderr, "%s:%d:%s(): " fmt, __FILE__,  __LINE__, __func__, ##args) 
#endif
#endif
#endif
