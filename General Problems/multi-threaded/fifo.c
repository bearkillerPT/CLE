/**
 *  \file fifo.c (implementation file)
 *
 *  \brief Problem name: Producers / Consumers.
 *
 *  Synchronization based on monitors.
 *  Both threads and the monitor are implemented using the pthread library which enables the creation of a
 *  monitor of the Lampson / Redell type.
 *
 *  Data transfer region implemented as a monitor.
 *
 *  Definition of the operations carried out by the producers / consumers:
 *     \li putVal
 *     \li getVal.
 *
 *  \author António Rui Borges - March 2019
 */
#include "fifo.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>


#define K 1000

/** \brief flag which warrants that the data transfer region is initialized exactly once */
static pthread_once_t init = PTHREAD_ONCE_INIT;


/** \brief storage region */
static struct matrix_t* mem[K];

/** \brief flag signaling the data transfer region is full */
static bool full;


/** \brief insertion pointer */
static unsigned int ii;

/** \brief retrieval pointer */
static unsigned int ri;

static int current_count = 0;
/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief producers synchronization point when the data transfer region is full */
static pthread_cond_t fifoFull;

/** \brief consumers synchronization point when the data transfer region is empty */
static pthread_cond_t fifoEmpty;

/**
 *  \brief Initialization of the data transfer region.
 *
 *  Internal monitor operation.
 */

void initialization (void)
{
  full = false;                                                                                  /* FIFO is not full */
  pthread_cond_init (&fifoFull, NULL);                                 /* initialize producers synchronization point */
  pthread_cond_init (&fifoEmpty, NULL);                                /* initialize consumers synchronization point */
}

int isEmpty() {
  int res;
  pthread_mutex_lock(&accessCR);
  res = !full && (ri==ii);
  pthread_mutex_unlock(&accessCR);
  return res;
}
/**
 *  \brief Store a value in the data transfer region.
 *
 *  Operation carried out by the producers.
 *
 *  \param val value to be stored
 */

void putVal (struct matrix_t* val)
{
  int status = 0;
  if ((status = pthread_mutex_lock (&accessCR)) != 0)                                   /* enter monitor */
     { errno = status;                                                         /* save error in errno */
       perror ("error on entering monitor(CF)");
       status  = EXIT_FAILURE;
       pthread_exit (&status);
     }
  pthread_once (&init, initialization);                                              /* internal data initialization */

  while (full)                                                           /* wait if the data transfer region is full */
  { if ((status = pthread_cond_wait (&fifoFull, &accessCR)) != 0)
       { errno = status;                                                          /* save error in errno */
         perror ("error on waiting in fifoFull");
         status = EXIT_FAILURE;
         pthread_exit (&status);
       }
  }

  mem[ii] = val;                                                                          /* store value in the FIFO */
  ii = (ii + 1) % K;
  full = (ii == ri);
  if ((status = pthread_cond_signal (&fifoEmpty)) != 0)      /* let a consumer know that a value has been
                                                                                                               stored */
     { errno = status;                                                             /* save error in errno */
       perror ("error on signaling in fifoEmpty");
       status = EXIT_FAILURE;
       pthread_exit (&status);
     }

  if ((status = pthread_mutex_unlock (&accessCR)) != 0)                                  /* exit monitor */
     { errno =status;                                                            /* save error in errno */
       perror ("error on exiting monitor(CF)");
      status = EXIT_FAILURE;
       pthread_exit (&status);
     }
}

/**
 *  \brief Get a value from the data transfer region.
 *
 *  Operation carried out by the consumers.
 *
 *  \param consId consumer identification
 *
 *  \return value
 */

struct matrix_t* getVal ()
{
  struct matrix_t* val;                                                                               /* retrieved value */
  int status;
  
  if ((status = pthread_mutex_lock (&accessCR)) != 0)                                   /* enter monitor */
     { errno = status;                                                            /* save error in errno */
       perror ("error on entering monitor(CF)");
       status = EXIT_FAILURE;
       pthread_exit (&status);
     }
  pthread_once (&init, initialization);                                              /* internal data initialization */

  while ((ii == ri) && !full)                                           /* wait if the data transfer region is empty */
  { if ((status = pthread_cond_wait (&fifoEmpty, &accessCR)) != 0)
       { errno = status;                                                          /* save error in errno */
         perror ("error on waiting in fifoEmpty");
         status = EXIT_FAILURE;
         pthread_exit (&status);
       }
  }

  val = mem[ri];                                                                   /* retrieve a  value from the FIFO */
  ri = (ri + 1) % K;
  full = false;

  if ((status = pthread_cond_signal (&fifoFull)) != 0)       /* let a producer know that a value has been
                                                                                                            retrieved */
     { errno = status;                                                             /* save error in errno */
       perror ("error on signaling in fifoFull");
       status = EXIT_FAILURE;
       pthread_exit (&status);
     }

  if ((status = pthread_mutex_unlock (&accessCR)) != 0)                                   /* exit monitor */
     { errno = status;                                                             /* save error in errno */
       perror ("error on exiting monitor(CF)");
       status = EXIT_FAILURE;
       pthread_exit (&status);
     }

  return val;
}
