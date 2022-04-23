/**
 *  \file fifo.h (interface file)
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

#ifndef FIFO_H
#define FIFO_H

/**
 *  \brief Store a value in the data transfer region.
 *
 *  Operation carried out by the producers.
 *
 *  \param val value to be stored
 */

typedef struct matrix_t
{
  int id;
  double **matrix;
  int size;
};

static void initialization(void);

extern void setDone(void);

extern void putVal(struct matrix_t *val);

/**
 *  \brief Get a value from the data transfer region.
 *
 *  Operation carried out by the consumers.
 *
 *  \param consId consumer identification
 *  \return value
 */

extern struct matrix_t *getVal();

#endif /* FIFO_H */
