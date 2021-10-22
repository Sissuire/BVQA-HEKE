/*  transmitter_simulator, version 0.2
 *  Copyright(c) 2007-2008 Matteo Naccari
 *  All Rights Reserved.
 *
 *  The author is with the Dipartimento di Elettronica e Informazione,
 *  Politecnico di Milano, Italy.
 *  email: naccari@elet.polimi.it | matteo.naccari@polimi.it
 *
 *  Permission to use, copy, or modify this software and its documentation
 *  for educational and research purposes only and without fee is hereby
 *  granted, provided that this copyright notice and the original authors'
 *  names appear on all copies and supporting documentation. This program
 *  shall not be used, rewritten, or adapted as the basis of a commercial
 *  software or hardware product without first obtaining permission of the
 *  authors. The authors make no representations about the suitability of
 *  this software for any purpose. It is provided "as is" without express
 *  or implied warranty.
 *
*/

#ifndef H_SIMULATOR_
#define H_SIMULATOR_

#include <stdio.h>
#include <iostream>
#include "packet.h"
#include "parameters.h"

/*!
 *
 *	\brief
 *	The simulator class which models the bitstream transmission over an error prone channel
 *
 *	\author
 *	Matteo Naccari
*/

class Simulator{

private:
	Packet *packet;
	Parameters *param;
	FILE *fp_bitstream;     //!	Transmitted bitstream
	FILE *fp_tr_bitstream;  //!	Received bitstream
	ifstream fp_losspattern;
	string loss_pattern;
	int numchar;

public:
    Simulator(Parameters *p);  //!	Constructor with configuration parameters
	~Simulator();
	void Run_Simulator();	   //!	Method to simulate the bitstream transmission
};

#endif