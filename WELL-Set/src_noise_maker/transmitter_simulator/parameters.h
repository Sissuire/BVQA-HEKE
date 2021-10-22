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

#ifndef H_PARAMETERS_
#define H_PARAMETERS_

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

/*!
 *
 *	\brief
 *	It models the parameters related to the transmission conditions
 *
 *	\author
 *	Matteo Naccari
 *
*/

class Parameters{
	private:
		string bitstream_original, bitstream_transmitted, loss_pattern_file;
		int modality, offset, packet_type;
		bool valid_line(string line);		
	public:
		//! First constructor: the parameters are passed through command line
		Parameters(char **argv);

		//! Second constructor: the parameters are passed through configuration file
		Parameters(char *argv);

		~Parameters();

		string get_bitstream_original_filename(){return bitstream_original;}
		string get_bitstream_transmitted_filename(){return bitstream_transmitted;}
		string get_loss_pattern_filename(){return loss_pattern_file;}
		int get_modality(){return modality;}
		int get_offset(){return offset;}
		int get_packet_type(){return packet_type;}
		void check_parameters();
};

#endif