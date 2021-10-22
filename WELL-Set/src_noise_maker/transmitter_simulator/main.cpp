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

#include "parameters.h"
#include "simulator.h"
#include <iostream>
#include <fstream>

#define VERSION 0.2

void inline_help();

/*!
 *  \brief
 *	The main function, i.e. the entry point of the program
 *  Note:
 *      The error patter file must contain only '0' and '1' ASCII
 *      characters (8 bits). The character '0' means that no channel error
 *      occurred whilst the character '1' means that a channel error
 *      occurred. A burst of channel errors is defined as a contiguous sequence of 2
 *      or more characters '1'.
 *
 *  \author
 *  Matteo Naccari
 *
*/

/* pa_768x432_H264_crf23.264 pa_768x432_H264_crf23_plr5_1.264 1 0-offset 0-frames */
int main(int argc, char **argv){
	Parameters *p;
	Simulator *sim;	

	if(argc == 2){		
		p = new Parameters(argv[1]);
	}
	else if(argc == 7)
		p = new Parameters(argv);
	else{
		inline_help();
                return 0;
        }
	sim = new Simulator(p);

	sim->Run_Simulator();
	delete p;
	delete sim;
        
        return 0;
}

/*!
 *  \brief
 *	It prints on the screen a little help on the program usage
 *
 *  \author
 *  Matteo Naccari
 *
*/

void inline_help(){

	cout<<endl<<endl<<"\tTransmitter Simulator version "<<VERSION<<"\n\n";
	cout<<"\tMatteo Naccari ISPG Lab. (Politecnico di Milano)"<<endl;
	cout<<"\tnaccari@elet.polimi.it"<<endl<<endl;
	cout<<"\tUsage (1): transmitter_simulator <in_bitstream> <out_bitstream> <loss_pattern_file> <packet_type> <offset> <modality>\n\n";
	cout<<"\t           transmitter_simulator xxxx.264 xxxx.264 errp_plr_0.4       1 (AnnexB)     RAND_INT  0(all) / 1(ex-I) / 2(only-I)\n\n";
	cout<<"\tUsage (2): transmitter_simulator <configuration_file>\n\n";
	cout<<"See configuration file for further information on parameters\n\n";

}
