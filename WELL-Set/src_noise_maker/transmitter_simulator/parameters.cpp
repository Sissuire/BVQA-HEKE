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

/*!
 *
 *	\brief
 *	First constructor whereby the parameters are passed via command line
 *
 *	\param 
 *	argv, string of input parameters
 *
 *	\author
 *	Matteo Naccari
 *
*/

Parameters::Parameters(char **argv){

	bitstream_original = argv[1];

	bitstream_transmitted = argv[2];

	loss_pattern_file = argv[3];

	packet_type = atoi(argv[4]);

	offset = atoi(argv[5]);

	modality = atoi(argv[6]);

	check_parameters();
}

/*!
 *
 *	\brief
 *	Second constructor whereby the parameters are passed via configuration file
 *
 *	\param 
 *	argv, name of the configuration file
 *
 *	\author
 *	Matteo Naccari
 *
*/

Parameters::Parameters(char *argv){
	string line;
	int i = 0;
	char temp[100];
	ifstream fin;

	fin.open(argv, ifstream::in);

	if(!fin){
		cout<<"Cannot open config file "<<argv<<" abort"<<endl;
		exit(-1);
	}

	while(getline(fin, line, '\n')){		
		if(valid_line(line)){
			switch(i)
			{
				case 0:
					sscanf(line.c_str(), "%s", temp);
					bitstream_original = temp;
					break;
				case 1:
					sscanf(line.c_str(), "%s", temp);
					bitstream_transmitted = temp;
					break;
				case 2:
					sscanf(line.c_str(), "%s", temp);
					loss_pattern_file = temp;
					break;
				case 3:
					sscanf(line.c_str(), "%d", &packet_type);
					break;
				case 4:
					sscanf(line.c_str(), "%d", &offset);
					break;
				case 5:
					sscanf(line.c_str(), "%d", &modality);
					break;
				default:
					cout<<"Something wrong: (?)"<<line<<endl;
			}
			i++;
		}
	}

	fin.close();
	check_parameters();
}

/*!
 *
 *	\brief
 *	It reads a valid line from the configuration file. A valid line is a text line
 *	that does not start with the following characters: #, carriage return, space or 
 *	new line
 *
 *	\param
 *	line a char array containing the current line read from the configuration file
 *
 *	\return
 *	True if the current line is valid line
 *	False otherwise
 *	
 *	\author
 *	Matteo Naccari
*/

bool Parameters::valid_line(string line){
	if(line.length() == 0)
		return 0;
	if(line.at(0) == '\r' || line.at(0) == '#' || line.at(0) == ' ' || line.at(0) == '\n')
		return 0;
	return 1;
}

Parameters::~Parameters(){
}

void Parameters::check_parameters(){

	if(offset < 0){
		cout<<"Warning! Offset = "<<offset<<" is not allowed, set it to zero\n";		
		offset = 0;			
	}
	if(!(0<=modality&&modality<=2)){
		cout<<"Warning! Modality = "<<modality<<" is not allowed, set it to zero\n";
		modality = 0;		
	}

}