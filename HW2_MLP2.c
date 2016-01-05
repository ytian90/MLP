// HW2_MLP2.cpp : Defines the entry point for the console application.
//


/*****************************************
*******     This is a simulator for     
*******     backpropagation neural net
*******     Created by Iren Valova    
*******     Copyright 1996-97,        
*******     Tokyo Institute of Technology
*******     Kosugi Lab, ISST             
****************************************/
//#include "stdafx.h"
#include "stdlib.h"
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <time.h>


#define INPUTS		2
#define HIDDEN_MAX	50
#define OUTPUTS		1

#define LEARNING_RATE1				0.6		/* was eta */
#define LEARNING_RATE2				LEARNING_RATE1
#define MOMENTUM					0.8		/* was alpha */
#define MEAN_SQR_ERROR_THRESHOLD	0.10	/* was criterion */
#define MAX_ITERATION				10000

#define TRAINSET_ENTRIES			326
#define TESTSET_ENTRIES				1681

#define RANDOM_WEIGHT() (((double)rand()/(double)INT_MAX)-0.5)

/* outputs for each neuron, all layers */
double o1[INPUTS], o2[HIDDEN_MAX], o3[HIDDEN_MAX], o4[OUTPUTS];
/* weights for each neuron, hidden and output */
double w12[HIDDEN_MAX][INPUTS], w23[HIDDEN_MAX][HIDDEN_MAX], w34[OUTPUTS][HIDDEN_MAX];
/* change in weights for each neuron, hidden and output */
double dw12[HIDDEN_MAX][INPUTS], dw23[HIDDEN_MAX][HIDDEN_MAX], dw34[OUTPUTS][HIDDEN_MAX];
/* previous weight for each neuron, hidden and output */
double olddw12[HIDDEN_MAX][INPUTS], olddw23[HIDDEN_MAX][HIDDEN_MAX], olddw34[OUTPUTS][HIDDEN_MAX];

/* bias weights, hidden and output */
double b2[HIDDEN_MAX], b3[HIDDEN_MAX], b4[OUTPUTS];
/* bias change in weights, hidden and output */
double db2[HIDDEN_MAX], db3[HIDDEN_MAX], db4[OUTPUTS];
/* bias previous weights, hidden and output */
double olddb2[HIDDEN_MAX], olddb3[HIDDEN_MAX], olddb4[OUTPUTS];


double error;
double meanSqrError;

/* input data from file */
double trainSetIn[TRAINSET_ENTRIES][INPUTS];
double trainSetOut[TRAINSET_ENTRIES][OUTPUTS];

double testSetIn[TESTSET_ENTRIES][INPUTS];

double setin[TESTSET_ENTRIES][INPUTS];  /* use the larger of the two, training/test */
double setout[TESTSET_ENTRIES][INPUTS]; /* use the larger of the two, training/test */

int cycle, update;
/*
int numHiddenUnits = 5; // number of hidden neurons
double learningRate = 100; // learning rate
double momentum = 0.1; // WHAT IS THIS????
*/
int numHiddenUnits;
double learningRate;
double momentum;

/* forward declarations */
void initnetwork();
void initset();
void forward(int set);
void backward(int set);
void modifyw();
void testnet();
void dumpNetworkFileData();


/* get the file data for both train and test */
void getFileData()
{
	int i,j;
	FILE *trainFile, *testFile;
	float tmpo;

	/* Read the training file */
	trainFile=fopen("spiral.dat","r");
	if (trainFile == NULL)
	{
		printf("initset: fopen failed, returned NULL\n");
	}

	for(i=0;i<TRAINSET_ENTRIES;i++)
	{
		for(j=0;j<INPUTS;j++)
		{
			fscanf(trainFile,"%f",&tmpo);
			trainSetIn[i][j]=tmpo;
		}

		for(j=0;j<OUTPUTS;j++)
		{
			fscanf(trainFile,"%f",&tmpo);
			trainSetOut[i][j]=tmpo;
		}
	}

	fclose(trainFile);

	/* Read the training file */
	testFile=fopen("spirtest.dat","r");
	if (testFile == NULL)
	{
		printf("initset: fopen failed, returned NULL\n");
	}

	for(i=0;i<TESTSET_ENTRIES;i++)
	{
		for(j=0;j<INPUTS;j++)
		{
			fscanf(testFile,"%f",&tmpo);
			testSetIn[i][j]=tmpo;
		}
	}

	fclose(testFile);
}


/* main MLP entry point */
void mainMLP()
{
	int iteration, set;

	initnetwork();
	initset();

	cycle=0; update=0;

	for(iteration=0;iteration<MAX_ITERATION;iteration++)
	{
		cycle=cycle+1;
		error=0.0;

		for(set=0;set<TRAINSET_ENTRIES;set++)
		{
			forward(set);
			backward(set);
		}

		meanSqrError=error/TRAINSET_ENTRIES/OUTPUTS;

//		if(iteration%10000==0)
//			printf("%d  %f\n",iteration,meanSqrError);

		if(meanSqrError<MEAN_SQR_ERROR_THRESHOLD)
		{
			//printf("\nMLP2 %d, %f, %f\n", numHiddenUnits, learningRate, momentum);
			//printf("!!!!!!End point, meanSqrError = %f, iteration = %d\n", meanSqrError, iteration);
			break;
		}
		else
		{
			modifyw();
		}
	}

	printf("meanSqrError = %f, iteration = %d\n", meanSqrError, iteration);

	testnet();
}

/* initializes all random weights/data */
void initnetwork()
{
	int i,j;

	/* initialize random weights */
	for(i=0;i<numHiddenUnits;i++)
		for(j=0;j<INPUTS;j++)
			w12[i][j]=RANDOM_WEIGHT();

	for(i=0;i<numHiddenUnits;i++)
		for(j=0;j<numHiddenUnits;j++)
			w23[i][j]=RANDOM_WEIGHT();

	for(i=0;i<OUTPUTS;i++)
		for(j=0;j<numHiddenUnits;j++)
			w34[i][j]=RANDOM_WEIGHT();

	for(i=0;i<numHiddenUnits;i++)
	{
		b2[i]=RANDOM_WEIGHT();
		db2[i]=0.0;
		olddb2[i]=0.0;
	}

	for(i=0;i<numHiddenUnits;i++)
	{
		b3[i]=RANDOM_WEIGHT();
		db3[i]=0.0;
		olddb3[i]=0.0;
	}

	for(i=0;i<OUTPUTS;i++)
	{
		b4[i]=RANDOM_WEIGHT();
		db4[i]=0.0;
		olddb4[i]=0.0;
	}
}

/* reads input/output data from train file */
void initset()
{
	int i,j;

	/* copy over the train data */
	for(i=0;i<TRAINSET_ENTRIES;i++)
	{
		for(j=0;j<INPUTS;j++)
		{
			setin[i][j]=trainSetIn[i][j];
		}

		for(j=0;j<OUTPUTS;j++)
		{
			setout[i][j]=trainSetOut[i][j];
		}
	}

	//dumpNetworkFileData();
}

/* print the input file data */
void dumpNetworkFileData()
{
	int i,j;

	/* loop through all entries */
	for(i=0;i<TRAINSET_ENTRIES;i++)
	{
		printf("%04d: ", i);
		/* output all inputs per entry */
		for(j=0;j<INPUTS;j++)
		{
			printf("%f, ", setin[i][j]);
		}
		/* ouput all outputs per entry */
		for(j=0;j<OUTPUTS;j++)
		{
			printf("%f", setout[i][j]);
		}

		printf("\n");
	}
}

// ??? Change the sigmoid activation function
/* sigmoid activation function */
float sigmoid(float sum)
{
	if(sum<(-80)) return(0.0);
	else if(sum>80) return(1.0);
		else return(1.0/(1.0+exp(-sum)));
}

/* propagate signals forward */
void forward(int set)
{
	int i,j;
	double sum;

	for(i=0;i<INPUTS;i++)
		o1[i]=setin[set][i];



	for(i=0;i<numHiddenUnits;i++)
	{
		sum=0.0;

		for(j=0;j<INPUTS;j++)
			sum=sum+w12[i][j]*o1[j];

		sum=sum-b2[i];
		o2[i]=sigmoid(sum);
	}



	for(i=0;i<numHiddenUnits;i++)
	{
		sum=0.0;

		for(j=0;j<numHiddenUnits;j++)
			sum=sum+w23[i][j]*o2[j];

		sum=sum-b3[i];
		o3[i]=sigmoid(sum);
	}



	for(i=0;i<OUTPUTS;i++)
	{
		sum=0.0;

		for(j=0;j<numHiddenUnits;j++)
			sum=sum+w34[i][j]*o3[j];

		sum=sum-b4[i];
		o4[i]=sigmoid(sum);
	}



	for(i=0;i<OUTPUTS;i++)
		error=error+(o4[i]-setout[set][i])*(o4[i]-setout[set][i]);
}

/* propagate signals backward */
void backward(int set)
{
	int i,j;
	double sum;
	double delta4[OUTPUTS],delta3[HIDDEN_MAX],delta2[HIDDEN_MAX];

	for(i=0;i<OUTPUTS;i++)
		delta4[i]=(o4[i]-setout[set][i])*o4[i]*(1.0-o4[i]);

	for(i=0;i<OUTPUTS;i++)
		for(j=0;j<numHiddenUnits;j++)
			dw34[i][j]=dw34[i][j]-learningRate*delta4[i]*o3[j];

	for(i=0;i<OUTPUTS;i++)
		db4[i]=db4[i]-learningRate*delta4[i]*(-1.0);




	for(i=0;i<numHiddenUnits;i++)
	{
		sum=0.0;

		for(j=0;j<OUTPUTS;j++)
			sum=sum+delta4[j]*w34[j][i];

		delta3[i]=o3[i]*(1.0-o3[i])*sum;
	}

	for(i=0;i<numHiddenUnits;i++)
		for(j=0;j<numHiddenUnits;j++)
			dw23[i][j]=dw23[i][j]-learningRate*delta3[i]*o2[j];

	for(i=0;i<numHiddenUnits;i++)
		db3[i]=db3[i]-learningRate*delta3[i]*(-1.0);




	for(i=0;i<numHiddenUnits;i++)
	{
		sum=0.0;

		for(j=0;j<numHiddenUnits;j++)
			sum=sum+delta3[j]*w23[j][i];

		delta2[i]=o2[i]*(1.0-o2[i])*sum;
	}

	for(i=0;i<numHiddenUnits;i++)
		for(j=0;j<INPUTS;j++)
			dw12[i][j]=dw12[i][j]-learningRate*delta2[i]*o1[j];

	for(i=0;i<numHiddenUnits;i++)
		db2[i]=db2[i]-learningRate*delta2[i]*(-1.0);
}

/* modify weights */
void modifyw()
{
	int i,j;

	for(i=0;i<numHiddenUnits;i++)
		for(j=0;j<INPUTS;j++)
		{
			w12[i][j]=w12[i][j]+dw12[i][j]+momentum*olddw12[i][j];
			olddw12[i][j]=dw12[i][j];
			dw12[i][j]=0.0;
			update=update+1;
		}

	for(i=0;i<numHiddenUnits;i++)
	{
		b2[i]=b2[i]+db2[i]+momentum*olddb2[i];
		olddb2[i]=db2[i];
		db2[i]=0.0;
	}





	for(i=0;i<numHiddenUnits;i++)
		for(j=0;j<numHiddenUnits;j++)
		{
			w23[i][j]=w23[i][j]+dw23[i][j]+momentum*olddw23[i][j];
			olddw23[i][j]=dw23[i][j];
			dw23[i][j]=0.0;
			update=update+1;
		}

  for(i=0;i<numHiddenUnits;i++)
  {
    b3[i]=b3[i]+db3[i]+momentum*olddb3[i];
    olddb3[i]=db3[i];
    db3[i]=0.0;
  }





	for(i=0;i<OUTPUTS;i++)
		for(j=0;j<numHiddenUnits;j++)
		{
			w34[i][j]=w34[i][j]+dw34[i][j]+momentum*olddw34[i][j];
			olddw34[i][j]=dw34[i][j];
			dw34[i][j]=0.0;
			update=update+1;
		}

	for(i=0;i<OUTPUTS;i++)
	{
		b4[i]=b4[i]+db4[i]+momentum*olddb4[i];
		olddb4[i]=db4[i];
		db4[i]=0.0;
	}
}

// WHY DO WE PRINT OUT THE FILES???
/* test the network */
void testnet()
{
	FILE *outFile;
	char outFileName[256];
	int i, j, k;
	float tmpin, tmpout;
	int count;


	sprintf(outFileName, "MLP2_%d_%02f_%02f.txt", numHiddenUnits, learningRate, momentum); 

	for(i=0;i<TESTSET_ENTRIES;i++)
	{
		for(j=0;j<INPUTS;j++)
		{
			setin[i][j]=testSetIn[i][j];
		}
	}

	/* this file is used to dump the output data */
	outFile = fopen(outFileName, "w");

	count=0;

	for(i=0;i<TESTSET_ENTRIES;i++)
	{

		for(k=0;k<INPUTS;k++)
		{
			fprintf(outFile,"%f, ", setin[i][k]);
		}

		/* push test data into network */
		forward(i);

		for(j=0;j<OUTPUTS;j++)
		{
			if (j==(OUTPUTS-1))
				fprintf(outFile, "%f", o4[j]);
			else
				fprintf(outFile, "%f, ", o4[j]);
		}

		fprintf(outFile, "\n");
	}

	//printf("Number of epochs %d, weight updates %d\n",cycle,update);
}



/* main entry point */
int main(int argc, const char * argv[])
{
	printf("Start\n");

	/* init random */
	srand(time(0));

	getFileData();
/*
	for (numHiddenUnits = 5; numHiddenUnits <= 50; numHiddenUnits+= 5) // increase the # of hidden neurons
	{
		for (learningRate = 0.01; learningRate <= .09; learningRate+= 0.01) // increase the learning rate
		{
			for (momentum = 0.01; momentum <= 0.09; momentum+= 0.01) // is this the hidden layers???
			{
				printf("\nMLP %d, %f, %f\n", numHiddenUnits, learningRate, momentum);
				mainMLP();
			}
		}
	}
*/
/*
  for (numHiddenUnits = 5; numHiddenUnits <= 20; numHiddenUnits += 5) {
    for (learningRate = 0.01; learningRate <= 0.09; learningRate += 0.03) {
      for (momentum = 0.01; momentum <= 0.09; momentum += 0.03) {
        printf("\nMLP %d, %f, %f\n", numHiddenUnits, learningRate, momentum);
        mainMLP();
      }
    }
  }
  */
  learningRate = 0.05;
  momentum = 0;
  for (numHiddenUnits = 5; numHiddenUnits <= 45; numHiddenUnits += 10) {
    printf("\nMLP %d, %f, %f\n", numHiddenUnits, learningRate, momentum);
    mainMLP();
  }



	printf("Finish\n");
}
