README for HS NLP with Python: A hands-on introduction using NLTK:

"ACRIC: Automatic Classification of Radical Ideological Contents (Group Project)"

Lecturer: Detmar Meurers
Winter 2015/16

Deadline: September 30, 2016

------------------------------------------------------------------------------------------
GENERAL INFORMATION
----------------------------------------------
- This folder contains all data concerning the present group project submission, organized in meaningful divisions.

- The following section describes the relevant data for the generation of feature vectors as an input for the WEKA workbench:
		
	- _NLTK_corpus: Contains the documents of the manually created corpus for the ACRIC project, divided into the two classification tasks:
		- non_radical
		- radical
	
	- _NLTK_project: Contains the python code, csv, arff and txt files that are necessary for the application of the classification task:
		- _data_vectors: csv-files with the results of the measures of the code described below
		
		CODE:
		- 01_: formats and preprocesses the respective documents of the corpus in order to be processed by the rest of the code
		- 02_: contains the functions calculating the results for the respective measures --> output: .csv files with results
		- 04_: merges the respective csv-files to one bigger csv file
		- dict_lower.txt, dict_needs.txt, dict_radicalism.txt, dict_functionwords.txt, jihad_verses.txt: dictionaries and lists for the aforementioned python functions
		- arff-files: WEKA-input, constructed by the aforementioned csv-files

------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
authors: Lisa Hiller, Daniela Stier
date: 26/09/2016

