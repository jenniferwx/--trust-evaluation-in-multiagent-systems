%% Example of how to generate synthetic trust network using "undirect_synthetic_generator.m" file

numNodes = 500; % number of nodes in the network
numLabels = 4;  % number of labels (trustworthiness level)
alpha = 0.3;   % the parameter for link density
dh = 0.8;      %  homophily
attrNoise = 0.2; % noise parameter for the agent's static features
vocabSize = 10;  % the number of static features (binary feature vector)
numObjs = 5;    % the maximum number of features can be true (value equals 1)

[Net,Attributes,label] = undirect_synthetic_generator(numNodes,alpha,dh,numLabels,attrNoise,numObjs,vocabSize);
