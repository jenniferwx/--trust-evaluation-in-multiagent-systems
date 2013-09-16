function [Net,Attributes,label] = undirect_synthetic_generator(numNodes,alpha,dh,numLabels,attrNoise,numObjs,vocabSize,numAct)
% Generate synthetic trust network based on the synthetic network generation explained 
% in paper "Link-based classification" by Sen and Getoor
%%%%%%%%%INPUT:%%%%%%%%%%%
% numNodes: the number of nodes in the network
% alpha: is a parameter that controls the number of links in the graph (between (0,1]); 
%        roughly the final graph should contain (1/(1-alpha))numNodes number of links.
% dh: the value for homophily choose between (0,1]
% numLabels: the total number of classes in the network
% numObjs: maximum number of words in a node
% vocabSize: the size of node's feature
%%%%%%%%OUTPUT%%%%%%%%%%%%
% Net: the network connectivity matrix
% Attributes: the features asscoiated with each node
% label: the node's label
% Updated by Xi Wang 09/21/2010
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

i = 0;
Net = zeros(numNodes,numNodes,'single');
label = [];
if(isempty(numAct))
    numAct=1;
end
while (i< numNodes)
    r = rand;
    if (r <= alpha && (i~=0 || i ~= 2))
        Net = connectNode(Net,i,numLabels,label,dh,numAct);
    else
        [Net,label] = addNode(Net, i, numLabels,label,dh,numAct);
        i = i+1;
    end
end
label = label';
Attributes = zeros(numNodes,vocabSize);
for i =1:numNodes
    Attributes  = genAttributes(i, numLabels, vocabSize, numObjs, attrNoise, Attributes,label);
end

function [Net,label] = addNode(Net, i, numLabels, label, dh)

mu = [0.9,0.6,0.4,0.3];
sigma = [0.05;0.1;0.1;0.05];
v = i+1;
c = chooseNewNodeClass(numLabels);      %can be used to implement a set of class priors.
%In all our experiments with synthetic data we used a set of uniform class priors.
mlinks = 1; %mlinks controls the number of links a new node can make to the existing network nodes.
label(v) = c;
if v>1
    Cn = chooseClass(c, dh, numLabels, label, v);
    Net = SFNW(Net, mlinks, Cn, v, label, mu, sigma);    
end

function Net = connectNode(Net,i,numLabels,label,dh,numAct)
% the mean and std of the Gaussian distribution for each trustworthiness level:
mu = [0.9,0.6,0.4,0.3]; 
sigma = [0.05;0.1;0.1;0.05];
mlinks = 1; %mlinks controls the number of links a new node can make to the existing network nodes.
if length(label)<numAct
    numAct = length(label);
end

for j = 1: numAct
    if i>2
        v =  random('unid',i);   % randomly chooose a node from G/ i is the current number of nodes
        c = label(v);
        Cn = chooseClass(c, dh , numLabels,label,v);
        Net = SFNW(Net, mlinks,Cn, v,label, mu, sigma);
    end
end


function Attributes = genAttributes(v,numLabels,vocabSize,numObjs,attrNoise,Attributes,label)
% numObjs: maximum number of words in a node
for i =1:numObjs
    % sample r uniformly at random
    %r = random('unid',numNodes)/numNodes;
    r = rand;
    if (r <= attrNoise)
        w = ceil(rand * vocabSize); % add noise
        Attributes(v,w) = 1;
    else
        p = (1 + (label(v)-1))/(1+numLabels);
        w = binornd(vocabSize-1,p); % generate binomial random number
        Attributes(v,w+1) = 1;    % label starts from 0
    end
end

function  Cn = chooseClass(c,dh, numLabels,label,v)
% dh specifying the percentage of a node's neighbor that is of the opposite type
% degree of the candidates (preferental attachement)

tmp =[];
while (isempty(tmp))
    r = rand;
    if (r >= dh )
        if (~isempty(find(label~=c)))
            ww = setdiff(1:numLabels,c);
            while (isempty(tmp))
                w = random('unid',length(ww));
                tmp = find(label == ww(w));
            end
            Cn = ww(w);
        else
            Cn = c;
            l = setdiff(1:length(label),v);
            tmp = find(label(l) == Cn);
        end
    else
        Cn = c;
        l = setdiff(1:length(label),v);
        tmp = find(label(l) == Cn);
    end
end

function c = chooseNewNodeClass(numLabels)

c = random('unid',numLabels);

function Net = SFNW(Net, mlinks, Cn, v,label, mu, sigma) %Scale-Free Network
% modified from the B-A ScaleFree Network ( Barabasi-Albert model)
% Cn is the label of the node to be connected
% mlinks controls the number of links a new node can make to the existing network nodes.

tp = setdiff(1:length(label),v);
label2 = label(tp);
tmp = find(label2 == Cn);
L = length(tmp);   % save the number of nodes with label Cn

if (L < mlinks)
    mlinks = L;
end

sumlinks = sum(sum(Net));

pos = v;
linkage = 0;

% generate mlinks for the given node
while linkage ~= mlinks
    t = ceil(rand * length(tmp)); % the index of the chosen node
    rnode = tp(tmp(t));
    deg = sum(Net(:,rnode))*2;
    rlink = rand * 1;
    act = normrnd(mu(Cn),sigma(Cn));
    if (sumlinks == 0)
        if act >= 0.5
            Net(pos,rnode) = Net(pos,rnode) + 1;
            Net(rnode,pos) = Net(rnode,pos) + 1;
            linkage = linkage +1;
        else
            break;
        end
    elseif (rlink < deg/sumlinks) || (sum(sum(Net(:,tp(tmp))))*2 == 0) %(Net(pos,rnode) ~= 1 && Net(rnode,pos) ~= 1);   %% p = (deg/sumlinks)
        if act >= 0.5
            Net(pos,rnode) = Net(pos,rnode) + 1;
            Net(rnode,pos) = Net(rnode,pos) + 1;
            linkage = linkage + 1;
        else
            break;
        end
    end
end




