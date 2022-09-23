function [classification,errorRate]=kNNClassifier(training,targets,test,k,testTargets)

% check inputs
if(nargin<4)
    error("The function requires at least 4 arguments");
end

if(size(training,2) ~= size(test,2))
    error("Invalid inputs: the numbers of columns of targets and test must be the same");
end

if(k<=0 || k>size(training,1))
    error("k is not valid");
end

% if k is divisible by the number of classes there might be a tie
if(mod(k,size(unique(targets)))==0)
    warning("k should not be divisible by the number of classes. There might be a tie.");
end

classification = zeros(size(test,1),1);

% Classificate test
for queryPoint=1:size(test,1)
    
    %calculate euclidean norm
    %the fastest function is vecnorm(), which instead of norm() can perform
    %norm on each row of a given matrix.
    %vecnorm(training-test,2,2) means calculate 2-norm on the rows of the
    %resulting matrix
    norms = vecnorm(training-test(queryPoint,:),2,2);
    
    
    %take k minimum-value's indexes inside norms
    [~,minIndexes]=mink(norms,k);
    
    %compute mode of the first k elements in test set
    classification(queryPoint,1)=mode(targets(minIndexes));
end

if(nargin==5)
    %compute error rate (number of errors / m)
    errorRate = sum(classification~=testTargets)/size(test,1);
else
    %error rate can not be computed
    errorRate = [];
end

end